import os
import sys
import cv2
import time
import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from model.Losses import Losses
from model.ChexClassifier import ChexClassifier
from model.utils.Logger import build_logger
from model.utils.Wandb import WandbRecoder
from model.utils.Metric import computeAUCROC, computeIoU, computeIoBB
from model.dataset.DataTransforms import DataTransforms
from model.dataset.DatasetGenerator import DatasetGenerator, BBoxDatasetGenerator, BBoxCollater


class ChexnetTrainer(object):

    currEpoch = 0
    startTime = 0
    logger = None

    @staticmethod
    def train(cfg):
        # set mode
        cfg.mode = "train"

        # network architecture
        model = ChexClassifier(cfg).cuda()

        # data transforms
        transSeqTrain = DataTransforms(cfg).compile("train")
        transSeqVal = DataTransforms(cfg).compile("val")

        # dataset builders
        datasetTrain = DatasetGenerator(pathImageDirectory=[cfg.dataset.images_path, cfg.dataset.segment_path],
                                        pathDatasetFile=cfg.dataset.file_train, transform=transSeqTrain)
        datasetVal = DatasetGenerator(pathImageDirectory=[cfg.dataset.images_path, cfg.dataset.segment_path],
                                      pathDatasetFile=cfg.dataset.file_val, transform=transSeqVal)
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=cfg.train.batch_size,
                                     shuffle=True,  num_workers=cfg.dataset.num_workers, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=cfg.val_test.batch_size,
                                   shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)

        # optimizer & scheduler
        optimizer = ChexnetTrainer.build_optimizer_from_cfg(model.parameters(), cfg)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')

        # loss
        loss = ChexnetTrainer.build_loss_from_cfg(cfg)

        # load checkpoint
        epochStart = ChexnetTrainer.load_checkpoint_from_cfg(cfg, model, optimizer)

        # set logger
        ChexnetTrainer.logger = ChexnetTrainer.build_logger_from_cfg(cfg)
        ChexnetTrainer.logger.info(cfg)

        # set wandb
        wandbRecoder = WandbRecoder(cfg, model)

        # set random seed
        nSeed = time.time() * 1000
        random.seed(nSeed)
        ChexnetTrainer.logger.info("Seed is " + str(nSeed))

        # train the network
        maxAUC = 0.0
        ChexnetTrainer.startTime = time.time()
        for epochID in range(epochStart, cfg.train.max_epoch):

            ChexnetTrainer.currEpoch = epochID
            ChexnetTrainer.epoch_train(model, dataLoaderTrain, optimizer, loss, cfg)
            lossVal, lossMean, aucVal, aucMean, accVal, accMean, bThr = ChexnetTrainer.epoch_val(model, dataLoaderVal, loss, cfg)
            scheduler.step(lossMean)

            # save checkpoints
            pthDict = \
                {'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_auc': maxAUC, 'optimizer':optimizer.state_dict()}
            if aucMean > maxAUC:
                maxAUC = aucMean
                torch.save(pthDict, os.path.join(cfg.train.model_save_path, 'best_model' + str(epochID + 1) + '-' + str(maxAUC) + '.pth.tar'))
                torch.save(pthDict, os.path.join(cfg.train.model_save_path, 'best_model.pth.tar'))
            torch.save(pthDict, os.path.join(cfg.train.model_save_path, 'newest_model.pth.tar'))

            # log
            summary = "Epoch:{} | meanLoss:{} |  meanAcc:{} | meanAuc:{} bestAUC:{} \n\t  AUC :{} \n\t  ACC :{} \n\t  LOSS:{} \n\t bThr:{}"\
                .format(epochID + 1, lossMean, accMean, aucMean, maxAUC, aucVal, accVal, lossVal, bThr)
            ChexnetTrainer.logger.info(summary)

            # wandb record one epoch
            wandbRecoder.record_epoch(epochID + 1, lossMean, accMean, aucMean)


    @staticmethod
    def epoch_train(model, dataLoader, optimizer, loss, cfg):
        model.train()

        lossHist = []
        gcl = 0.0
        dtime = 0

        progressBar = tqdm.tqdm(dataLoader, colour='white', file=sys.stdout)
        for batchID, sample in enumerate (progressBar):
            progressBar.write(' Epoch: {} | Global loss: {} | Running loss: {:1.5f} | Time: {}h {}m {}s '.format(
                    ChexnetTrainer.currEpoch, gcl, np.mean(lossHist), int(dtime / 3600), int((dtime / 60) % 60),
                    dtime % 60)) if dtime > 0 else None

            target, image, segment = sample["lab"].cuda(), sample["img"].cuda(), sample["seg"].cuda()

            pred, logitMap = model(image, segment)

            lossValue, _ = loss(pred=pred, gt=target, heatmaps=logitMap, seg=segment)

            optimizer.zero_grad()
            lossSum = lossValue[0] + cfg.train.scl_alpha * lossValue[1]
            lossSum.backward()
            optimizer.step()

            gcl = lossValue.detach().cpu().numpy()
            lossHist.append(gcl)
            dtime = int(time.time() - ChexnetTrainer.startTime)


    @staticmethod
    def epoch_val(model, dataLoader, loss, cfg):
        model.eval()

        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        outLossItem = np.zeros(cfg.dataset.num_classes)

        with torch.no_grad():
            progressBar = tqdm.tqdm(dataLoader, colour='green', file=sys.stdout)
            for batchID, sample in enumerate(progressBar):

                target, image, segment = sample["lab"].cuda(), sample["img"].cuda(), sample["seg"].cuda()

                pred, logitMap = model(image)

                _, lossIdv = loss(pred=pred, gt=target, heatmaps=logitMap, seg=segment)

                outLossItem += lossIdv[0]
                outGT = torch.cat((outGT, target), 0)
                outPRED = torch.cat((outPRED, torch.cat(pred, 1)), 0)

            lossIndividual = outLossItem / len(dataLoader)
            lossMean = lossIndividual.mean()

            aucIndividual, accIndividual, bestThreshholds = \
                computeAUCROC(outGT, outPRED, cfg.dataset.num_classes)

            aucMean = np.array(aucIndividual).mean()
            accMean = np.array(accIndividual).mean()

        return lossIndividual.tolist(), lossMean, aucIndividual, aucMean, accIndividual, accMean, bestThreshholds


    @staticmethod
    def test(cfg):
        # set mode
        cfg.mode = "test"

        # network architecture
        model = ChexClassifier(cfg).cuda()

        # data transforms
        transformSequence = DataTransforms(cfg).compile("test")

        # dataset builders
        datasetTest = DatasetGenerator(pathImageDirectory=cfg.dataset.images_path, pathDatasetFile=cfg.dataset.file_test,
                                      transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=cfg.val_test.batch_size, shuffle=False,
                                   num_workers=cfg.dataset.num_workers, pin_memory=True)

        # load checkpoint
        ChexnetTrainer.load_checkpoint_from_cfg(cfg, model)

        # set ChexnetTrainer.logger
        ChexnetTrainer.logger = ChexnetTrainer.build_logger_from_cfg(cfg)
        ChexnetTrainer.logger.info(cfg)

        # start test
        model.eval()
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

        with torch.no_grad():
            for i, sample in enumerate (tqdm.tqdm(dataLoaderTest, colour='green')):

                target, input = sample["lab"], sample["img"]

                varInput = torch.autograd.Variable(input).cuda()
                varTarget = torch.autograd.Variable(target).cuda()
                varOutput, logitMap = model(varInput)

                outGT = torch.cat((outGT, varTarget), 0)
                outPRED = torch.cat((outPRED, torch.cat(varOutput, 1)), 0)

            aucIndividual, accIndividual, bestThreshholds = \
                computeAUCROC(outGT, outPRED, cfg.dataset.num_classes)

            aucMean = np.array(aucIndividual).mean()
            accMean = np.array(accIndividual).mean()

        summary = "meanAUC:{} AUC:{}\r\n meanACC:{} ACC:{}\r\n BestThresholds:{}"\
            .format(aucMean, aucIndividual, accMean, accIndividual, bestThreshholds)

        ChexnetTrainer.logger.info(summary)

        return aucIndividual, aucMean


    @staticmethod
    def heatmap(img_path, threshholds, cfg):
        # set mode
        cfg.mode = "test"

        # network architecture
        model = ChexClassifier(cfg).cuda()

        # load checkpoint
        ChexnetTrainer.load_checkpoint_from_cfg(cfg, model)

        # data transforms
        transformSequence = DataTransforms(cfg).compile("test")

        # load retinaNet to detect chest ROI
        sys.path.append("./retinanet")
        from retinanet.chest_detection import detect_chest, get_model
        retinanet = get_model(cfg.val_test.retinanet_path)
        result = detect_chest(img_path, retinanet, 0.999)
        if result:
            chest_roi = result[1]
        else:
            raise ValueError(f"Detect image {img_path} chest ROI failed.")

        # load image
        imageData = Image.open(img_path).convert('RGB')
        item = {
            "img": imageData,
            "seg": None,
            "roi": chest_roi
        }

        x1, x2, y1, y2 = chest_roi
        oriImage = torch.from_numpy(np.array(F.crop(imageData, y1, x1, y2 - y1, x2 - x1))).unsqueeze_(0)
        imageData = transformSequence(item)["img"].unsqueeze_(0)

        # start test
        model.eval()
        with torch.no_grad():

            oriInput = torch.autograd.Variable(oriImage).cuda()
            oriInput = transforms.Resize([imageData.shape[2], imageData.shape[3]])(oriInput.permute(0, 3, 1, 2))
            oriInput = oriInput.squeeze(0).permute(1, 2, 0).cpu().data.numpy()

            varInput = torch.autograd.Variable(imageData).cuda()
            varOutput, logitMap = model(varInput)

            count = 1
            plt.figure(figsize=(20, 20))
            plt.subplot(3, 5, count)
            plt.title("origin", pad=20)
            plt.imshow(oriInput)
            for index in range(cfg.dataset.num_classes):
                score = varOutput[index].view(-1)
                score = torch.sigmoid(score)

                if score.ge(threshholds[index]) == 1:
                    count += 1

                    #--- get attention
                    probHmap = torch.sigmoid(logitMap[index])
                    weightMap = probHmap / probHmap.sum(dim=[2, 3], keepdim=True)
                    weightMap = transforms.Resize([imageData.shape[2], imageData.shape[3]])(weightMap)
                    weightMap = weightMap.squeeze(0).permute(1, 2, 0).cpu().data.numpy()

                    #--- generate heatmap
                    plt.subplot(3, 5, count)
                    plt.title(cfg.dataset.class_names[index] + " " + str(score.item()), pad=20)
                    heatmap = weightMap * oriInput
                    cam = heatmap / np.max(heatmap)
                    cam = cv2.applyColorMap(np.uint8(256 * cam), cv2.COLORMAP_JET)
                    cam = cam[:, :, ::-1]
                    heatmap = weightMap * cam
                    heatmap = oriInput/255.0 + heatmap
                    plt.imshow(heatmap)
                    # plt.imshow(cam)

            plt.tight_layout(pad=1.08)
            plt.waitforbuttonpress()

        return heatmap


    @staticmethod
    def localization(iobbThreshold, cfg, showImg=True):
        # set mode
        cfg.mode = "test"

        # network architecture
        model = ChexClassifier(cfg).cuda()

        # load checkpoint
        ChexnetTrainer.load_checkpoint_from_cfg(cfg, model)

        # data transforms
        transformSequence = DataTransforms(cfg).compile("test")

        # dataset builders
        datasetTest = BBoxDatasetGenerator(pathImageDirectory=cfg.dataset.images_path, pathDatasetFile=cfg.dataset.file_bbox,
                                      transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=cfg.val_test.batch_size, shuffle=False,
                                   num_workers=cfg.dataset.num_workers, collate_fn=BBoxCollater)

        # start test
        listIoBB = [np.array([])] * cfg.dataset.num_classes
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm.tqdm(dataLoaderTest, colour='green')):
                oriInputBatch = sample["cop"]
                varInputBatch = torch.autograd.Variable(sample["img"]).cuda()
                varOutputBatch, logitMapBatch = model(varInputBatch)

                for nItem in range(len(oriInputBatch)):
                    oriInput = oriInputBatch[nItem]
                    count = 1
                    for nClass in range(cfg.dataset.num_classes):
                        score = varOutputBatch[nClass][nItem].view(-1)
                        score = torch.sigmoid(score)
                        if nClass in sample["box"][nItem]:
                            # get attention
                            probHmap = torch.sigmoid(logitMapBatch[nClass][nItem])
                            weightMap = probHmap / probHmap.sum(dim=[1, 2], keepdim=True)
                            weightMap = transforms.Resize([oriInput.shape[0], oriInput.shape[1]])(weightMap)
                            weightMap = weightMap.permute(1, 2, 0).cpu().data.numpy()

                            # get ground truth box [x1,y1,x2,y2]
                            gtBox = np.array([[sample["box"][nItem][nClass][0],
                                    sample["box"][nItem][nClass][2],
                                    sample["box"][nItem][nClass][1],
                                    sample["box"][nItem][nClass][3]]])

                            # get pseudo bbox [x1,y1,x2,y2]
                            pseudoBox = []
                            _, bWeightMap = cv2.threshold(weightMap, np.max(weightMap) / 2, 1.0, cv2.THRESH_BINARY)
                            contours, hierarchy = cv2.findContours((bWeightMap * 255).astype("uint8"), cv2.RETR_TREE,
                                                                   cv2.CHAIN_APPROX_NONE)
                            for c in contours:
                                x, y, w, h = cv2.boundingRect(c)
                                pseudoBox.append([x, y, x + w, y + h])
                            pseudoBox = np.array(pseudoBox)

                            # generate heatmap
                            if showImg:
                                if count == 1:
                                    plt.figure(figsize=(10, 5))
                                    plt.subplot(2, 3, count)
                                    plt.title(os.path.basename(sample["pth"][nItem]), pad=20)
                                    plt.imshow(oriInput)
                                count += 1
                                plt.subplot(2, 3, count)
                                plt.title(cfg.dataset.class_names[nClass] + " " + str(score.item()), pad=20)
                                heatmap = weightMap * oriInput
                                cam = heatmap / np.max(heatmap)
                                cam = cv2.applyColorMap(np.uint8(256 * cam), cv2.COLORMAP_JET)
                                cam = cam[:, :, ::-1]
                                heatmap = weightMap * cam
                                heatmap = oriInput/255.0 + heatmap
                                plt.imshow(heatmap)
                                # plt.imshow(cam)
                                ax = plt.gca()
                                ax.add_patch(plt.Rectangle(
                                    (gtBox[0][0], gtBox[0][1]), gtBox[0][2]-gtBox[0][0], gtBox[0][3]-gtBox[0][1],
                                    color="red", fill=False, linewidth=1))
                                for box in pseudoBox:
                                    ax.add_patch(plt.Rectangle(
                                        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                        color="blue", fill=False, linewidth=1))
                            # calculate IoBB
                            iobb = computeIoBB(gtBox, pseudoBox)
                            listIoBB[nClass] = np.append(listIoBB[nClass], 1.0 if np.max(iobb) > iobbThreshold else 0.0)
                plt.show()
                # plt.waitforbuttonpress()

        print("IOBB:")
        for n in range(8):
            print(cfg.dataset.class_names[n] + ": " + str(np.mean(listIoBB[n])))


    @staticmethod
    def build_optimizer_from_cfg(params, cfg):
        if cfg.train.optimizer == 'SGD':
            return SGD(params, lr=cfg.train.lr, momentum=cfg.train.momentum,
                       weight_decay=cfg.train.weight_decay)
        elif cfg.train.optimizer == 'Adadelta':
            return Adadelta(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        elif cfg.train.optimizer == 'Adagrad':
            return Adagrad(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        elif cfg.train.optimizer == 'Adam':
            return Adam(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        elif cfg.train.optimizer == 'RMSprop':
            return RMSprop(params, lr=cfg.train.lr, momentum=cfg.train.momentum,
                           weight_decay=cfg.train.weight_decay)
        else:
            raise Exception('Unknown optimizer : {}'.format(cfg.train.optimizer))


    @staticmethod
    def build_loss_from_cfg(cfg):
        losses = Losses(cfg)
        losses.compile()
        return losses


    @staticmethod
    def build_logger_from_cfg(cfg):
        log_name = time.strftime("_%m-%d_%Hh%Mm%Ss", time.localtime())
        log_name = cfg.mode + log_name + ".txt"
        logger = build_logger(log_name, cfg.logger.log_path+"heatmap_size_"+str(cfg.model.heatmap_size), cfg.logger.enable, when=cfg.logger.Rotating)

        return logger


    @staticmethod
    def load_checkpoint_from_cfg(cfg, model, optimizer=None):
        epochStart = 0
        if cfg.mode == "train":
            model_path = cfg.model.check_point
        else:
            model_path = cfg.val_test.model_path

        if model_path:
            modelCheckpoint = torch.load(model_path)
            model.load_state_dict(modelCheckpoint['state_dict'])
            if cfg.mode == "train":
                optimizer.load_state_dict(modelCheckpoint['optimizer'])
                epochStart = modelCheckpoint['epoch']
            print(f"\nload checkpoint from {model_path} epoch {epochStart}\n")

        return epochStart
