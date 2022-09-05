""" Preproccessing CXR14 Datasets for RetinaNet
    ChristianNeil
    2022/7/22
"""
import csv
import sys
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("./retinanet")
from retinanet.chest_detection import detect_chest, get_model


CXR14_IMG_PATH = "F:\\dataset\\CXR14\\images\\"
CXR14_SEG_PATH = "F:\\dataset\\CXR14\\segmentations\\"
CXR14_BOX_PATH = "F:\\dataset\\CXR14\\BBox_List_2017.csv"
CXR14_ANO_PATH = "F:\\dataset\\CXR14\\Data_Entry_2017_v2020.csv"

CSV_OUTPUT_PATH = "./dataset/"
RETINANET_MODEL_PATH = "./retinanet/models/trained_without_neg_sample_res101/csv_retinanet_epoch3.pt"

DISEASE_CLASSES = [
    ["Atelectasis", 0],
    ["Cardiomegaly", 1],
    ["Effusion", 2],
    ["Infiltration", 3],
    ["Mass", 4],
    ["Nodule", 5],
    ["Pneumonia", 6],
    ["Pneumothorax", 7],
    ["Consolidation", 8],
    ["Edema", 9],
    ["Emphysema", 10],
    ["Fibrosis", 11],
    ["Pleural_Thickening", 12],
    ["Hernia", 13],
    ["No Finding", 14]
]


def create_dataset(path, annos_train, annos_test, annos_val):
    """ Generate CSV file for RetinaNet training
    """
    with open(path + "train.csv", mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(annos_train)

    if annos_val is not None:
        with open(path + "val.csv", mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(annos_val)

    if annos_test is not None:
        with open(path + "test.csv", mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(annos_test)


def create_bbox_dataset(path, annos_bbox):
    """ Generate CSV file for RetinaNet training
    """
    with open(path + "val_bbox.csv", mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(annos_bbox)


def preproc_cxr14_dataset():
    """ preproccessing CXR14 dataset to CSV file
    """
    annos = []
    annos_val = []
    annos_test = []
    dataset = [annos_val, annos_test, annos]
    class_counter = {}
    class_map = {}

    model = get_model(RETINANET_MODEL_PATH)

    for c in DISEASE_CLASSES:
        class_counter[c[0]] = 0
        class_map[c[0]] = c[1]

    with open(CXR14_ANO_PATH) as f:

        f_csv = csv.reader(f)
        next(f_csv)
        items = [x for x in csv.reader(f)]

        for row in tqdm(items):
            row = tuple(str(val) for val in row)

            img_path = CXR14_IMG_PATH + row[0]
            labels = row[1].split('|')

            #Divide datasets by 7 :2 :1
            prob = random.randint(0, 9)
            if prob == 0:
                index = 0
            elif prob < 3:
                index = 1
            else:
                index = 2

            result = detect_chest(img_path, model, 0.999)

            if result:
                chest_roi = result[1]
                one_hot = [0 for x in range(len(DISEASE_CLASSES) - 1)]
                for label in labels:
                    if label != "No Finding":
                        one_hot[class_map[label]] = 1
                    target = [row[0]]
                    target.extend(chest_roi)
                    target.extend(one_hot)
                    dataset[index].append(target)
                    class_counter[label] += 1

        print(class_counter)
    create_dataset(CSV_OUTPUT_PATH, annos, annos_test, annos_val)


def preproc_cxr14_bbox_dataset():
    """ preproccessing CXR14 dataset to CSV file
    """
    annos = {}
    class_counter = {}
    class_map = {}
    dataset = []

    model = get_model(RETINANET_MODEL_PATH)

    for c in DISEASE_CLASSES:
        class_counter[c[0]] = 0
        class_map[c[0]] = c[1]

    with open(CXR14_BOX_PATH) as f:

        f_csv = csv.reader(f)
        next(f_csv)
        items = [x for x in csv.reader(f)]

        # collect meta datas
        for row in items:
            row = list(str(val) for val in row)
            img_idx = row[0]
            img_path = CXR14_IMG_PATH + img_idx

            label = row[1]
            boxes = [int(float(x)) for x in row[2:6]]
            meta = {"path": img_path, "label": label, "boxes": boxes}

            if img_idx not in annos:
                annos[img_idx] = [meta]
            else:
                annos[img_idx].append(meta)

        # extract ROI
        for key, values in tqdm(annos.items()):
            result = detect_chest(CXR14_IMG_PATH + key, model, 0.999)
            if result:
                chest_roi = result[1]
                for meta in values:
                    label = meta["label"]
                    if label != "No Finding":
                        if label == "Infiltrate":
                            label = "Infiltration"
                        class_label = [class_map[label]]
                    target = [key]
                    target.extend(chest_roi)
                    x = (meta["boxes"][0]) - chest_roi[0]
                    y = (meta["boxes"][1]) - chest_roi[2]
                    w = (meta["boxes"][2])
                    h = (meta["boxes"][3])
                    boxes = [x, x + w, y, y + h]
                    target.extend(boxes)
                    target.extend(class_label)
                    dataset.append(target)
                    class_counter[label] += 1
                    # plt.figure(figsize=(5, 5))
                    # plt.imshow(result[0])
                    # ax = plt.gca()
                    # ax.add_patch(
                    #     plt.Rectangle((boxes[0], boxes[2]), boxes[1]-boxes[0], boxes[3]-boxes[2], color="red", fill=False,
                    #                   linewidth=2))
                    # print(f"{key} {boxes}")
                    # plt.show()
                    # plt.waitforbuttonpress()


        print(class_counter)
    create_bbox_dataset(CSV_OUTPUT_PATH, dataset)


def dataset_clean():
    listsave = []

    # ---- Open file, get image paths and labels
    fileDescriptor = open("D:\\projects\\儿童医院\\参考代码\\ChexNet\\CheXNet++\\dataset\\test_1.txt", "r")
    fileDescriptor2 = open("D:\\projects\\儿童医院\\参考代码\\ChexNet\\CheXNet++\\dataset\\all.csv", "r")

    dictCsv = {}
    line = True
    while line:
        line = fileDescriptor2.readline()
        # --- if not empty
        if line:
            lineItems = line.split(',')
            dictCsv[lineItems[0]] = line

    line = True
    while line:
        line = fileDescriptor.readline()
        # --- if not empty
        if line:
            lineItems = line.split(' ')
            if lineItems[0] in dictCsv:
                listsave.append(dictCsv[lineItems[0]])

    with open("D:\\projects\\儿童医院\\参考代码\\ChexNet\\CheXNet++\\dataset\\test.csv", "w") as f:
        for item in listsave:
            f.write(item)

    fileDescriptor.close()


if __name__ == '__main__':
    # preproc_cxr14_dataset()
    preproc_cxr14_bbox_dataset()
    # dataset_clean()
