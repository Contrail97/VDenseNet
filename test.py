import torch

if __name__ =="__main__":
    # torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
    #             'optimizer': optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
    pth = torch.load("D:\\projects\\儿童医院\\参考代码\\CheXNet-master\\m-11082022-113043.pth.tar")

    print("epoch:", pth["epoch"], " best_loss:", pth["best_loss"])