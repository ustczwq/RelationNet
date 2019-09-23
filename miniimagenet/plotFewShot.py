import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import MetaTask as mt
import os
import math
import argparse
from scipy import stats
import matplotlib.pyplot as plt
from PIL import Image
import cv2


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=64)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=5)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=1)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=10)
parser.add_argument("-e", "--episode", type=int, default=10)
parser.add_argument("-t", "--test_episode", type=int, default=600)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
HIDDEN_UNIT = args.hidden_unit


def getImgsOrderByLabels(rows, labels):
    order = [[] for i in range(rows)]
    columns = [0 for i in range(rows)]
    maxCol = 0

    for i, l in enumerate(labels):
        order[l].append(i)
        columns[l] += 1
        if columns[l] > maxCol:
            maxCol = columns[l]

    print(order)
    print(labels)

    return order, maxCol


def imgRestore(img, mean, std):
    arr = img
    h, w, _ = np.shape(arr)
    for x in range(h):
        for y in range(w):
            arr[x, y] *= std
            arr[x, y] += mean

    return np.clip(arr, 0, 1)


def tensor2Imgs(tensor):
    imgs = tensor.cpu().numpy()
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    for i, img in enumerate(imgs):
        imgs[i] = imgRestore(
            img,
            mean=[0.92206, 0.92206, 0.92206],
            std=[0.08426, 0.08426, 0.08426]
        )
    return np.uint8(255 * imgs)


def getCamOnImgs(cams, imgs, alpha=0.5):
    maps = []
    cmap = plt.get_cmap('jet')
    for i, img in enumerate(imgs):
        cam = cv2.resize(
            cams[i],
            (img.shape[0], img.shape[1]),
            interpolation=cv2.INTER_CUBIC
        )
        cam = np.uint8(255 * cmap(cam))
        cam = Image.fromarray(cam).convert('RGBA')
        img = Image.fromarray(img).convert('RGBA')
        maps.append(Image.blend(img, cam, alpha))
    return maps


def plotOneShotImgs(sampleImgs, testImgs, rows, labels):
    order, cols = getImgsOrderByLabels(rows, labels)

    fig, axs = plt.subplots(rows, cols + 1)
    for row, ax in enumerate(axs):
        ax[0].imshow(sampleImgs[row])
        for col, num in enumerate(order[row]):
            ax[col + 1].imshow(testImgs[num])

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set(aspect=1, adjustable='box')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def meanConfidenceInterval(data, confidence=0.95):
    arr = 1.0 * np.array(data)
    n = len(arr)
    mean, sem = np.mean(arr), stats.sem(arr)
    h = sem * stats.t._ppf((1 + confidence)/2., n - 1)
    return mean, h


class GradCAM(object):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, model):
        self.model = model.to(self.device)

    def hookLayer(self, layerName, hookNum):
        self.conv = [[] for i in range(hookNum)]
        self.grad = [[] for i in range(hookNum)]
        self.forwardIdx = 0
        self.backwardIdx = 0

        def forwardHook(module, gradIn, gradOut):
            self.conv[self.forwardIdx] = gradOut.detach().cpu().numpy()
            self.forwardIdx += 1
            self.forwardIdx %= hookNum
            # print(gradIn[0].size(), gradOut.size(), 'hook forward++++')

        def backwardHook(module, gradIn, gradOut):
            self.grad[hookNum - 1 -
                      self.backwardIdx] = gradOut[0].detach().cpu().numpy()
            self.backwardIdx += 1
            self.backwardIdx %= hookNum
            # print(gradIn[0].size(), gradOut[0].size(), 'hook backward----')

        self.model._modules[layerName].register_forward_hook(forwardHook)
        self.model._modules[layerName].register_backward_hook(backwardHook)

    def showGrad(self):
        for item in self.conv:
            print('  hooked conv:', np.shape(item))
        for item in self.grad:
            print('  hooked grad:', np.shape(item))

    def getCAM(self):
        weights = np.maximum(self.grad[1], 0)
        convOuts = np.maximum(self.conv[1], 0)

        maps = weights * convOuts
        maps = maps.mean(axis=1)
        maps = (maps - np.min(maps)) / (np.max(maps) - np.min(maps))
        maps = np.uint8(maps * 255)

        return maps


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class RelationNet(nn.Module):

    def __init__(self, inputSize, hiddenSize):
        super(RelationNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(inputSize * 3 * 3, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


def hookFunc(module, gradIn, gradOut):
    print(gradIn[0].size(), gradOut[0].size(), "hooked!!!!!!!")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.92206, 0.92206, 0.92206],
            std=[0.08426, 0.08426, 0.08426]
        )
    ])

    print("load data folders")

    trainFolders, testFolders = mt.taskFolders(
        trainRoot="/home/zwq/Projects/LearningToCompare_FSL/datas/miniImagenet/train",
        testRoot="/home/zwq/Projects/LearningToCompare_FSL/datas/miniImagenet/test"
    )

    print("init neural networks")

    featureEncoder = Encoder()
    relationNet = RelationNet(FEATURE_DIM, RELATION_DIM)

    gradCAM = GradCAM(featureEncoder)
    gradCAM.hookLayer('layer1', 2)

    featureEncoder.to(device)
    relationNet.to(device)

    path = "./models/miniimagenet_feature_encoder_" + \
        str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl"
    if os.path.exists(path):
        featureEncoder.load_state_dict(torch.load(path, map_location=device))
        print("  load feature encoder success")

    path = "./models/miniimagenet_relation_network_" + \
        str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl"
    if os.path.exists(path):
        relationNet.load_state_dict(torch.load(path, map_location=device))
        print("  load relation network success")

    totalAcc = 0.0

    for episode in range(EPISODE):
        print("testing...")
        accs = []
        for i in range(TEST_EPISODE):
            totalRewards = 0
            counter = 0
            task = mt.MetaTask(
                testFolders,
                CLASS_NUM,
                trainNum=1,
                testNum=15
            )
            sampleLoader = mt.getMetaDataLoader(
                task,
                MetaDataset=mt.MiniImagenet,
                transform=transform,
                numPerClass=1,
                split="train",
                shuffle=False
            )
            testLoader = mt.getMetaDataLoader(
                task,
                MetaDataset=mt.MiniImagenet,
                transform=transform,
                numPerClass=3,
                split="test",
                shuffle=True
            )

            sampleImgs, sampleLabels = sampleLoader.__iter__().next()

            for testImgs, testLabels in testLoader:
                batchSize = testLabels.shape[0]
                sampleFeatures = featureEncoder(
                    Variable(sampleImgs).to(device))
                testFeatures = featureEncoder(
                    Variable(testImgs).to(device))

                sampleFeaturesExt = sampleFeatures.unsqueeze(
                    0).repeat(batchSize, 1, 1, 1, 1)
                testFeaturesExt = testFeatures.unsqueeze(
                    0).repeat(CLASS_NUM, 1, 1, 1, 1)
                testFeaturesExt = torch.transpose(testFeaturesExt, 0, 1)
                relationPairs = torch.cat(
                    (sampleFeaturesExt, testFeaturesExt), 2).view(-1, FEATURE_DIM*2, 19, 19)
                relations = relationNet(relationPairs).view(-1, CLASS_NUM)

                _, predictLabels = torch.max(relations.data, 1)
                testLabels = testLabels.to(device)

                totalRewards += predictLabels.eq(
                    testLabels).sum().cpu().numpy()
                counter += batchSize

                oneHot = torch.zeros(relations.size(), device=device)
                for i in range(relations.size(0)):
                    l = testLabels.cpu().numpy()[i]
                    oneHot[i][l] = 1
                relations.backward(gradient=oneHot)

                # gradCAM.showGrad()
                sImgs = tensor2Imgs(sampleImgs)
                tImgs = tensor2Imgs(testImgs)
                cams = gradCAM.getCAM()
                maps = getCamOnImgs(cams, tImgs, 0.2)

                plotOneShotImgs(
                    sImgs,
                    maps,
                    rows=CLASS_NUM,
                    labels=testLabels.cpu().numpy()
                )

            acc = totalRewards/1.0/counter
            accs.append(acc)

        testAcc, h = meanConfidenceInterval(accs)
        print("test acc:", testAcc, "h:", h)

        totalAcc += testAcc
    print("average acc:", totalAcc/EPISODE)


if __name__ == "__main__":
    main()
