import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import MetaTask as mt
import os
import math
import argparse
from scipy import stats


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


def meanConfidenceInterval(data, confidence=0.95):
    arr = 1.0 * np.array(data)
    n = len(arr)
    mean, sem = np.mean(arr), stats.sem(arr)
    h = sem * stats.t._ppf((1 + confidence)/2., n - 1)
    return mean, h


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


def main():
    device = "cuda:0"
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

            acc = totalRewards/1.0/counter
            accs.append(acc)

        testAcc, h = meanConfidenceInterval(accs)
        print("test acc:", testAcc, "h:", h)

        totalAcc += testAcc
    print("average acc:", totalAcc/EPISODE)


if __name__ == "__main__":
    main()
