import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def getSubFolders(root):
    folders = [os.path.join(root, sub)
               for sub in os.listdir(root)
               if os.path.isdir(os.path.join(root, sub))]
    return folders


def taskFolders(trainRoot, testRoot):
    trainFolders = getSubFolders(trainRoot)
    testFolders = getSubFolders(testRoot)

    random.seed(1)
    random.shuffle(trainFolders)
    random.shuffle(testFolders)

    return trainFolders, testFolders


class MetaTask(object):

    def __init__(self, folders, classNum, trainNum, testNum):
        self.folders = folders
        self.classNum = classNum
        self.trainNum = trainNum
        self.testNum = testNum

        classFolders = random.sample(self.folders, self.classNum)
        labels = np.array(range(len(classFolders)))
        labels = dict(zip(classFolders, labels))
        samples = dict()

        self.trainImgs = []
        self.testImgs = []
        self.trainLabels = []
        self.testLabels = []

        for c in classFolders:
            imgs = [os.path.join(c, img) for img in os.listdir(c)]
            samples[c] = random.sample(imgs, len(imgs))
            random.shuffle(samples[c])

            self.trainImgs += samples[c][: trainNum]
            self.testImgs += samples[c][trainNum: trainNum + testNum]
            self.trainLabels += [labels[c]] * trainNum
            self.testLabels += [labels[c]] * testNum


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, targetTransform=None):
        self.task = task
        self.split = split
        self.transform = transform
        self.targetTransform = targetTransform
        self.imgs = self.task.trainImgs if self.split == 'train' else self.task.testImgs
        self.labels = self.task.trainLabels if self.split == 'train' else self.task.testLabels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "Subclass this class for your particular dataset.")


class MiniImagenet(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(MiniImagenet, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)
        if self.targetTransform is not None:
            label = self.targetTransform(label)

        return img, label


class ClassBalancedSampler(Sampler):

    def __init__(self, numPerClass, classNum, instanceNum, shuffle=True):
        self.numPerClass = numPerClass
        self.classNum = classNum
        self.instanceNum = instanceNum
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [[i + j*self.instanceNum for i in torch.randperm(self.instanceNum)[: self.numPerClass]]
                     for j in range(self.classNum)]
        else:
            batch = [[i + j*self.instanceNum for i in range(self.numPerClass)]
                     for j in range(self.classNum)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def getMetaDataLoader(metaTask, MetaDataset, transform, numPerClass=1, split='train', shuffle=False):
    dataset = MetaDataset(metaTask, split=split, transform=transform)

    sampler = ClassBalancedSampler(
        numPerClass,
        metaTask.classNum,
        metaTask.trainNum if split == 'train' else metaTask.testNum,
        shuffle=shuffle
    )

    loader = DataLoader(
        dataset,
        batch_size=numPerClass * metaTask.classNum,
        sampler=sampler
    )

    return loader
