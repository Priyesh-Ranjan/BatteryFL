from __future__ import print_function

import pickle

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from dataloader import *
import os
from torchvision.io import read_image


"""
class Net(nn.Module):
    #num_classes = 10
    #model = resnet18(pretrained=True)
    #n = model.fc.in_features
    #model.fc = nn.Linear(n, num_classes)
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,   32,  3, padding = 'same')
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  32, 3, padding = 'same')
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.3)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 'same')
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding = 'same')
        self.conv5 = nn.Conv2d(64, 128, 3, padding = 'same')
        self.fc1 = nn.Linear(128 * 12 * 12, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 24)
        self.fc5 = nn.Linear(24,2)


    def forward(self, x):
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.conv1_bn(self.pool(F.relu(self.conv2(x))))
        x = self.drop1(x)
        x = self.conv2_bn(F.relu(self.conv3(x)))
        x = self.pool(self.conv2_bn(F.relu(self.conv4(x))))
        x = self.drop1(x)
        x = F.relu(self.conv5(x))
        x = x.view(-1, 128 * 12 * 12)
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = self.fc2_bn(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.drop1(x)
        x = F.relu(self.fc4(x))
        #x = F.softmax(self.fc5(x), dim=0)
        x = self.fc5(x)
        return x
"""

def Net():
    num_classes = 2
    model = resnet18(pretrained=False)
    n = model.fc.in_features
    model.fc = nn.Linear(n, num_classes)
    return model


class DDSMDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.targets = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.targets.iloc[idx, 0])
        image = read_image(img_path)
        image = torch.as_tensor(np.asarray(image), dtype=torch.float32) 
        label = self.targets.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def getDataset():
    dataset = DDSMDataset('./Images/Train/Annotations.csv', './Images/Train',
                            transform=transforms.Compose([transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]))
    #dataset = datasets.CIFAR10('./data',train=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]))
    #dataset = DataLoader(datasets, batch_size=4, shuffle=True)
    return dataset


def basic_loader(num_clients, loader_type):
    dataset = getDataset()
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk'):
    assert loader_type in ['iid', 'byLabel',
                           'dirichlet'], 'Loader has to be one of the  \'iid\',\'byLabel\',\'dirichlet\''
    if loader_type == 'iid':
        loader_type = iidLoader
    elif loader_type == 'byLabel':
        loader_type = byLabelLoader
    elif loader_type == 'dirichlet':
        loader_type = dirichletLoader

    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except:
            print('loader not found, initialize one')
            loader = basic_loader(num_clients, loader_type)
    else:
        print('initialize a data loader')
        loader = basic_loader(num_clients, loader_type)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader


def test_dataloader(test_batch_size):
    #test_loader = torch.utils.data.DataLoader(
    #    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),
    #                                                                          transforms.Normalize((0.5,0.5,0.5),
    #                                                                                               (0.5,0.5,0.5))])),
    #    batch_size=test_batch_size, shuffle=True)
    test_loader = DataLoader(DDSMDataset('./Images/Test/Annotations.csv', './Images/Test',
                            transform=transforms.Compose([transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])), 
                            batch_size=test_batch_size, shuffle=True)
    
    return test_loader


if __name__ == '__main__':
    from torchsummary import summary

    print("#Initialize a network")
    net = Net()
    summary(net.cuda(), (3, 50, 50))

    print("\n#Initialize dataloaders")
    loader_types = ['iid', 'dirichlet']
    for i in range(len(loader_types)):
        loader = train_dataloader(50, loader_types[i], store=False)
        print(f"Initialized {len(loader)} loaders (type: {loader_types[i]}), each with batch size {loader.bsz}.\
        \nThe size of dataset in each loader are:")
        print([len(loader[i].dataset) for i in range(len(loader))])
        print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    print("\n#Feeding data to network")
    x = next(iter(loader[i]))[0].cuda()
    y = net(x)
    print(f"Size of input:  {x.shape} \nSize of output: {y.shape}")
