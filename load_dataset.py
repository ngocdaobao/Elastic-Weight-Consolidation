import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

class Permutation:
    def __init__(self, permute):
        self.permute = permute
    def __call__(self, img):
        img = img.view(-1)
        img = img[self.permute]
        img = img.view(1,28,28)
        return img

def permute_mnist(permute):

    PermutationMNIST = Permutation(permute)
    transform_permute = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
                                PermutationMNIST,
                               ])
    # Load the dataset
    dataset = torchvision.datasets.MNIST(root='./data',  # Directory to store the data
                                          train=True,      # Load training data
                                          download=True,   # Download if not present
                                          transform=transform_permute)  # Apply the transformations
    # Split into train and test set:
    trainset,testset = random_split(dataset, [0.5,0.5])
    testset,_ = random_split(testset, [0.25, 0.75])
    return trainset, testset

class custom_data(Dataset):
    def __init__(self, dataset):
        self.data = dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        data =  self.data[idx][0],
        label =  self.data[idx][1]
        return data, label
    def __getsample__(self, sample_size):
      idx = random.sample(range(len(self.data)), sample_size)
      sample = [self.data[i] for i in idx]
      return sample


