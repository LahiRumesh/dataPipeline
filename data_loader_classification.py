import torch
from torch.utils.data import Dataset,DataLoader
import os
from torchvision import transforms
import numpy as np


training_data=np.load("training_data.npy",allow_pickle=True)
testing_data=np.load("testing_data.npy",allow_pickle=True)


class Classification_DATASET(Dataset):
    def __init__(self,data_set,transform=None):
        self.data_set=data_set
        self.transform=transform
        #self.labels=labels

    def __len__(self):
        return(len(self.data_set))

    def __getitem__(self,idx):
        data = self.data_set[idx][0]

        if self.transform:
            data=self.transform(data)
            return (data,self.data_set[idx][1])
        else:
            return (data,self.data_set[idx][1])

data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


train_data = Classification_DATASET(training_data)
test_data = Classification_DATASET(testing_data)

trainloader = DataLoader(train_data, batch_size=128, shuffle=False)
testloader = DataLoader(test_data, batch_size=128, shuffle=True)
