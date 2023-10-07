# %%
import os
import torch

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np

# global settings
total_samples = 16
batch_size = 2
encode_step = 4
train_split = int(total_samples*(1/10))


# %% custom training dataset
class Custom_dataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype = torch.float32)
        self.y = torch.tensor(y,dtype = torch.float32)
        self.x = self.x.unsqueeze(2)
        self.y = self.y.unsqueeze(2)

        
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
        
    def __len__(self):
        return self.x.shape[0]

#training_dataset = Custom_dataset(X,Y)
#train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)

# %%
class TopologyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir = 'data/0_16', batch_size = batch_size, total_samples = total_samples,
                 encode_step = encode_step,train_split = train_split):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.total_samples = total_samples
        self.X = []
        self.Y = []
        self.encode_step = encode_step
        self.train_split = train_split
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        
    def prepare_data(self):
        for i in range(total_samples):
            path = '/'.join([self.data_dir,str(i)+".npz"])
            with np.load(path) as data:
                x_t = data['arr_0'][:self.encode_step]
                y_t = data['arr_0'][self.encode_step:]
                self.X.append(x_t)
                self.Y.append(y_t)
        self.X = np.stack(self.X) # batch_size × time_step × H × W 
        self.Y = np.stack(self.Y)
    
    def setup(self, stage = None):
        dataset = Custom_dataset(self.X,self.Y)
        # train_dataset,test_dataset = random_split(dataset,[self.train_split,self.total_samples - self.train_split])
        train_dataset,val_dataset = random_split(dataset,[self.train_split,self.total_samples - self.train_split])
        if stage == 'fit' or stage is None:
            # self.train_dataset = dataset
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        if stage == 'validate' or stage is None:
            self.val_dataset = val_dataset
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=24)  # batch, time, channel, height, width
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size)
    
    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=24)
        
# %%
if __name__ == "__main__":
    dt = TopologyDataModule()
    dt.prepare_data()
    dt.setup()  # b t 1 l h w

# %%
