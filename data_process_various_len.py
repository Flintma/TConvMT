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
from torch.nn.utils.rnn import pad_sequence

# global settings
total_samples = 500
batch_size = 1
encode_step = 4
train_split = int(total_samples*(9/10))


# %% custom training dataset
class Custom_dataset(Dataset):
    def __init__(self,x,y):
        # self.x = torch.tensor(x,dtype = torch.float32)
        # self.y = torch.tensor(y,dtype = torch.float32)
        # self.x = self.x.unsqueeze(2)
        # self.y = self.y.unsqueeze(2)
        self.x = x
        self.y = y
        
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
        
    def __len__(self):
        return len(self.x)

#training_dataset = Custom_dataset(X,Y)
#train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)

# %%
class TopologyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir = 'data_0', batch_size = batch_size, total_samples = total_samples,
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
            rand_encode_step = np.random.randint(1, self.encode_step + 1)
            path = '/'.join([self.data_dir,str(i)+".npz"])
            with np.load(path) as data:
                x_t = data['arr_0'][:rand_encode_step]
                y_t = data['arr_0'][rand_encode_step:]
                self.X.append(x_t)
                self.Y.append(y_t)
        #self.X = np.stack(self.X) # batch_size × time_step × H × W 
        #self.Y = np.stack(self.Y)
    
    def setup(self, stage = None):
        dataset = Custom_dataset(self.X,self.Y)
        train_dataset,test_dataset = random_split(dataset,[self.train_split,self.total_samples - self.train_split])
        if stage == 'fit' or stage is None:
            self.train_dataset = train_dataset
        if stage == 'test' or stage is None:
            self.test_dataset = test_dataset
    
    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers=24, collate_fn = self.pad_collate)  # batch, time, channel, height, width
        return DataLoader(self.train_dataset, batch_size = 1, shuffle = True, num_workers=24)
        
    def test_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers=24, collate_fn = self.pad_collate)
        return DataLoader(self.train_dataset, batch_size = 1, shuffle = True, num_workers=24)
        
    def pad_collate(_,y):
        xx, yy = [], []
        x_lens, y_lens = [], []
        for i, j in y:
            xx.append(torch.tensor(i,dtype = torch.float32))
            yy.append(torch.tensor(j,dtype = torch.float32))
            x_lens.append(i.shape[0])
            y_lens.append(j.shape[0])
        
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=-1)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=-1)

        return xx_pad, yy_pad, x_lens, y_lens
# %%
if __name__ == "__main__":
    dt = TopologyDataModule()
    dt.prepare_data()
    dt.setup()
    ss = dt.train_dataloader()     #padded_x(batch,time_sequence,H,W), padded_y, pad_lenx(list of padding length), pad_leny
    samples = list(ss)
    
    aa = print("dddd")
# %%
