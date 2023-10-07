# %%
import sys
import torch
from torch.optim import Adam
from torch.optim import AdamW
import pytorch_lightning as pl
from models.TConvMT import EncoderDecoderTconMT
import data_process
import numpy as np
from pyvtk import CellData, LookupTable, Scalars, UnstructuredGrid, VtkData
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import os
import pathlib
from visualisation import create_3d_geom
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# global config
n_hidden_dim = 100
# n_hidden_dim = 32
n_gpus = 1
max_epochs = 300
batch_size = 2
encode_step = 4
decode_step = 100 - encode_step
log_images = False


beta_1 = .9
beta_2 = .98

#%%
class TCONVMT(pl.LightningModule):
    def __init__(self,model = None):
        super().__init__()
        self.model = model
        
        self.log_images = log_images
        
        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.encode_step = encode_step
        self.decode_step = decode_step
        self.save_hyperparameters()
        
    def create_video(self, x, y_hat, y):
        # predictions with input for illustration purposes
        preds = torch.cat([x, y_hat.unsqueeze(2)], dim=1)[0]

        # entire input and ground truth
        y_plot = torch.cat([x, y.unsqueeze(2)], dim=1)[0]

        # error (l2 norm) plot between pred and ground truth
        # difference = (torch.pow(y_hat[0] - y[0], 2)).detach()
        # zeros = torch.zeros(difference.shape)
        # difference_plot = torch.cat([zeros.unsqueeze(0), difference.unsqueeze(0)], dim=1)[
        #     0].unsqueeze(1)

        # concat all images
        # final_image = torch.cat([preds, y_plot, difference_plot], dim=0)
        final_image = torch.cat([preds, y_plot], dim=0)

        # make them into a single grid image file
        grid = torchvision.utils.make_grid(final_image, nrow=self.encode_step + self.decode_step)
        return grid
    
    def forward(self,x):
        output = self.model(x, future_seq = self.decode_step)
        return output

    def training_step(self,batch,batch_idx):
        x, y = batch
        y = y.squeeze()
        y_hat = self.forward(x).squeeze()

        loss = self.criterion(y_hat, y)

        self.log("MSE_loss",loss, on_step = True, prog_bar=True,on_epoch=True, logger = True)
        
        # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved)

        # save predicted images every 250 global_step
        if self.log_images:
            if self.global_step % 250 == 0:
                final_image = self.create_video(x, y_hat, y)

                self.logger.experiment.add_image(
                    'epoch_' + str(self.current_epoch) + '_step' + str(self.global_step) + '_generated_images',
                    final_image, 0)
                plt.close()

        tensorboard_logs = {'train_mse_loss': loss.detach(),
                            'learning_rate': lr_saved}

        return {'loss': loss, 'log': tensorboard_logs}
        
    def configure_optimizers(self):
        # return Adam(self.parameters(),lr = 1e-3, betas = (beta_1,beta_2))
        return AdamW(self.parameters(),lr = 1e-3, betas = (beta_1,beta_2),weight_decay = 0.01)
        
    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y = y.squeeze()
        y_hat = self.forward(x).squeeze()
        loss = self.criterion(y_hat, y)
        
        self.log("val_loss",loss, on_step = True, prog_bar=True,on_epoch=True, logger = True)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


#%%
if __name__ == '__main__':
    tconvmt_model = EncoderDecoderTconMT(nf = n_hidden_dim, in_chan = 28)
    
    
    model = TCONVMT(model = tconvmt_model)
    trainer = pl.Trainer(max_epochs=max_epochs,gpus=n_gpus)
    dm = data_process.TopologyDataModule()
   
    start = time.perf_counter()
    trainer.fit(model, dm)
    end = time.perf_counter()
    print(f"Training spent {end - start:0.4f} seconds")
    # trainer.fit(model,dm,ckpt_path = 'lightning_logs/40_40_10_encoder_step/checkpoints/epoch=299-step=11399.ckpt')
    
    
#%%

model = TCONVMT.load_from_checkpoint("lightning_logs/version_0/checkpoints/firstmodel.ckpt")
model.eval()

# %% for encoder-decoder
path = '/'.join(["data/0_16",str(16)+".npz"])
with np.load(path) as data:
    a = data['arr_0']
    

x = torch.tensor(a[:4]).unsqueeze(0)
x = x.unsqueeze(2)

y_hat = model(x.float()).detach().numpy()

directory = './results'
if not os.path.exists(directory):
        pathlib.Path(directory).mkdir(parents=True)

for t in range(96):
    params = {
        'prefix': 'prediction',
         'iternum': t+1,
         'time': 'none',
         'dir': directory
    }
    create_3d_geom(y_hat[0,t,:,:,:], **params)

###########################################################################################
# %%
# path = '/'.join(["data/test_data","square_2_loads_40_40.npz"])
# with np.load(path) as data:
#     a = data['arr_0']
    

# x = torch.tensor(a[:50]).unsqueeze(0)
# x = x.unsqueeze(2)
# y = torch.tensor(a[99])
# model.to(torch.device("cpu"))
# y_hat = model(x.float()).detach().numpy()
# input = a[4]

# f,axarr = plt.subplots(1,3)
# axarr[0].imshow(input,cmap = 'Greys', interpolation = 'none')
# axarr[0].set_title('Input')
# axarr[1].imshow(y,cmap = 'Greys', interpolation = 'none')
# axarr[1].set_title('Ground Truth')
# axarr[2].imshow(y_hat[0][0][-1],cmap = 'Greys', interpolation = 'none')
# axarr[2].set_title('Prediction')
# %%
path = '/'.join(["data/0_16",str(2) +".npz"])
with np.load(path) as data:
    a = data['arr_0']
    

x = torch.tensor(a[:4]).unsqueeze(0)
x = x.unsqueeze(2)
y = torch.tensor(a[99])
model.to(torch.device("cpu"))
y_hat = model(x.float()).detach().numpy()
input = a[4]

# %% IoU comparison
def pixel_value_error(outputs,labels):
    # error = np.divide(abs(outputs-labels),labels)
    error = abs(outputs-labels)
    return error.mean()
pixel_value_error(y_hat[0][-1],a[-1])
# %%
p_error = []
for i in range(96):
    p_error.extend([pixel_value_error(y_hat[0][i],a[i+4])*1.3])
# %%
fig = plt.figure(1,(7,4))
ax = fig.add_subplot(1,1,1)
ax.plot(range(4,100),p_error)
# fmt = '%.00f%%'
# yticks = mtick.FormatStrFormatter(fmt)
# ax.yaxis.set_major_formatter(yticks)
# plt.ylim([0.01,0.03])
plt.xlabel('step')
plt.ylabel('pixel values error')
plt.savefig('pixel_value_error.png')
# %%
