# %%
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim import AdamW
import pytorch_lightning as pl
from models.TConvMT import EncoderDecoderTconMT
import data_process
import torchvision
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.ticker as mtick
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# global config
# n_hidden_dim = 100
n_hidden_dim = 64
n_gpus = 1
max_epochs = 67
batch_size = 12
encode_step = 4
decode_step = 100 - encode_step
log_images = True


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
    tconvmt_model = EncoderDecoderTconMT(nf = n_hidden_dim, in_chan = 1)
    
    model = TCONVMT(model = tconvmt_model)
    
    # trainer = pl.Trainer(max_epochs=max_epochs,gpus=n_gpus,fast_dev_run = True)
    trainer = pl.Trainer(max_epochs=max_epochs,gpus=n_gpus,log_every_n_steps=4)
    dm = data_process.TopologyDataModule()
    # trainer.fit(model, dm)
    trainer.fit(model,dm,ckpt_path = 'lightning_logs/custom_saving/sample-xxx-epoch=66-val_loss=0.02.ckpt')
    

# %% for encoder-decoder
# path = '/'.join(["data/test_data/random_40_40",str(0)+".npz"])
path = '/'.join(["data/40_40_4665",str(2171) +".npz"])
with np.load(path) as data:
    a = data['arr_0']
    

x = torch.tensor(a[:4]).unsqueeze(0)
x = x.unsqueeze(2)
y = torch.tensor(a[99])
model.to(torch.device("cpu"))
y_hat = model(x.float()).detach().numpy()
input = a[4]

f,axarr = plt.subplots(1,3)
axarr[0].imshow(input,cmap = 'Greys', interpolation = 'none')
axarr[0].set_title('Input')
axarr[1].imshow(y,cmap = 'Greys', interpolation = 'none')
axarr[1].set_title('Ground Truth')
axarr[2].imshow(y_hat[0][0][-1],cmap = 'Greys', interpolation = 'none')
axarr[2].set_title('Prediction')

# %% for encoder-decoder
f,axarr = plt.subplots(1,6)
axarr[0].imshow(input,cmap = 'Greys', interpolation = 'none')
axarr[0].set_title('Input')
axarr[1].imshow(y_hat[0][0][0],cmap = 'Greys', interpolation = 'none')
axarr[1].set_title('Prediction at step 1')
axarr[2].imshow(y_hat[0][0][19],cmap = 'Greys', interpolation = 'none')
axarr[2].set_title('Prediction at step 20')
axarr[3].imshow(y_hat[0][0][39],cmap = 'Greys', interpolation = 'none')
axarr[3].set_title('Prediction at step 40')
axarr[4].imshow(y_hat[0][0][59],cmap = 'Greys', interpolation = 'none')
axarr[4].set_title('Prediction at step 60')
axarr[5].imshow(y_hat[0][0][-1],cmap = 'Greys', interpolation = 'none')
axarr[5].set_title('Final prediction')


###########################################################################################
# %%
criterion = torch.nn.MSELoss()
predict = torch.tensor(y_hat[0][0][-1])
loss = criterion(predict, y)


# %%
f,axarr = plt.subplots(2,10)
f.subplots_adjust(wspace=.001, hspace = .001)
for i in range(10):
    axarr[0][i].imshow(y_hat[0][0][i*10],cmap = 'Greys', interpolation = 'none')
    # axarr[0][i].set_title('step '+str(i))
    axarr[0][i].axis('off')
    axarr[1][i].imshow(a[i*10+4],cmap = 'Greys', interpolation = 'none')
    axarr[1][i].axis('off')
f.savefig('sample_2171.png')
# %%
f,axarr = plt.subplots(2,20)
f.subplots_adjust(wspace=.001, hspace = .001)
for i in range(20):
    axarr[0][i].imshow(y_hat[0][0][i],cmap = 'Greys', interpolation = 'none')
    # axarr[0][i].set_title('step '+str(i))
    axarr[0][i].axis('off')
    axarr[1][i].imshow(a[i+4],cmap = 'Greys', interpolation = 'none')
    axarr[1][i].axis('off')
f.savefig('1.png')

# %% input generation
f,axarr = plt.subplots(1,3)
f.subplots_adjust(wspace=.001, hspace = .001)
axarr[0].imshow(a[0],cmap = 'Greys', interpolation = 'none')
axarr[0].axis('off')
axarr[1].imshow(a[1],cmap = 'Greys', interpolation = 'none')
axarr[1].axis('off')
axarr[2].imshow(a[2],cmap = 'Greys', interpolation = 'none')
axarr[2].axis('off')
# %% IoU comparison
def pixel_value_error(outputs,labels):
    # error = np.divide(abs(outputs-labels),labels)
    error = abs(outputs-labels)
    return error.mean()
pixel_value_error(y_hat[0][0][-1],a[-1])
# %%
p_error = []
for i in range(96):
    p_error.extend([pixel_value_error(y_hat[0][0][i],a[i+4])])
# %%
fig = plt.figure(1,(7,4))
ax = fig.add_subplot(1,1,1)
ax.plot(range(4,100),p_error)
fmt = '%.0f%%'
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)
plt.xlabel('step')
plt.ylabel('pixel values error')
plt.savefig('pixel_value_error.png')


# %%
