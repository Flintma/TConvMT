import torch
import torch.nn as nn

from models.Cell import Cell

class EncoderDecoderTconMT(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderTconMT, self).__init__()

        self.encoder_1_tconvmt = Cell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_tconvmt = Cell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_tconvmt = Cell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_tconvmt = Cell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=28,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []
        x = x.permute(0,5,3,4,1,2).squeeze(5)
        # encoder
        for t in range(seq_len):
            input_x = x[:,:,:,:,t]
            h_t, c_t = self.encoder_1_tconvmt(input_tensor=input_x,
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_tconvmt(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_tconvmt(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_tconvmt(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)
        outputs = outputs.permute(0,2,3,4,1)
        return outputs

    # def forward(self, x, future_seq=0, hidden_state=None):
    def forward(self, x,future_seq = 0, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        # x = x.unsqueeze(2)
        b, seq_len, _, l, h, w = x.size()
        # future_seq = 100 - seq_len
        
        # initialize hidden states
        h_t, c_t = self.encoder_1_tconvmt.init_hidden(batch_size=b, image_size=(l, h))
        h_t2, c_t2 = self.encoder_2_tconvmt.init_hidden(batch_size=b, image_size=(l, h))
        h_t3, c_t3 = self.decoder_1_tconvmt.init_hidden(batch_size=b, image_size=(l, h))
        h_t4, c_t4 = self.decoder_2_tconvmt.init_hidden(batch_size=b, image_size=(l, h))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4) 


        return outputs      # [b,1,decode_steps,H,W]
    