import torch
import torch.nn as nn
from torch.functional import F

from models.DualEncoder import Backbone, Encoder, Decoder, DualEncoder, Classifier
from models.funcs import reparameter

IN_H = 28
IN_W = 28
IN_C = 1
hidden_dims = [64, 64 * 2, 64 * 4, 64 * 8]

class BackboneMNIST(Backbone):
    def __init__(self, in_channel, hidden_dims):
        super(BackboneMNIST, self).__init__()

        self.hidden_dims = hidden_dims
        self.in_channel = in_channel
        self.layers = nn.ModuleList()

        # for images of 28 x 28 padding=3. for images of 32 x 32 padding = 1.
        first_layer = nn.Sequential(nn.Conv2d(in_channels=self.in_channel, out_channels=self.hidden_dims[0]
                                              , kernel_size=4, stride=2, padding=3)
                                    , nn.BatchNorm2d(self.hidden_dims[0])
                                    , nn.LeakyReLU())
        cur_channel = self.hidden_dims[0]
        self.layers.append(first_layer)
        for h_dim in self.hidden_dims[1:-1]:
            cur_layer = nn.Sequential(nn.Conv2d(in_channels=cur_channel, out_channels=h_dim
                                                , kernel_size=4, stride=2, padding=1)
                                      , nn.BatchNorm2d(h_dim)
                                      , nn.LeakyReLU())
            cur_channel = h_dim
            self.layers.append(cur_layer)
        last_layer = nn.Sequential(nn.Conv2d(in_channels=cur_channel, out_channels=self.hidden_dims[-1]
                                             , kernel_size=4, stride=1, padding=0)
                                   , nn.BatchNorm2d(self.hidden_dims[-1])
                                   , nn.LeakyReLU())
        self.layers.append(last_layer)

    def forward(self, x):
        for layer in self.layers:  # [N x C x H x W] -> [N x hidden_dims[-1] x H' x W']
            x = layer(x)
        return x


class DecoderMNIST(Decoder):
    def __init__(self, in_channels, hidden_dims, out_channels):
        super(DecoderMNIST, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = list(reversed(hidden_dims))
        self.out_channels = out_channels

        self.fc_1 = nn.Linear(self.in_channels, int(self.hidden_dims[0] / 2))
        self.fc_2 = nn.Linear(int(self.hidden_dims[0] / 2), self.hidden_dims[0])

        self.layers = nn.ModuleList()
        first_layer = nn.Sequential(nn.ConvTranspose2d(self.hidden_dims[0], self.hidden_dims[1]
                                                       , kernel_size=4, stride=1, padding=0)
                                    , nn.BatchNorm2d(self.hidden_dims[1])
                                    , nn.LeakyReLU())
        cur_channel = self.hidden_dims[1]
        self.layers.append(first_layer)
        for h_dim in self.hidden_dims[2:]:
            cur_layer = nn.Sequential(nn.ConvTranspose2d(cur_channel, h_dim
                                                         , kernel_size=4, stride=2, padding=1)
                                      , nn.BatchNorm2d(h_dim)
                                      , nn.LeakyReLU())
            cur_channel = h_dim
            self.layers.append(cur_layer)
        last_layer = nn.Sequential(nn.ConvTranspose2d(cur_channel, self.out_channels
                                                      , kernel_size=4, stride=2, padding=3)
                                   , nn.Tanh())
        self.layers.append(last_layer)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = x.view(-1, self.hidden_dims[0], 1, 1)
        for layer in self.layers:
            x = layer(x)
        return x


class ClassifierMNIST(Classifier):
    def __init__(self, d_in, d_out):
        super(ClassifierMNIST, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.fc_1 = nn.Linear(self.d_in, int(self.d_in / 2))
        self.fc_2 = nn.Linear(int(self.d_in / 2), self.d_out)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        outputs = F.softmax(self.fc_2(x))
        return outputs


class DualEncoderMNIST(DualEncoder):
    MODEL_TYPE = "DualEncoderMNIST"

    def __init__(self, d_z, d_c):
        super(DualEncoderMNIST, self).__init__()
        self.in_h = IN_H
        self.in_w = IN_W
        self.in_channel = IN_C
        self.hidden_dims = hidden_dims
        self.d_z = d_z
        self.d_c = d_c

        self.backbone_z = BackboneMNIST(self.in_channel, self.hidden_dims)
        self.encoder_z = Encoder(self.hidden_dims[-1], self.d_z)

        self.embedding_z = nn.Linear(self.d_z, self.in_h * self.in_w)
        self.embedding_x = nn.Conv2d(self.in_channel, self.in_channel, 1)
        self.backbone_c = BackboneMNIST(self.in_channel + 1, self.hidden_dims)
        self.encoder_c = Encoder(self.hidden_dims[-1], self.d_c)

        self.decoder = DecoderMNIST(self.d_z + self.d_c, self.hidden_dims, self.in_channel)

    def forward(self, x, on_c):
        x_z = self.backbone_z(x)
        x_z = torch.flatten(x_z, start_dim=1)  # N x (hidden_dims[-1]H'W')
        mu_z, log_var_z = self.encoder_z(x_z)
        z = reparameter(mu_z, log_var_z)
        if not on_c:
            random_c = torch.zeros([2, x.shape[0], self.d_c], dtype=torch.float, device=x.device)
            mu_c, log_var_c = random_c[0], random_c[1]
        else:
            e_z = self.embedding_z(z.detach()).view(-1, self.in_h, self.in_w).unsqueeze(1)
            e_x = self.embedding_x(x)
            x_c = torch.cat([e_x, e_z], dim=1)
            x_c = self.backbone_c(x_c)
            x_c = torch.flatten(x_c, start_dim=1)  # N x (hidden_dims[-1]H'W')
            mu_c, log_var_c = self.encoder_c(x_c)

        c = reparameter(mu_c, log_var_c)
        x_hat = self.decoder(torch.cat([z, c], dim=1))
        return x_hat, z,  c, mu_z, log_var_z, mu_c, log_var_c

    def generate(self, z, c):
        output = self.decoder(torch.cat([z, c], dim=1))
        return output
