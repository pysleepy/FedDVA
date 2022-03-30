import torch
import torch.nn as nn
from torch.functional import F

from models.funcs import reparameter

IN_H = 28
IN_W = 28
IN_C = 1
hidden_dims = [64, 64 * 2, 64 * 4, 64 * 8]


class BackboneMNIST(nn.Module):
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


class EncoderMNIST(nn.Module):
    def __init__(self, in_features, d_latent):
        super(EncoderMNIST, self).__init__()
        self.in_features = in_features
        self.d_latent = d_latent

        self.fc_1 = nn.Linear(self.in_features, int(self.in_features / 2))
        self.fc_2 = nn.Linear(int(self.in_features / 2), self.d_latent)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        return x


class ClassifierMNIST(nn.Module):
    def __init__(self, d_in, d_out):
        super(ClassifierMNIST, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.fc_1 = nn.Linear(self.d_in, int(self.d_in / 2))
        self.fc_2 = nn.Linear(int(self.d_in / 2), self.d_out)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        outputs = F.softmax(self.fc_2(x), dim=1)
        return outputs


class CNNMnist(nn.Module):
    MODEL_TYPE = "CNNMNIST"

    def __init__(self, d_in, d_out):
        super(CNNMnist, self).__init__()
        self.in_h = IN_H
        self.in_w = IN_W
        self.in_channel = IN_C
        self.hidden_dims = hidden_dims
        self.d_in = d_in
        self.d_out = d_out

        self.backbone = BackboneMNIST(self.in_channel, self.hidden_dims)
        self.encoder = EncoderMNIST(self.hidden_dims[-1], self.d_in)

        self.classifier = ClassifierMNIST(self.d_in, self.d_out)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = self.classifier(x)
        return x
