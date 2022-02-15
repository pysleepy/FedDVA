import torch
import torch.nn as nn
from torch.functional import F
from models.LinearWithChannel import LinearWithChannel


class Backbone(nn.Module):
    pass


class Encoder(nn.Module):
    def __init__(self, in_features, d_latent):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.d_latent = d_latent

        self.fc_1 = LinearWithChannel(self.in_features, int(self.in_features / 2), 2)
        self.fc_2 = LinearWithChannel(int(self.in_features / 2), self.d_latent, 2)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=0).repeat(2, 1, 1)  # [N x I -> 2 x N x I]
        x = F.relu(self.fc_1(x))  # [2 x N x I] -> [2 x N x I/2]
        x = self.fc_2(x)  # [2 x N x I/2] -> [2 x N x D]
        mu, log_var = x[0], x[1]
        return mu, log_var


class Decoder(nn.Module):
    pass


class DualEncoder(nn.Module):
    def generate(self, z, c):
        raise Exception("Function Not implemented")

