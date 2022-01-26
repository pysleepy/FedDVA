import math
import torch
import torch.nn as nn
from torch.functional import F


class LinearWithChannel(nn.Module):
    # reference https://github.com/pytorch/pytorch/issues/36591
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.channel_size = channel_size

        # initialize weights
        # [C x I x O]
        self.w = torch.nn.Parameter(torch.zeros(self.channel_size, self.input_size, self.output_size))
        # [C x 1 x O]
        self.b = torch.nn.Parameter(torch.zeros(self.channel_size, 1, self.output_size))

        # change weights to kaiming
        self.reset_parameters(self.w, self.b)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):  # [C x N x I]
        return torch.bmm(x, self.w) + self.b  # [C x N x O]


def reparameter(mu, log_var):
    # reparameterization trick refers to https://arxiv.org/abs/1312.6114
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


def loss_dkl(mu, log_var, mu_prior, log_var_prior):
    # Equation (24)
    var_p = log_var_prior.exp()
    var = log_var.exp()
    loss = (mu - mu_prior) ** 2 / var_p - (log_var - log_var_prior) + var / var_p - 1
    loss = 0.5 * torch.sum(loss, dim=1)
    return loss


def loss_reg_c(mu_c, log_var_c):
    # Equation (26)
    n_sample = mu_c.shape[0]
    d_sample = mu_c.shape[1]

    # Descartes operations through increasing tensor dimension rather 'for' loop
    mu_c_expand = mu_c.expand(n_sample, n_sample, d_sample)
    log_var_c_expand = log_var_c.expand(n_sample, n_sample, d_sample)
    var_c = log_var_c.exp()
    var_c_expand = var_c.expand(n_sample, n_sample, d_sample)

    term_1 = (mu_c_expand.permute(1, 0, 2) - mu_c_expand) ** 2 / var_c_expand

    term_2 = - (log_var_c_expand.permute(1, 0, 2) - log_var_c_expand)

    term_3 = var_c_expand.permute(1, 0, 2) / var_c_expand

    loss = term_1 + term_2 + term_3 - 1

    loss = torch.mean(0.5 * (torch.sum(loss, dim=2)), dim=1)

    return loss


class Backbone(nn.Module):
    def __init__(self, in_channel, hidden_dims):
        super(Backbone, self).__init__()

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
    def __init__(self, in_channels, hidden_dims, out_channels):
        super(Decoder, self).__init__()
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


class DualEncoder(nn.Module):
    def __init__(self, in_h, in_w, in_channel, hidden_dims, d_z, d_c):
        super(DualEncoder, self).__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.in_channel = in_channel
        self.hidden_dims = hidden_dims
        self.d_z = d_z
        self.d_c = d_c

        self.backbone_z = Backbone(self.in_channel, self.hidden_dims)
        self.encoder_z = Encoder(self.hidden_dims[-1], self.d_z)

        self.embedding_z = nn.Linear(self.d_z, self.in_h * self.in_w)
        self.embedding_x = nn.Conv2d(self.in_channel, self.in_channel, 1)
        self.backbone_c = Backbone(self.in_channel + 1, self.hidden_dims)
        self.encoder_c = Encoder(self.hidden_dims[-1], self.d_c)

        self.decoder = Decoder(self.d_z + self.d_c, self.hidden_dims, self.in_channel)

    def forward(self, x, on_c):
        x_z = self.backbone_z(x)
        x_z = torch.flatten(x_z, start_dim=1)  # N x (hidden_dims[-1]H'W')
        mu_z, log_var_z = self.encoder_z(x_z)
        z = reparameter(mu_z, log_var_z)
        if not on_c:
            random_c = torch.zeros([2, x.shape[0], self.d_c], dtype=torch.float)
            mu_c, log_var_c = random_c[0], random_c[1]
        else:
            e_z = self.embedding_z(z).view(-1, self.in_h, self.in_w).unsqueeze(1)
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
