import torch
import torch.nn as nn
from torch.functional import F

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
    def __init__(self,  in_channels, hidden_dims, d_in):
        super(Backbone, self).__init__()

        self.d_in = d_in
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        cur_dim = self.d_in
        for hid_dim in self.hidden_dims:
            cur_layer = nn.Sequential(nn.Linear(cur_dim, hid_dim)
                                      , nn.ReLU())
            cur_dim = hid_dim
            self.layers.append(cur_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_in, d_out):
        super(Encoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.fc_mu_z = nn.Linear(self.d_in, self.d_out)
        self.fc_log_var_z = nn.Linear(self.d_in, self.d_out)

    def forward(self, x):
        mu_z = self.fc_mu_z(x)
        log_var_z = self.fc_log_var_z(x)
        return mu_z, log_var_z


class Decoder(nn.Module):
    def __init__(self, d_encoding, hidden_dims, out_channels):
        super(Decoder, self).__init__()
        self.d_encoding = d_encoding
        self.out_channels = out_channels
        self.hidden_dims = list(reversed(hidden_dims))

        self.fc_input = nn.Linear(self.d_encoding, self.hidden_dims[0])
        self.layers = nn.ModuleList()
        cur_dim = self.hidden_dims[0]
        for h_dim in self.hidden_dims[1:]:
            cur_layer = nn.Sequential(nn.Linear(cur_dim, h_dim)
                                      , nn.ReLU())
            cur_dim = h_dim
            self.layers.append(cur_layer)
        last_layer = nn.Sequential(nn.Linear(cur_dim, 28 * 28 * self.out_channels)
                                   , nn.Tanh())
        self.layers.append(last_layer)

    def forward(self, encoding):
        x = F.relu(self.fc_input(encoding))
        for layer in self.layers:
            x = layer(x)
        return x


class DualEncoder(nn.Module):
    def __init__(self, in_channel, hidden_dims, d_z, d_c):
        super(DualEncoder, self).__init__()
        self.in_channel = in_channel
        self.hidden_dims = hidden_dims
        self.d_z = d_z
        self.d_c = d_c

        self.backbone_z = Backbone(self.in_channel, self.hidden_dims, 28 * 28)
        self.backbone_c = Backbone(self.in_channel, self.hidden_dims, 28 * 28 + self.d_z)
        self.encoder_z = Encoder(self.hidden_dims[-1], self.d_z)
        self.encoder_c = Encoder(self.hidden_dims[-1], self.d_c)
        self.decoder_z = Decoder(self.d_z, self.hidden_dims, self.in_channel)
        self.decoder_c = Decoder(self.d_z + self.d_c, self.hidden_dims, self.in_channel)

    def forward(self, x, z=None):
        x = x.view(-1, 1 * 28 * 28)
        if z is None:
            x_z = self.backbone_z(x)
            mu_z, log_var_z = self.encoder_z(x_z)
            z = reparameter(mu_z, log_var_z)
            x_hat = self.decoder_z(z).view(-1, 1, 28, 28)
            return x_hat, z, mu_z, log_var_z
        else:
            x_c = self.backbone_c(torch.cat([x, z], dim=1))
            mu_c, log_var_c = self.encoder_c(x_c)
            c = reparameter(mu_c, log_var_c)
            x_hat = self.decoder_c(torch.cat([z.detach(), c], dim=1)).view(-1, 1, 28, 28)
            return x_hat, c, mu_c, log_var_c

    def generate(self, z, c):
        output_z = self.decoder_z(z).view(-1, 1, 28, 28)
        output_c = self.decoder_c(torch.cat([z, c], dim=1)).view(-1, 1, 28, 28)
        return output_z, output_c



