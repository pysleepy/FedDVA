import logging

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
    def __init__(self,  in_channels, hidden_dims):
        super(Backbone, self).__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        cur_in_channels = self.in_channels
        for h_dim in hidden_dims:
            cur_layer = nn.Sequential(nn.Conv2d(cur_in_channels, out_channels=h_dim, kernel_size=3
                                                               , stride=2, padding=1)
                                      # , nn.BatchNorm2d(h_dim)
                                      , nn.LeakyReLU())
            self.layers.append(cur_layer)
            cur_in_channels = h_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        return x


class Encoder_z(nn.Module):
    def __init__(self, d_x, d_z):
        super(Encoder_z, self).__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.fc_mu_z = nn.Linear(self.d_x, self.d_z)
        self.fc_log_var_z = nn.Linear(self.d_x, self.d_z)

    def forward(self, x):
        mu_z = self.fc_mu_z(x)
        log_var_z = self.fc_log_var_z(x)
        return mu_z, log_var_z


class Encoder_c(nn.Module):
    def __init__(self, d_x, d_z, d_c):
        super(Encoder_c, self).__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.d_c = d_c
        self.fc_mu_c = nn.Linear(self.d_x + self.d_z, self.d_c)
        self.fc_log_var_c = nn.Linear(self.d_x + self.d_z, self.d_c)

    def forward(self, x, z):
        mu_c = self.fc_mu_c(torch.cat([x, z], dim=1))
        log_var_c = self.fc_log_var_c(torch.cat([x, z], dim=1))
        return mu_c, log_var_c


class Decoder(nn.Module):
    def __init__(self, d_encoding, hidden_dims):
        super(Decoder, self).__init__()
        self.d_encoding = d_encoding
        self.hidden_dims = list(reversed(hidden_dims))

        self.fc_input = nn.Linear(self.d_encoding, self.hidden_dims[0])
        self.layers = nn.ModuleList()
        for l_id in range(len(self.hidden_dims) - 3):
            cur_layer = nn.Sequential(nn.ConvTranspose2d(self.hidden_dims[l_id], self.hidden_dims[l_id + 1]
                                                         , kernel_size=3, stride=2
                                                         , padding=1, output_padding=1)
                                      # , nn.BatchNorm2d(self.hidden_dims[l_id + 1])
                                      , nn.LeakyReLU())
            self.layers.append(cur_layer)
        last_but_two = nn.Sequential(nn.ConvTranspose2d(self.hidden_dims[-3], self.hidden_dims[-2]
                                                        , kernel_size=3, stride=2, padding=1, output_padding=0)
                                     # , nn.BatchNorm2d(self.hidden_dims[-2])
                                     , nn.LeakyReLU())
        self.layers.append(last_but_two)
        last_but_one = nn.Sequential(nn.ConvTranspose2d(self.hidden_dims[-2], self.hidden_dims[-1]
                                                        , kernel_size=3, stride=2, padding=1, output_padding=1)
                                     # , nn.BatchNorm2d(self.hidden_dims[-1])
                                     , nn.LeakyReLU())
        self.layers.append(last_but_one)
        last_layer = nn.Sequential(nn.ConvTranspose2d(self.hidden_dims[-1], self.hidden_dims[-1]
                                                      , kernel_size=3, stride=2, padding=1, output_padding=1)
                                   # , nn.BatchNorm2d(self.hidden_dims[-1])
                                   , nn.LeakyReLU()
                                   , nn.Conv2d(self.hidden_dims[-1], out_channels=1, kernel_size=3, padding=1)
                                   , nn.Tanh())
        self.layers.append(last_layer)

    def forward(self, encoding_z, encoding_c):
        x = F.relu(self.fc_input(torch.cat([encoding_z, encoding_c], dim=1)))
        x = x.view(-1, self.hidden_dims[0], 1, 1)
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

        self.backbone = Backbone(self.in_channel, self.hidden_dims)
        self.encoder_z = Encoder_z(self.hidden_dims[-1], self.d_z)
        self.encoder_c = Encoder_c(self.hidden_dims[-1], self.d_z, self.d_c)
        self.decoder = Decoder(self.d_z + self.d_c, self.hidden_dims)

    def forward(self, x):
        x = self.backbone(x)

        mu_z, log_var_z = self.encoder_z(x)
        z = reparameter(mu_z, log_var_z)

        mu_c, log_var_c = self.encoder_c(x, z)
        c = reparameter(mu_c, log_var_c)

        output = self.decoder(z, c)

        return output, z, c, mu_z, log_var_z, mu_c, log_var_c

    def generate(self, z, c):
        output = self.decoder(z, c)
        return output


class DualEncodersDigits:
    def __init__(self, device, shared_module, optimizer_func):
        self.device = device
        self.shared_list = shared_module
        self.optimizer = optimizer_func

        self.in_channel = 3
        # self.hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = [8, 16, 32, 64, 128]
        self.d_z = 2
        self.d_c = 2

        self.lbd_dec = 1
        self.lbd_z = 0.01
        self.lbd_c = 0.01
        self.xi = 0.5

        self.model = DualEncoder(self.in_channel, self.hidden_dims, self.d_z, self.d_c)
        self.round = -1
        self.client_id = -1
        self.n_resample = 1

    def fit(self, client_id, tr_loader, epoch_encoder, epoch_decoder, criterion, lr):
        self.client_id = client_id
        self.model = self.model.to(self.device)
        self.model.train()

        criterion_dec = criterion
        optimizer_backbone = self.optimizer(self.model.backbone.parameters(), lr=lr)
        optimizer_encoder_z = self.optimizer(self.model.encoder_z.parameters(), lr=lr)
        optimizer_encoder_c = self.optimizer(self.model.encoder_c.parameters(), lr=lr)
        optimizer_dec = self.optimizer(self.model.decoder.parameters(), lr=lr)

        for ep in range(epoch_encoder):
            logging.info("Round: {:d}, Client: {:d}, Epoch: {:d}".format(self.round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch: {:d}".format(self.round, self.client_id, ep))
            for b_id, data in enumerate(tr_loader):
                # loading data
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                x = x.repeat(self.n_resample, 1, 1, 1)
                y = y.repeat(self.n_resample, 1, 1, 1)

                optimizer_backbone.zero_grad()
                optimizer_encoder_z.zero_grad()
                optimizer_encoder_c.zero_grad()
                optimizer_dec.zero_grad()

                x_hat, z, c, mu_z, log_var_z, mu_c, log_var_c = self.model(x)

                # loss
                loss_dec = criterion_dec(x_hat, x)
                # loss_diff = criterion_dec(x_c, x_z)

                mu_z_prior = torch.zeros_like(mu_z, dtype=torch.float)
                log_var_z_prior = torch.zeros_like(log_var_z, dtype=torch.float)
                mu_c_prior = torch.zeros_like(mu_c, dtype=torch.float)
                log_var_c_prior = torch.zeros_like(log_var_c, dtype=torch.float)

                loss_dkl_z = loss_dkl(mu_c, log_var_z, mu_z_prior, log_var_z_prior)
                loss_dkl_c = loss_dkl(mu_c, log_var_c, mu_c_prior, log_var_c_prior)
                loss_constr_c = loss_reg_c(mu_c, log_var_c)

                loss = self.lbd_dec * loss_dec \
                    + self.lbd_z * loss_dkl_z \
                    + self.lbd_c * loss_dkl_c \
                    + self.lbd_c * F.relu(self.xi + loss_constr_c - loss_dkl_c)

                loss = torch.mean(loss, dim=0)

                logging.info(loss.item())
                print(loss.item())
                loss.backward()

                optimizer_backbone.step()
                optimizer_encoder_z.step()
                optimizer_encoder_c.step()
                optimizer_dec.step()

    def generate(self, z, c):
        self.model = self.model.to(self.device)
        self.model.eval()
        z, c = z.to(self.device), c.to(self.device)
        return self.model.generate(z, c)

    def update_model(self, model_source):
        for named_para_target, named_para_source in zip(self.model.named_parameters(),
                                                        model_source.model.named_parameters()):
            if named_para_source[0].split(".")[0] in self.shared_list:
                named_para_target[1].data = named_para_source[1].detach().clone().data
        return self.model

    def evaluate(self, client_id, ts_loader, criterion):
        pass

