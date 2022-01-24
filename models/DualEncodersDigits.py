import logging

import numpy as np

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
        cur_layer = nn.Sequential(nn.Linear(self.d_in, self.hidden_dims[0])
                                  , nn.ReLU())
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
        last_layer = nn.Sequential(nn.Linear(self.hidden_dims[0], 28 * 28 * self.out_channels)
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

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x_z = self.backbone_z(x)
        mu_z, log_var_z = self.encoder_z(x_z)
        z = reparameter(mu_z, log_var_z)
        x_given_z = self.decoder_z(z).view(-1, 1, 28, 28)

        x_c = self.backbone_c(torch.cat([x, z.detach()], dim=1))
        mu_c, log_var_c = self.encoder_c(x_c)
        c = reparameter(mu_c, log_var_c)
        x_given_c = self.decoder_c(torch.cat([z.detach(), c], dim=1)).view(-1, 1, 28, 28)

        return x_given_z, x_given_c, z, c, mu_z, log_var_z, mu_c, log_var_c

    def generate(self, z, c):
        output_z = self.decoder_z(z)
        output_c = self.decoder_c(torch.cat([z, c], dim=1))
        return output_z, output_c


class DualEncodersDigits:
    def __init__(self, client_id, shared_module, optimizer_func, criterion
                 , d_z, d_c, xi, lbd_dec_z, lbd_dec_c, lbd_z, lbd_c, lbd_cc, n_resample, n_channels):
        self.shared_list = shared_module
        self.optimizer = optimizer_func
        self.criterion_dec_z = criterion
        self.criterion_dec_c = criterion

        self.in_channel = n_channels
        # self.hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = [512]
        # self.hidden_dims = [8, 16, 32, 64, 128]
        self.d_z = d_z
        self.d_c = d_c

        self.lbd_dec_z = lbd_dec_z
        self.lbd_dec_c = lbd_dec_c
        self.lbd_z = lbd_z
        self.lbd_c = lbd_c
        self.lbd_cc = lbd_cc
        self.xi = xi
        self.n_resample = n_resample

        self.model = DualEncoder(self.in_channel, self.hidden_dims, self.d_z, self.d_c)
        self.round = []
        self.client_id = client_id

    def fit(self, device, round, tr_loader, epoch_encoder, epoch_decoder, lr):
        self.model = self.model.to(device)
        self.round.append(round)
        self.model.train()

        optimizer_backbone_z = self.optimizer(self.model.backbone_z.parameters(), lr=lr)
        optimizer_backbone_c = self.optimizer(self.model.backbone_c.parameters(), lr=lr)
        optimizer_encoder_z = self.optimizer(self.model.encoder_z.parameters(), lr=lr)
        optimizer_encoder_c = self.optimizer(self.model.encoder_c.parameters(), lr=lr)
        optimizer_dec_z = self.optimizer(self.model.decoder_z.parameters(), lr=lr)
        optimizer_dec_c = self.optimizer(self.model.decoder_c.parameters(), lr=lr)

        logging.info("Optimizing Decoder")
        print("Optimizing Decoder")
        for ep in range(epoch_decoder):
            logging.info("Round: {:d}, Client: {:d}, Epoch Dec_z: {:d}".format(round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch Dec_z: {:d}".format(round, self.client_id, ep))
            epoch_loss_z = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                x, y = data
                x, y = x.to(device), y.to(device)
                x = x.repeat(self.n_resample, 1, 1, 1)
                y = y.repeat(self.n_resample, 1, 1, 1)

                optimizer_dec_z.zero_grad()
                optimizer_dec_c.zero_grad()
                x_given_z, x_given_c, _, _, _, _, _, _ = self.model(x)

                # rec loss
                loss_dec_z = self.criterion_dec_z(x_given_z, x)
                epoch_loss_z.append(loss_dec_z.item())
                loss_dec_z = self.lbd_dec_z * loss_dec_z
                loss_dec_z.backward()
                optimizer_dec_z.step()

            logging.info('Epoch Decoder_z Loss: ' + str(np.mean(epoch_loss_z)))
            print('Epoch Decoder_z Loss: ' + str(np.mean(epoch_loss_z)))

        for ep in range(epoch_decoder):
            logging.info("Round: {:d}, Client: {:d}, Epoch Dec_c: {:d}".format(round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch Dec_c: {:d}".format(round, self.client_id, ep))
            epoch_loss_c = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                x, y = data
                x, y = x.to(device), y.to(device)
                x = x.repeat(self.n_resample, 1, 1, 1)
                y = y.repeat(self.n_resample, 1, 1, 1)

                optimizer_dec_c.zero_grad()
                x_given_z, x_given_c, _, _, _, _, _, _ = self.model(x)

                # rec loss
                loss_dec_c = self.criterion_dec_c(x_given_c, x)
                loss_dec_c = torch.mean(loss_dec_c, dim=0)
                epoch_loss_c.append(loss_dec_c.item())
                loss_dec_c = self.lbd_dec_c * loss_dec_c
                loss_dec_c.backward()
                optimizer_dec_c.step()

            logging.info('Epoch Decoder_c Loss: ' + str(np.mean(epoch_loss_c)))
            print('Epoch Decoder_c Loss: ' + str(np.mean(epoch_loss_c)))

        logging.info("Optimizing Encoder")
        print("Optimizing Encoder")
        for ep in range(epoch_encoder):
            logging.info("Round: {:d}, Client: {:d}, Epoch Enc_z: {:d}".format(round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch Enc_z: {:d}".format(round, self.client_id, ep))
            epoch_dec_z = []
            epoch_dec_c = []
            epoch_dkl_z = []
            epoch_dkl_c = []
            epoch_constr_c = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                x, y = data
                x, y = x.to(device), y.to(device)
                x = x.repeat(self.n_resample, 1, 1, 1)
                y = y.repeat(self.n_resample, 1, 1, 1)

                optimizer_backbone_z.zero_grad()
                optimizer_backbone_c.zero_grad()
                optimizer_encoder_z.zero_grad()
                optimizer_encoder_c.zero_grad()
                optimizer_dec_z.zero_grad()
                optimizer_dec_c.zero_grad()

                x_given_z, x_given_c, z, c, mu_z, log_var_z, mu_c, log_var_c = self.model(x)

                # rec loss
                loss_dec_z = self.criterion_dec_z(x_given_z, x)
                loss_dec_c = self.criterion_dec_c(x_given_c, x)
                # loss_dec = self.criterion_dec(x_given_z, x)
                loss_dec = loss_dec_z + loss_dec_c

                mu_z_prior = torch.zeros_like(mu_z, dtype=torch.float)
                log_var_z_prior = torch.zeros_like(log_var_z, dtype=torch.float)
                mu_c_prior = torch.zeros_like(mu_c, dtype=torch.float)
                log_var_c_prior = torch.zeros_like(log_var_c, dtype=torch.float)

                loss_dkl_z = loss_dkl(mu_c, log_var_z, mu_z_prior, log_var_z_prior)
                loss_dkl_c = loss_dkl(mu_c, log_var_c, mu_c_prior, log_var_c_prior)
                loss_constr_c = loss_reg_c(mu_c, log_var_c)

                epoch_dec_z.append(torch.mean(loss_dec_z, dim=0).item())
                epoch_dec_c.append(torch.mean(loss_dec_c, dim=0).item())
                epoch_dkl_z.append(torch.mean(loss_dkl_z, dim=0).item())
                epoch_dkl_c.append(torch.mean(loss_dkl_c, dim=0).item())
                epoch_constr_c.append(torch.mean(loss_constr_c, dim=0).item())

                loss = self.lbd_dec_z * loss_dec_z \
                    + self.lbd_dec_c * loss_dec_c \
                    + self.lbd_z * loss_dkl_z \
                    + self.lbd_c * loss_dkl_c \
                    + self.lbd_cc * F.relu(self.xi + loss_constr_c - loss_dkl_c)

                loss = torch.mean(loss, dim=0)
                loss.backward()
                optimizer_backbone_z.step()
                optimizer_backbone_c.step()
                optimizer_encoder_z.step()
                optimizer_encoder_c.step()

            logging.info('Epoch Decoder_z Loss: ' + str(np.mean(epoch_dec_z)))
            print('Epoch Decoder Loss_z: ' + str(np.mean(epoch_dec_z)))
            logging.info('Epoch Decoder_c Loss: ' + str(np.mean(epoch_dec_c)))
            print('Epoch Decoder Loss_c: ' + str(np.mean(epoch_dec_c)))
            logging.info('Epoch DKL z Loss: ' + str(np.mean(epoch_dkl_z)))
            print('Epoch DKL z Loss: ' + str(np.mean(epoch_dkl_z)))
            logging.info('Epoch DKL c Loss: ' + str(np.mean(epoch_dkl_c)))
            print('Epoch DKL c Loss: ' + str(np.mean(epoch_dkl_c)))
            logging.info('Epoch Constr c Loss : ' + str(np.mean(epoch_constr_c)))
            print('Epoch Constr c Loss : ' + str(np.mean(epoch_constr_c)))

    def generate(self, device, z, c):
        self.model = self.model.to(device)
        self.model.eval()
        z, c = z.to(device), c.to(device)
        return self.model.generate(z, c)

    def update_model(self, model_source):
        for named_para_target, named_para_source in zip(self.model.named_parameters(),
                                                        model_source.model.named_parameters()):
            if named_para_source[0].split(".")[0] in self.shared_list:
                named_para_target[1].data = named_para_source[1].detach().clone().data
        return self.model

    def evaluate(self, client_id, ts_loader, criterion):
        pass

