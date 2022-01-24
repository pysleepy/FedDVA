import numpy as np
import torch
from torch.functional import F
import logging

from models.DualEncodersDigits import DualEncoder, loss_dkl, loss_reg_c


class FedClient:
    def __init__(self, client_id, shared_module, optimizer_func, criterion
                 , d_z, d_c, xi, lbd_dec_z, lbd_dec_c, lbd_z, lbd_c, lbd_cc, n_resample, n_channels):
        self.shared_list = shared_module
        self.optimizer = optimizer_func
        self.criterion_dec_z = criterion
        self.criterion_dec_c = criterion

        self.in_channel = n_channels
        # self.hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = [512, 256, 128]
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
            logging.info("Round: {:d}, Client: {:d}, Epoch Dec: {:d}".format(round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch Dec: {:d}".format(round, self.client_id, ep))
            epoch_loss_z = []
            epoch_loss_c = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                x, y = data
                x, y = x.to(device), y.to(device)
                x = x.repeat(self.n_resample, 1, 1, 1)
                y = y.repeat(self.n_resample, 1, 1, 1)

                optimizer_dec_z.zero_grad()
                optimizer_dec_c.zero_grad()
                x_hat_z, z, mu_z, log_var_z = self.model(x)
                x_hat_c, c, mu_c, log_var_c = self.model(x, z.detach())

                # rec loss
                loss_dec_z = self.criterion_dec_z(x_hat_z, x)
                loss_dec_c = self.criterion_dec_z(x_hat_c, x)
                epoch_loss_z.append(loss_dec_z.item())
                epoch_loss_c.append(loss_dec_c.item())
                loss_dec_z = self.lbd_dec_z * loss_dec_z
                loss_dec_c = self.lbd_dec_c * loss_dec_c
                loss_dec_z.backward()
                loss_dec_c.backward()
                optimizer_dec_z.step()
                optimizer_dec_c.step()

            logging.info('Epoch Decoder_z Loss: {:.4f}, Decoder_z Loss: {:.4f}'.format(
                np.mean(epoch_loss_z), np.mean(epoch_loss_c)))
            print('Epoch Decoder_z Loss: {:.4f}, Decoder_z Loss: {:.4f}'.format(
                np.mean(epoch_loss_z), np.mean(epoch_loss_c)))

        logging.info("Optimizing Encoder")
        print("Optimizing Encoder")
        for ep in range(epoch_encoder):
            logging.info("Round: {:d}, Client: {:d}, Epoch Enc_z: {:d}".format(round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch Enc_z: {:d}".format(round, self.client_id, ep))
            epoch_dec_z = []
            epoch_dkl_z = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                x, y = data
                x, y = x.to(device), y.to(device)
                x = x.repeat(self.n_resample, 1, 1, 1)
                y = y.repeat(self.n_resample, 1, 1, 1)

                optimizer_backbone_z.zero_grad()
                optimizer_encoder_z.zero_grad()
                optimizer_dec_z.zero_grad()

                x_hat_z, z, mu_z, log_var_z = self.model(x)

                # rec loss
                loss_dec_z = self.criterion_dec_z(x_hat_z, x)
                mu_z_prior = torch.zeros_like(mu_z, dtype=torch.float)
                log_var_z_prior = torch.zeros_like(log_var_z, dtype=torch.float)
                loss_dkl_z = loss_dkl(mu_z, log_var_z, mu_z_prior, log_var_z_prior)

                epoch_dec_z.append(torch.mean(loss_dec_z, dim=0).item())
                epoch_dkl_z.append(torch.mean(loss_dkl_z, dim=0).item())

                loss = self.lbd_dec_z * loss_dec_z + self.lbd_z * loss_dkl_z
                loss = torch.mean(loss, dim=0)
                loss.backward()
                optimizer_backbone_z.step()
                optimizer_encoder_z.step()

            logging.info('Epoch Decoder_z Loss: ' + str(np.mean(epoch_dec_z)))
            print('Epoch Decoder Loss_z: ' + str(np.mean(epoch_dec_z)))
            logging.info('Epoch DKL z Loss: ' + str(np.mean(epoch_dkl_z)))
            print('Epoch DKL z Loss: ' + str(np.mean(epoch_dkl_z)))

        for ep in range(epoch_encoder):
            logging.info("Round: {:d}, Client: {:d}, Epoch Enc_c: {:d}".format(round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch Enc_c: {:d}".format(round, self.client_id, ep))
            epoch_dec_c = []
            epoch_dkl_c = []
            epoch_constr_c = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                x, y = data
                x, y = x.to(device), y.to(device)
                x = x.repeat(self.n_resample, 1, 1, 1)
                y = y.repeat(self.n_resample, 1, 1, 1)

                optimizer_backbone_c.zero_grad()
                optimizer_encoder_c.zero_grad()
                optimizer_dec_c.zero_grad()

                x_hat_z, z, mu_z, log_var_z = self.model(x)
                x_hat_c, c, mu_c, log_var_c = self.model(x, z.detach())

                # rec loss
                loss_dec_c = self.criterion_dec_c(x_hat_c, x)
                mu_c_prior = torch.zeros_like(mu_c, dtype=torch.float)
                log_var_c_prior = torch.zeros_like(log_var_c, dtype=torch.float)
                loss_dkl_c = loss_dkl(mu_c, log_var_c, mu_c_prior, log_var_c_prior)
                loss_constr_c = loss_reg_c(mu_c, log_var_c)

                epoch_dec_c.append(torch.mean(loss_dec_c, dim=0).item())
                epoch_dkl_c.append(torch.mean(loss_dkl_c, dim=0).item())
                epoch_constr_c.append(torch.mean(loss_constr_c, dim=0).item())

                loss = self.lbd_dec_c * loss_dec_c + self.lbd_c * loss_dkl_c \
                    + self.lbd_cc * F.relu(self.xi + loss_constr_c - loss_dkl_c)
                loss = torch.mean(loss, dim=0)
                loss.backward()
                optimizer_backbone_c.step()
                optimizer_encoder_c.step()

            logging.info('Epoch Decoder_c Loss: ' + str(np.mean(epoch_dec_c)))
            print('Epoch Decoder Loss_c: ' + str(np.mean(epoch_dec_c)))
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

    def evaluate(self, device, ts_loader):
        epoch_dec_c = []
        epoch_dkl_c = []
        epoch_constr_c = []

        for b_id, data in enumerate(ts_loader):
            # loading data
            x, y = data
            x, y = x.to(device), y.to(device)
            x = x.repeat(self.n_resample, 1, 1, 1)
            y = y.repeat(self.n_resample, 1, 1, 1)

            x_hat_z, z, mu_z, log_var_z = self.model(x)
            x_hat_c, c, mu_c, log_var_c = self.model(x, z.detach())

            # rec loss
            loss_dec_c = self.criterion_dec_c(x_hat_c, x)
            mu_c_prior = torch.zeros_like(mu_c, dtype=torch.float)
            log_var_c_prior = torch.zeros_like(log_var_c, dtype=torch.float)
            loss_dkl_c = loss_dkl(mu_c, log_var_c, mu_c_prior, log_var_c_prior)
            loss_constr_c = loss_reg_c(mu_c, log_var_c)

            epoch_dec_c.append(torch.mean(loss_dec_c, dim=0).item())
            epoch_dkl_c.append(torch.mean(loss_dkl_c, dim=0).item())
            epoch_constr_c.append(torch.mean(loss_constr_c, dim=0).item())

            logging.info('Epoch Decoder_c Loss: ' + str(np.mean(epoch_dec_c)))
            print('Epoch Decoder Loss_c: ' + str(np.mean(epoch_dec_c)))
            logging.info('Epoch DKL c Loss: ' + str(np.mean(epoch_dkl_c)))
            print('Epoch DKL c Loss: ' + str(np.mean(epoch_dkl_c)))
            logging.info('Epoch Constr c Loss : ' + str(np.mean(epoch_constr_c)))
            print('Epoch Constr c Loss : ' + str(np.mean(epoch_constr_c)))
            return x, x_hat_z, z,  mu_z, log_var_z, x_hat_c, c, mu_c, log_var_c

    def upload_model(self):
        return self.shared_list, self.model