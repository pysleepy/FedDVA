import numpy as np
import torch
from torch.functional import F
import logging

from models.DualEncoders import DualEncoder, loss_dkl, loss_reg_c


class FedClient:
    def __init__(self, client_id, shared_list, optimizer_func, criterion
                 , in_h, in_w, in_channels, hidden_dims
                 , d_z, d_c, xi, lbd_dec, lbd_z, lbd_c, lbd_cc):
        self.client_id = int(client_id)
        self.shared_list = shared_list
        self.optimizer = optimizer_func
        self.criterion_dec = criterion

        self.in_h = in_h
        self.in_w = in_w
        self.in_channel = in_channels
        self.hidden_dims = hidden_dims  # [64, 64 * 2, 64 * 4, 64*8]

        self.d_z = d_z
        self.d_c = d_c
        self.xi = xi
        self.lbd_dec = lbd_dec
        self.lbd_z = lbd_z
        self.lbd_c = lbd_c
        self.lbd_cc = lbd_cc

        self.model = DualEncoder(self.in_h, self.in_w, self.in_channel, self.hidden_dims, self.d_z, self.d_c)
        self.round = []

    def fit(self, device, cur_round, tr_loader, epoch_encoder_z, epoch_encoder_c, epoch_decoder, lr, n_resamples):
        self.model = self.model.to(device)
        self.round.append(cur_round)
        self.model.train()

        optimizer_backbone_z = self.optimizer(self.model.backbone_z.parameters(), lr=lr)
        optimizer_encoder_z = self.optimizer(self.model.encoder_z.parameters(), lr=lr)

        optimizer_embedding_z = self.optimizer(self.model.embedding_z.parameters(), lr=lr)
        optimizer_embedding_x = self.optimizer(self.model.embedding_x.parameters(), lr=lr)
        optimizer_backbone_c = self.optimizer(self.model.backbone_c.parameters(), lr=lr)
        optimizer_encoder_c = self.optimizer(self.model.encoder_c.parameters(), lr=lr)

        optimizer_decoder = self.optimizer(self.model.decoder.parameters(), lr=lr)

        logging.info("Optimizing Decoder")
        print("Optimizing Decoder")
        for ep in range(epoch_decoder):
            logging.info("Round: {:d}, Client: {:d}, Epoch Dec: {:d}".format(cur_round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch Dec: {:d}".format(cur_round, self.client_id, ep))
            epoch_loss_dec = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                x, y = data
                x, y = x.to(device), y.to(device)
                x = x.repeat(n_resamples, 1, 1, 1)
                y = y.repeat(n_resamples, 1, 1, 1)

                optimizer_decoder.zero_grad()
                x_hat, z,  c, mu_z, log_var_z, mu_c, log_var_c = self.model(x, False)

                # rec loss
                loss_dec = self.criterion_dec(x_hat, x)
                epoch_loss_dec.append(loss_dec.item())
                loss_dec = self.lbd_dec * loss_dec
                loss_dec.backward()
                optimizer_decoder.step()

            logging.info('Epoch Decoder Loss: {:.4f}'.format(np.mean(epoch_loss_dec)))
            print('Epoch Decoder Loss: {:.4f}'.format(np.mean(epoch_loss_dec)))

        logging.info("Optimizing Encoder")
        print("Optimizing Encoder")
        for ep in range(epoch_encoder_z):
            logging.info("Round: {:d}, Client: {:d}, Epoch Enc_z: {:d}".format(cur_round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch Enc_z: {:d}".format(cur_round, self.client_id, ep))
            epoch_dec_z = []
            epoch_dkl_z = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                x, y = data
                x, y = x.to(device), y.to(device)
                x = x.repeat(n_resamples, 1, 1, 1)
                y = y.repeat(n_resamples, 1, 1, 1)

                optimizer_backbone_z.zero_grad()
                optimizer_encoder_z.zero_grad()
                optimizer_decoder.zero_grad()

                x_hat, z,  c, mu_z, log_var_z, mu_c, log_var_c = self.model(x, False)

                # rec loss
                loss_dec_z = self.criterion_dec(x_hat, x)
                mu_z_prior = torch.zeros_like(mu_z, dtype=torch.float)
                log_var_z_prior = torch.zeros_like(log_var_z, dtype=torch.float)
                loss_dkl_z = loss_dkl(mu_z, log_var_z, mu_z_prior, log_var_z_prior)

                epoch_dec_z.append(torch.mean(loss_dec_z, dim=0).item())
                epoch_dkl_z.append(torch.mean(loss_dkl_z, dim=0).item())

                loss = self.lbd_dec * loss_dec_z + self.lbd_z * loss_dkl_z
                loss = torch.mean(loss, dim=0)
                loss.backward()
                optimizer_backbone_z.step()
                optimizer_encoder_z.step()

            logging.info('Epoch Decoder_z Loss: ' + str(np.mean(epoch_dec_z)))
            print('Epoch Decoder Loss_z: ' + str(np.mean(epoch_dec_z)))
            logging.info('Epoch DKL z Loss: ' + str(np.mean(epoch_dkl_z)))
            print('Epoch DKL z Loss: ' + str(np.mean(epoch_dkl_z)))

        for ep in range(epoch_encoder_c):
            logging.info("Round: {:d}, Client: {:d}, Epoch Enc_c: {:d}".format(cur_round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch Enc_c: {:d}".format(cur_round, self.client_id, ep))
            epoch_dec_c = []
            epoch_dkl_c = []
            epoch_constr_c = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                x, y = data
                x, y = x.to(device), y.to(device)
                x = x.repeat(n_resamples, 1, 1, 1)
                y = y.repeat(n_resamples, 1, 1, 1)

                optimizer_embedding_z.zero_grad()
                optimizer_embedding_x.zero_grad()
                optimizer_backbone_c.zero_grad()
                optimizer_encoder_c.zero_grad()
                optimizer_decoder.zero_grad()

                x_hat, z,  c, mu_z, log_var_z, mu_c, log_var_c = self.model(x, True)

                # rec loss
                loss_dec_c = self.criterion_dec(x_hat, x)
                mu_c_prior = torch.zeros_like(mu_c, dtype=torch.float)
                log_var_c_prior = torch.zeros_like(log_var_c, dtype=torch.float)
                loss_dkl_c = loss_dkl(mu_c, log_var_c, mu_c_prior, log_var_c_prior)
                loss_constr_c = loss_reg_c(mu_c, log_var_c)

                epoch_dec_c.append(torch.mean(loss_dec_c, dim=0).item())
                epoch_dkl_c.append(torch.mean(loss_dkl_c, dim=0).item())
                epoch_constr_c.append(torch.mean(loss_constr_c, dim=0).item())

                loss = self.lbd_dec * loss_dec_c + self.lbd_c * loss_dkl_c \
                    + self.lbd_cc * F.relu(self.xi + loss_constr_c - loss_dkl_c)
                loss = torch.mean(loss, dim=0)
                loss.backward()
                optimizer_embedding_z.step()
                optimizer_embedding_x.step()
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

    def evaluate(self, device, ts_loader, n_resamples):
        self.model.to(device)
        self.model.eval()
        epoch_dec = []
        epoch_dkl_z = []
        epoch_dkl_c = []
        epoch_constr_c = []

        for b_id, data in enumerate(ts_loader):
            # loading data
            x, y = data
            x, y = x.to(device), y.to(device)
            x = x.repeat(n_resamples, 1, 1, 1)
            y = y.repeat(n_resamples, 1, 1, 1)

            x_hat, z,  c, mu_z, log_var_z, mu_c, log_var_c = self.model(x, True)

            # rec loss
            loss_dec = self.criterion_dec(x_hat, x)

            mu_z_prior = torch.zeros_like(mu_z, dtype=torch.float)
            log_var_z_prior = torch.zeros_like(log_var_z, dtype=torch.float)
            loss_dkl_z = loss_dkl(mu_z, log_var_z, mu_z_prior, log_var_z_prior)

            mu_c_prior = torch.zeros_like(mu_c, dtype=torch.float)
            log_var_c_prior = torch.zeros_like(log_var_c, dtype=torch.float)
            loss_dkl_c = loss_dkl(mu_c, log_var_c, mu_c_prior, log_var_c_prior)
            loss_constr_c = loss_reg_c(mu_c, log_var_c)

            epoch_dec.append(torch.mean(loss_dec, dim=0).item())
            epoch_dkl_z.append(torch.mean(loss_dkl_z, dim=0).item())
            epoch_dkl_c.append(torch.mean(loss_dkl_c, dim=0).item())
            epoch_constr_c.append(torch.mean(loss_constr_c, dim=0).item())

            logging.info('Epoch Decoder_c Loss: ' + str(np.mean(epoch_dec)))
            print('Epoch Decoder Loss_c: ' + str(np.mean(epoch_dec)))
            logging.info('Epoch DKL z Loss: ' + str(np.mean(epoch_dkl_z)))
            print('Epoch DKL z Loss: ' + str(np.mean(epoch_dkl_z)))
            logging.info('Epoch DKL c Loss: ' + str(np.mean(epoch_dkl_c)))
            print('Epoch DKL c Loss: ' + str(np.mean(epoch_dkl_c)))
            logging.info('Epoch Constr c Loss : ' + str(np.mean(epoch_constr_c)))
            print('Epoch Constr c Loss : ' + str(np.mean(epoch_constr_c)))
            return x, x_hat, z,  c, mu_z, log_var_z, mu_c, log_var_c

    def upload_model(self):
        return self.shared_list, self.model