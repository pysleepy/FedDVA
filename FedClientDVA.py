import os
import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
from models.funcs import loss_dkl, loss_reg_c, loss_reg_c_2


class FedClient:
    def __init__(self, client_id, shared_list, optimizer_func, criterion
                 , client_root, model_name, dual_encoder_model
                 , xi, lbd_dec, lbd_z, lbd_c, lbd_cc, fed_classifier=None):
        self.client_id = int(client_id)
        self.shared_list = shared_list
        self.optimizer = optimizer_func
        self.criterion_dec = criterion

        self.client_root = client_root
        self.model_name = model_name
        self.model = dual_encoder_model
        self.classifier = fed_classifier

        self.path_to_snapshots = os.path.join(self.client_root, str(self.client_id), "snapshots")
        if self.client_id != -1 and not os.path.exists(self.path_to_snapshots):
            os.makedirs(self.path_to_snapshots)
        self.path_to_logs = os.path.join(self.client_root, str(self.client_id), "logs")
        if self.client_id != -1 and not os.path.exists(self.path_to_logs):
            os.makedirs(self.path_to_logs)
        if self.client_id != -1:
            tmp_n = 0
            tmp_log_name = self.model_name + "_" + datetime.datetime.today().strftime("log_%Y_%m_%d")
            self.log_name = tmp_log_name + "_{:d}".format(tmp_n)
            while os.path.exists(os.path.join(self.path_to_logs, self.log_name)):
                tmp_n += 1
                self.log_name = tmp_log_name + "_{:d}".format(tmp_n)
            self.logger = logging.getLogger('federation.client:{:d}'.format(self.client_id))
            # create file handler which logs even info messages
            fh = logging.FileHandler(os.path.join(self.path_to_logs, self.log_name))
            # create formatter and add it to the handlers
            formatter = logging.Formatter('%(message)s')
            # add the handlers to the logger
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.info(datetime.datetime.today().strftime("client: {:d} log %Y_%m_%d").format(self.client_id))
            self.logger.info('Initialize client')
            self.logger.info("model name: " + str(self.model_name))
            self.logger.info("model type: " + str(self.model.MODEL_TYPE))
            self.logger.info("d_latent_z: {:d}, d_latent_c: {:d}".format(self.model.d_z, self.model.d_c))

        self.xi = xi
        self.lbd_dec = lbd_dec
        self.lbd_z = lbd_z
        self.lbd_c = lbd_c
        self.lbd_cc = lbd_cc

        self.n_round = 0
        self.n_epc_ec_z = 0
        self.n_epc_ec_c = 0
        self.n_epc_dec = 0

    def fit(self, device, cur_round, tr_loader
            , epoch_encoder_z, epoch_encoder_c, epoch_decoder, lr, n_resamples):
        self.n_round += 1

        self.model = self.model.to(device)
        self.model.train()

        optimizer_backbone_z = self.optimizer(self.model.backbone_z.parameters(), lr=lr)
        optimizer_encoder_z = self.optimizer(self.model.encoder_z.parameters(), lr=lr)

        optimizer_embedding_z = self.optimizer(self.model.embedding_z.parameters(), lr=lr)
        optimizer_embedding_x = self.optimizer(self.model.embedding_x.parameters(), lr=lr)
        optimizer_backbone_c = self.optimizer(self.model.backbone_c.parameters(), lr=lr)
        optimizer_encoder_c = self.optimizer(self.model.encoder_c.parameters(), lr=lr)

        optimizer_decoder = self.optimizer(self.model.decoder.parameters(), lr=lr)

        self.logger.info("Optimizing Decoder")
        for ep in range(epoch_decoder):
            self.logger.info("Round: {:d}, Epoch Dec: {:d}".format(cur_round, ep))
            self.n_epc_dec += 1
            epoch_loss_dec = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                if type(data) == list:
                    x, _ = data
                else:
                    x = data
                x = x.to(device)
                if n_resamples > 0:
                    x = x.repeat(n_resamples, 1, 1, 1)

                optimizer_decoder.zero_grad()
                x_hat, z, c, mu_z, log_var_z, mu_c, log_var_c = self.model(x, False)

                # rec loss
                loss_dec = self.criterion_dec(x_hat, x)
                epoch_loss_dec.append(loss_dec.item())
                loss_dec = self.lbd_dec * loss_dec
                loss_dec.backward()
                optimizer_decoder.step()

            self.logger.info('Epoch Decoder Loss: {:.4f}'.format(np.mean(epoch_loss_dec)))

        self.logger.info("Optimizing Encoder")
        for ep in range(epoch_encoder_z):
            self.logger.info("Round: {:d}, Epoch Enc_z: {:d}".format(cur_round, ep))
            self.n_epc_ec_z += 1
            epoch_dec_z = []
            epoch_dkl_z = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                if type(data) == list:
                    x, _ = data
                else:
                    x = data
                x = x.to(device)
                if n_resamples > 0:
                    x = x.repeat(n_resamples, 1, 1, 1)

                optimizer_backbone_z.zero_grad()
                optimizer_encoder_z.zero_grad()
                optimizer_decoder.zero_grad()

                x_hat, z, c, mu_z, log_var_z, mu_c, log_var_c = self.model(x, False)

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
                # optimizer_decoder.step()
                optimizer_backbone_z.step()
                optimizer_encoder_z.step()

            self.logger.info('Epoch Decoder_z Loss: ' + str(np.mean(epoch_dec_z)))
            self.logger.info('Epoch DKL z Loss: ' + str(np.mean(epoch_dkl_z)))

        for ep in range(epoch_encoder_c):
            self.logger.info("Round: {:d}, Epoch Enc_c: {:d}".format(cur_round, ep))
            self.n_epc_ec_c += 1
            epoch_dec_c = []
            epoch_dkl_c = []
            eooch_dkl_c_local = []
            epoch_constr_c = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                if type(data) == list:
                    x, _ = data
                else:
                    x = data
                x = x.to(device)
                if n_resamples > 0:
                    x = x.repeat(n_resamples, 1, 1, 1)

                optimizer_embedding_z.zero_grad()
                optimizer_embedding_x.zero_grad()
                optimizer_backbone_c.zero_grad()
                optimizer_encoder_c.zero_grad()
                optimizer_decoder.zero_grad()

                x_hat, z, c, mu_z, log_var_z, mu_c, log_var_c = self.model(x, True)

                # rec loss
                loss_dec_c = self.criterion_dec(x_hat, x)
                mu_c_prior = torch.zeros_like(mu_c, dtype=torch.float)
                mu_c_prior_local = torch.ones_like(mu_c) * mu_c.mean(dim=0).detach()
                log_var_c_prior = torch.zeros_like(log_var_c, dtype=torch.float)

                # tmp
                loss_dkl_c = loss_dkl(mu_c, log_var_c, mu_c_prior, log_var_c_prior)  # N(0, 1)
                loss_dkl_c_local = loss_dkl(mu_c, log_var_c, mu_c_prior_local, log_var_c_prior)  # N(mu_c, 1)
                loss_dkl_c_local_2 = loss_dkl(mu_c.detach(), log_var_c.detach(), mu_c_prior_local, log_var_c_prior)  # N(mu_c, 1)
                loss_dkl_c_reverse = loss_dkl(mu_c_prior, log_var_c_prior, mu_c, log_var_c)

                loss_constr_c = loss_reg_c(mu_c, log_var_c)
                loss_constr_c_2 = loss_reg_c_2(mu_c, log_var_c)

                epoch_dec_c.append(torch.mean(loss_dec_c, dim=0).item())
                epoch_dkl_c.append(torch.mean(loss_dkl_c, dim=0).item())
                eooch_dkl_c_local.append(torch.mean(loss_dkl_c_local, dim=0).item())
                epoch_constr_c.append(torch.mean(loss_constr_c, dim=0).item())

                # 2022-02-24 loss = self.lbd_dec * loss_dec_c + self.lbd_c * loss_dkl_c \
                # + self.lbd_cc * F.relu(self.xi + loss_constr_c - loss_dkl_c)
                # 2022-03-03 loss = self.lbd_dec * loss_dec_c + self.lbd_c * loss_dkl_c_local \
                # + self.lbd_cc * F.relu(self.xi + loss_constr_c - loss_dkl_c)
                loss = self.lbd_dec * loss_dec_c + self.lbd_c * loss_dkl_c_local \
                    + self.lbd_cc * F.relu(self.xi + loss_constr_c - loss_dkl_c)

                loss = torch.mean(loss, dim=0)
                loss.backward()
                optimizer_embedding_z.step()
                optimizer_embedding_x.step()
                optimizer_backbone_c.step()
                optimizer_encoder_c.step()
                optimizer_decoder.step()

            self.logger.info('Epoch Decoder_c Loss: ' + str(np.mean(epoch_dec_c)))
            self.logger.info('Epoch DKL c Loss: ' + str(np.mean(epoch_dkl_c)))
            self.logger.info('Epoch DKL c local Loss: ' + str(np.mean(eooch_dkl_c_local)))
            self.logger.info('Epoch Constr c Loss : ' + str(np.mean(epoch_constr_c)))

    def generate(self, device, z, c):
        self.logger.info("generate samples")
        self.model = self.model.to(device)
        self.model.eval()
        z, c = z.to(device), c.to(device)
        return self.model.generate(z, c)

    def fit_classifier(self, device, cur_round, tr_loader, epoch_classifier, lr):
        self.logger.info("Training Classifier")
        self.classifier = self.classifier.to(device)
        self.classifier.train()
        criterion = nn.CrossEntropyLoss()
        optimizer_classifier = self.optimizer(self.model.decoder.parameters(), lr=lr)

        self.logger.info("Optimizing Classifier")
        for ep in range(epoch_classifier):
            self.logger.info("Round: {:d}, Epoch Dec: {:d}".format(cur_round, ep))
            epoch_loss_classifier = []

            correct = 0
            for b_id, data in enumerate(tr_loader):
                x, y = data
                x, y = x.to(device), y.to(device)
                optimizer_classifier.zero_grad()
                x_hat, z, c, mu_z, log_var_z, mu_c, log_var_c = self.model(x, True)
                y_hat = self.classifier(z)
                # classifier loss
                loss_classifier = criterion(y_hat, y)
                epoch_loss_classifier.append(loss_classifier.item())
                loss_classifier.backward()
                optimizer_classifier.step()

                pred = y_hat.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum().item()
            self.logger.info("Epoch Accuracy: {:.4f}".format(correct / len(tr_loader.dataset)))
            self.logger.info('Epoch Classifier Loss: {:.4f}'.format(np.mean(epoch_loss_classifier)))

    def evaluate_classify(self, device, ts_loader, n_resamples):
        self.logger.info("evaluate classifier")
        self.classifier = self.classifier.to(device)
        self.classifier.eval()
        epoch_classifier = []
        criterion = nn.CrossEntropyLoss()
        for b_id, data in enumerate(ts_loader):
            x, y = data
            x, y = x.to(device), y.to(device)
            x_hat, z, c, mu_z, log_var_z, mu_c, log_var_c = self.model(x, True)
            y_hat = self.classifier(z)
            loss_classifier = criterion(y_hat, y)
            epoch_classifier.append(loss_classifier.item())
            self.logger.info('Evaluate Decoder Loss: ' + str(np.mean(epoch_classifier)))
            return x, x_hat, y, y_hat, z, c, mu_z, log_var_z, mu_c, log_var_c

    def update_model(self, model_source):
        self.logger.info("update local models")
        for named_para_target, named_para_source in zip(self.model.named_parameters(),
                                                        model_source.model.named_parameters()):
            if named_para_source[0].split(".")[0] in self.shared_list:
                named_para_target[1].data = named_para_source[1].detach().clone().data
        return self.model

    def evaluate(self, device, ts_loader, n_resamples):
        self.logger.info("evaluate models")
        self.model = self.model.to(device)
        self.model.eval()
        epoch_dec = []
        epoch_dkl_z = []
        epoch_dkl_c = []
        epoch_constr_c = []

        for b_id, data in enumerate(ts_loader):
            # loading data
            if type(data) == list:
                x, _ = data
            else:
                x = data
            x = x.to(device)
            if n_resamples > 0:
                x = x.repeat(n_resamples, 1, 1, 1)

            x_hat, z, c, mu_z, log_var_z, mu_c, log_var_c = self.model(x, True)

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

            self.logger.info('Evaluate Decoder Loss: ' + str(np.mean(epoch_dec)))
            self.logger.info('Evaluate DKL z Loss: ' + str(np.mean(epoch_dkl_z)))
            self.logger.info('Evaluate DKL c Loss: ' + str(np.mean(epoch_dkl_c)))
            self.logger.info('Evaluate Constr c Loss : ' + str(np.mean(epoch_constr_c)))
            return x, x_hat, z, c, mu_z, log_var_z, mu_c, log_var_c

    def upload_model(self):
        self.logger.info("upload local model")
        return self.shared_list, self.model

    def save_model(self, file_name):
        self.logger.info("save models: " + file_name)
        pth_to_file = os.path.join(self.path_to_snapshots, file_name)
        f = {"model_name": self.model_name
             , "log_name": self.log_name
             , "n_round": self.n_round
             , "n_epc_ec_z": self.n_epc_ec_z
             , "n_epc_ec_c": self.n_epc_ec_c
             , "n_epc_dec": self.n_epc_dec
             , "model_state_dict": self.model.state_dict()}
        torch.save(f, pth_to_file)

    def load_model(self, file_name):
        self.logger.info("load models: " + file_name)
        pth_to_file = os.path.join(self.path_to_snapshots, file_name)
        f = torch.load(pth_to_file)
        self.model_name = f["model_name"]
        self.log_name = f["log_name"]
        self.n_round = f["n_round"]
        self.n_epc_ec_z = f["n_epc_ec_z"]
        self.n_epc_ec_c = f["n_epc_ec_c"]
        self.n_epc_dec = f["n_epc_dec"]
        self.model.load_state_dict(f["model_state_dict"])

        self.logger = logging.getLogger('federation.client:{:d}'.format(self.client_id))
        # create file handler which logs even info messages
        fh = logging.FileHandler(os.path.join(self.path_to_logs, self.log_name))
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(message)s')
        # add the handlers to the logger
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.info(datetime.datetime.today().strftime("client: {:d} log %Y_%m_%d").format(self.client_id))
        self.logger.info('Load client')
        self.logger.info("model name: " + str(self.model_name))
        self.logger.info("model type: " + str(self.model.MODEL_TYPE))
        self.logger.info("d_latent_z: {:d}, d_latent_c: {:d}".format(self.model.d_z, self.model.d_c))
