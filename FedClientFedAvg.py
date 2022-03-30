import os
import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
from models.funcs import loss_dkl, loss_reg_c, loss_reg_c_2


class FedClient:
    def __init__(self, client_id, shared_list, optimizer_func, criterion_cls
                 , client_root, model_name, fed_model):
        self.client_id = int(client_id)
        self.shared_list = shared_list
        self.optimizer = optimizer_func
        self.criterion_cls = criterion_cls

        self.client_root = client_root
        self.model_name = model_name
        self.model = fed_model

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
            self.logger.info("d_in: {:d},".format(self.model.d_in))

        self.n_round = 0
        self.n_epc_cls = 0
        self.n_epc_ec = 0

    def fit(self, device, cur_round, tr_loader, epoch_backbone, epoch_cls, lr):
        self.n_round += 1

        self.model = self.model.to(device)
        self.model.train()

        optimizer_backbone = self.optimizer(self.model.backbone.parameters(), lr=lr)
        optimizer_encoder = self.optimizer(self.model.encoder.parameters(), lr=lr)
        optimizer_classifier = self.optimizer(self.model.classifier.parameters(), lr=lr)

        self.logger.info("Optimizing Classifier")
        for ep in range(epoch_cls):
            self.logger.info("Round: {:d}, Epoch classifier: {:d}".format(cur_round, ep))
            self.n_epc_cls += 1
            epoch_loss_cls = []
            epoch_correct = 0.0
            epoch_total = 0.0
            for b_id, data in enumerate(tr_loader):
                # loading data
                x, y = data
                x, y = x.to(device), y.to(device)
                optimizer_classifier.zero_grad()
                y_hat = self.model(x, False)

                # rec loss
                loss_cls = self.criterion_cls(y_hat, y)
                epoch_loss_cls.append(loss_cls.item())

                pred = y_hat.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(y.view_as(pred)).sum().item()
                epoch_total += y.shape[0]

                loss_dec = loss_cls
                loss_dec.backward()
                optimizer_classifier.step()

            self.logger.info('Epoch Classifier Loss: {:.4f}'.format(np.mean(epoch_loss_cls)))
            self.logger.info('Epoch Accuracy Loss: {:.4f}'.format(epoch_correct / epoch_total))

        self.logger.info("Optimizing Backbone")
        for ep in range(epoch_backbone):
            self.logger.info("Round: {:d}, Epoch Backbone_z: {:d}".format(cur_round, ep))
            self.n_epc_ec += 1
            epoch_cls = []
            epoch_correct = 0.0
            epoch_total = 0.0

            for b_id, data in enumerate(tr_loader):
                x, y = data
                x, y = x.to(device), y.to(device)
                optimizer_backbone.zero_grad()
                optimizer_encoder.zero_grad()
                optimizer_classifier.zero_grad()

                y_hat = self.model(x, False)

                # cls loss
                loss_cls = self.criterion_cls(y_hat, y)
                epoch_cls.append(torch.mean(loss_cls, dim=0).item())

                pred = y_hat.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(y.view_as(pred)).sum().item()
                epoch_total += y.shape[0]

                loss = loss_cls
                loss = torch.mean(loss, dim=0)
                loss.backward()
                # optimizer_decoder.step()
                optimizer_backbone.step()
                optimizer_encoder.step()

            self.logger.info('Epoch Classifier Loss: ' + str(np.mean(epoch_cls)))
            self.logger.info("Epoch Accuracy Loss: " + str(epoch_correct / epoch_total))

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
        epoch_cls = []
        epoch_correct = 0.0
        epoch_total = 0.0

        for b_id, data in enumerate(ts_loader):
            # loading data
            x, y = data
            x, y = x.to(device), y.to(device)
            y_hat = self.model(x, True)

            # cls loss
            loss_cls = self.criterion_cls(y_hat, y)

            pred = y_hat.argmax(dim=1, keepdim=True)
            epoch_correct += pred.eq(y.view_as(pred)).sum().item()
            epoch_total += y.shape[0]

            epoch_cls.append(torch.mean(loss_cls, dim=0).item())

            self.logger.info("Evaluate Classifier Loss: " + str(np.mean(epoch_cls)))
            self.logger.info("Evaluate Accuracy Loss: " + str(epoch_correct / epoch_total))

            return x, y, y_hat

    def upload_model(self):
        self.logger.info("upload local model")
        return self.shared_list, self.model

    def save_model(self, file_name):
        self.logger.info("save models: " + file_name)
        pth_to_file = os.path.join(self.path_to_snapshots, file_name)
        f = {"model_name": self.model_name
             , "log_name": self.log_name
             , "n_round": self.n_round
             , "n_epc_ec": self.n_epc_ec
             , "n_epc_cls": self.n_epc_cls
             , "model_state_dict": self.model.state_dict()}
        torch.save(f, pth_to_file)

    def load_model(self, file_name):
        self.logger.info("load models: " + file_name)
        pth_to_file = os.path.join(self.path_to_snapshots, file_name)
        f = torch.load(pth_to_file)
        self.model_name = f["model_name"]
        self.log_name = f["log_name"]
        self.n_round = f["n_round"]
        self.n_epc_ec = f["n_epc_ec"]
        self.n_epc_cls = f["n_epc_cls"]
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
        self.logger.info("d_in: {:d}, d_out: {:d}".format(self.model.d_in, self.model.d_out))
