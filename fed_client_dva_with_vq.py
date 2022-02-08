import numpy as np
import torch
from torch.functional import F
import logging

from models.DualEncodersWithVQ import DualEncoderWithVQ


class FedClient:
    def __init__(self, client_id, shared_module, optimizer_func, criterion
                 , in_h, in_w, in_channels
                 , hidden_dims
                 , k_hidden_z, k_hidden_c
                 , lbd_dec, beta_z, beta_c):
        self.shared_list = shared_module
        self.optimizer = optimizer_func
        self.criterion_dec_z = criterion
        self.criterion_dec_c = criterion

        self.in_h = in_h
        self.in_w = in_w
        self.in_channel = in_channels

        self.hidden_dims = hidden_dims
        self.z_h = int(self.in_h / (2 ** len(self.hidden_dims)))
        self.z_w = int(self.in_w / (2 ** len(self.hidden_dims)))
        self.c_h = int(self.in_h / (2 ** len(self.hidden_dims)))
        self.c_w = int(self.in_w / (2 ** len(self.hidden_dims)))

        self.z_channel = self.hidden_dims[-1]
        self.c_channel = self.hidden_dims[-1]

        self.k_hidden_z = k_hidden_z
        self.k_hidden_c = k_hidden_c

        self.lbd_dec = lbd_dec
        self.beta_z = beta_z
        self.beta_c = beta_c

        self.model = DualEncoderWithVQ(self.in_h, self.in_w, self.in_channel, self.hidden_dims
                                       , self.z_w, self.z_h, self.z_channel
                                       , self.c_w, self.c_h, self.c_channel
                                       , self.k_hidden_z, self.k_hidden_c
                                       , self.beta_z, self.beta_c)
        self.round = []
        self.client_id = client_id

    def fit(self, device, round, tr_loader, epoch_encoder_z, epoch_encoder_c, epoch_decoder, lr):
        self.model = self.model.to(device)
        self.round.append(round)
        self.model.train()

        optimizer_encoder_z = self.optimizer(self.model.encoder_z.parameters(), lr=lr)
        optimizer_vq_z = self.optimizer(self.model.vq_layer_z.parameters(), lr=lr)

        optimizer_embedding_z = self.optimizer(self.model.embedding_z.parameters(), lr=lr)
        optimizer_embedding_x = self.optimizer(self.model.embedding_x.parameters(), lr=lr)
        optimizer_encoder_c = self.optimizer(self.model.encoder_c.parameters(), lr=lr)
        optimizer_vq_c = self.optimizer(self.model.vq_layer_c.parameters(), lr=lr)

        optimizer_dec = self.optimizer(self.model.decoder.parameters(), lr=lr)

        logging.info("Optimizing Decoder")
        print("Optimizing Decoder")
        for ep in range(epoch_decoder):
            logging.info("Round: {:d}, Client: {:d}, Epoch Dec: {:d}".format(round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch Dec: {:d}".format(round, self.client_id, ep))
            epoch_loss = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                if type(data) == list:
                    x, y = data
                    x, y = x.to(device), y.to(device)
                else:
                    x = data.to(device)

                optimizer_dec.zero_grad()
                x_hat, z, c, vq_loss_z, vq_loss_c = self.model(x, False)

                # rec loss
                loss_dec = self.criterion_dec_z(x_hat, x)
                epoch_loss.append(loss_dec.item())
                loss_dec = self.lbd_dec * loss_dec
                loss_dec.backward()
                optimizer_dec.step()

            logging.info('Epoch Decoder Loss: {:.4f}'.format(np.mean(epoch_loss)))
            print('Epoch Decoder Loss: {:.4f}'.format(np.mean(epoch_loss)))

        logging.info("Optimizing Encoder")
        print("Optimizing Encoder")
        for ep in range(epoch_encoder_z):
            logging.info("Round: {:d}, Client: {:d}, Epoch Enc_z: {:d}".format(round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch Enc_z: {:d}".format(round, self.client_id, ep))
            epoch_dec_z = []
            epoch_vq_loss_z = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                if type(data) == list:
                    x, y = data
                    x, y = x.to(device), y.to(device)
                else:
                    x = data.to(device)

                optimizer_encoder_z.zero_grad()
                optimizer_vq_z.zero_grad()

                optimizer_embedding_z.zero_grad()
                optimizer_embedding_x.zero_grad()
                optimizer_encoder_c.zero_grad()
                optimizer_vq_c.zero_grad()
                optimizer_dec.zero_grad()

                x_hat, z, c, vq_loss_z, vq_loss_c = self.model(x, False)

                # rec loss
                loss_dec_z = self.criterion_dec_z(x_hat, x)

                epoch_dec_z.append(torch.mean(loss_dec_z, dim=0).item())
                epoch_vq_loss_z.append(torch.mean(vq_loss_z, dim=0).item())

                loss = self.lbd_dec * loss_dec_z + vq_loss_z
                loss = torch.mean(loss, dim=0)
                loss.backward()
                optimizer_encoder_z.step()
                optimizer_vq_z.step()

            logging.info('Epoch Decoder Loss_z: ' + str(np.mean(epoch_dec_z)))
            print('Epoch Decoder Loss_z: ' + str(np.mean(epoch_dec_z)))
            logging.info('Epoch VQ_z Loss: ' + str(np.mean(epoch_vq_loss_z)))
            print('Epoch VQ_z Loss: ' + str(np.mean(epoch_vq_loss_z)))

        for ep in range(epoch_encoder_c):
            logging.info("Round: {:d}, Client: {:d}, Epoch Enc_c: {:d}".format(round, self.client_id, ep))
            print("Round: {:d}, Client: {:d}, Epoch Enc_c: {:d}".format(round, self.client_id, ep))
            epoch_dec_c = []
            epoch_vq_loss_c = []
            for b_id, data in enumerate(tr_loader):
                # loading data
                if type(data) == list:
                    x, y = data
                    x, y = x.to(device), y.to(device)
                else:
                    x = data.to(device)

                optimizer_encoder_z.zero_grad()
                optimizer_vq_z.zero_grad()

                optimizer_embedding_z.zero_grad()
                optimizer_embedding_x.zero_grad()
                optimizer_encoder_c.zero_grad()
                optimizer_vq_c.zero_grad()
                optimizer_dec.zero_grad()

                x_hat_c, z, c, vq_loss_z, vq_loss_c = self.model(x, True)

                # rec loss
                loss_dec_c = self.criterion_dec_c(x_hat_c, x)
                epoch_dec_c.append(torch.mean(loss_dec_c, dim=0).item())
                epoch_vq_loss_c.append(torch.mean(vq_loss_c, dim=0).item())

                loss = self.lbd_dec * loss_dec_c + vq_loss_c
                loss = torch.mean(loss, dim=0)
                loss.backward()

                optimizer_embedding_z.step()
                optimizer_embedding_x.step()
                optimizer_encoder_c.step()
                optimizer_vq_c.step()
                optimizer_dec.step()

            logging.info('Epoch Decoder Loss_c: ' + str(np.mean(epoch_dec_c)))
            print('Epoch Decoder Loss_c: ' + str(np.mean(epoch_dec_c)))
            logging.info('Epoch VQ_c Loss: ' + str(np.mean(epoch_vq_loss_c)))
            print('Epoch VQ_c Loss: ' + str(np.mean(epoch_vq_loss_c)))

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
        self.model.to(device)
        self.model.eval()
        epoch_dec = []
        epoch_vq_loss_z = []
        epoch_vq_loss_c = []

        for b_id, data in enumerate(ts_loader):
            # loading data
            if type(data) == list:
                x, y = data
                x, y = x.to(device), y.to(device)
            else:
                x = data.to(device)

            x_hat, z, c, vq_loss_z, vq_loss_c = self.model(x, True)

            # rec loss
            loss_dec = self.criterion_dec_c(x_hat, x)
            epoch_dec.append(torch.mean(loss_dec, dim=0).item())
            epoch_vq_loss_z.append(torch.mean(vq_loss_z, dim=0).item())
            epoch_vq_loss_c.append(torch.mean(vq_loss_c, dim=0).item())

            logging.info('Epoch Decoder_c Loss: ' + str(np.mean(epoch_dec)))
            print('Epoch Decoder Loss_c: ' + str(np.mean(epoch_dec)))

            logging.info('Epoch vq_z Loss: ' + str(np.mean(epoch_vq_loss_z)))
            print('Epoch vq_z Loss: ' + str(np.mean(epoch_vq_loss_z)))

            logging.info('Epoch vq_c Loss: ' + str(np.mean(epoch_vq_loss_c)))
            print('Epoch vq_c Loss: ' + str(np.mean(epoch_vq_loss_c)))

            return x, x_hat, z, c

    def upload_model(self):
        return self.shared_list, self.model
