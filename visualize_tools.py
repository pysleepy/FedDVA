import os.path

import numpy as np

import torch
import matplotlib.pyplot as plt


def generate_coordinates(n_row, n_col, min_z, max_z, min_c, max_c):
    # generate coordinates
    z_labels = []
    c_labels = []

    z = []
    c = []

    steps_z = (max_z - min_z) / (n_row - 1)
    steps_c = (max_c - min_c) / (n_col - 1)

    for row in range(n_row):
        z_labels.append("{:.2f}".format(min_z + row * steps_z))
        for col in range(n_col):
            z.append([min_z + row * steps_z])
            c.append([min_c + col * steps_c])
            if row == n_row - 1:
                c_labels.append("{:.2f}".format(min_c + col * steps_c))

    z, c = torch.tensor(z), torch.tensor(c)

    return z, c, z_labels, c_labels


loss_types = ["Decoder", "Decoder_z", "DKL z", "Decoder_c", "DKL c", "Constr c"]


def parse_logs(client_root, c_id, log_name, loss_types):
    path_to_log = os.path.join(client_root, str(c_id), "logs", log_name)
    epc_per_round = 5
    loss = dict()
    tmp_loss = dict()

    for tp in loss_types:
        loss[tp] = []
        tmp_loss[tp] = []

    with open(path_to_log) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Epoch"):
                line = line.split("Epoch")[1].strip()
                k, v = line.split("Loss")
                k, v = k.strip(), float(v.strip().strip(":").strip())
                tmp_loss[k].append(v)
                if len(tmp_loss[k]) % epc_per_round == 0:
                    loss[k].append(np.mean(tmp_loss[k]))
                    tmp_loss[k] = []

        plt.figure()
        plt.title(log_name + " " + "Client: {:d}".format(c_id))
        for k in loss_types:
            plt.plot(loss[k], label=k)
        # plt.legend(handles=[l1, l2], labels=['dec_c', 'dec_z'], loc='best')

        plt.legend()
        plt.show()
