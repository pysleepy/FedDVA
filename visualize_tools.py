import os.path

import numpy as np

import torch
import matplotlib.pyplot as plt

"""
def saliency(img, model):
    # we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False

    # set model in eval mode
    model.eval()
    # transoform input PIL image to torch.Tensor and normalize
    input = transform(img)
    input.unsqueeze_(0)

    # we want to calculate gradient of higest score w.r.t. input
    # so set requires_grad to True for input
    input.requires_grad = True
    # forward pass to calculate predictions
    preds = model(input)
    score, indices = torch.max(preds, 1)
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    # normalize to [0..1]
    slc = (slc - slc.min()) / (slc.max() - slc.min())

    # apply inverse transform on image
    with torch.no_grad():
        input_img = inv_normalize(input[0])
    # plot image and its saleincy map
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(input_img.detach().numpy(), (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])
    plt.show()
"""


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


loss_types = ["Decoder", "Decoder_z", "DKL z", "Decoder_c", "DKL c", "DKL c local", "Constr c"]


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
                if k in loss_types:
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


def parse_multiple_logs(client_root, c_id, log_names, loss_type):
    epc_per_round = 5
    loss = dict()
    tmp_loss = dict()

    for log_name in log_names:
        print(log_name)
        path_to_log = os.path.join(client_root, str(c_id), "logs", log_name)
        loss[log_name] = []
        tmp_loss[log_name] = []

        with open(path_to_log) as f:
            for line in f:
                line = line.strip()
                if line.startswith("Epoch"):
                    line = line.split("Epoch")[1].strip()
                    k, v = line.split("Loss")
                    k, v = k.strip(), float(v.strip().strip(":").strip())
                    if k == loss_type:
                        tmp_loss[log_name].append(v)
                        if len(tmp_loss[log_name]) % epc_per_round == 0:
                            loss[log_name].append(np.mean(tmp_loss[log_name]))
                            tmp_loss[log_name] = []

        plt.figure()
        plt.title(loss_type + " " + "Client: {:d}".format(c_id))
        for log_name in log_names:
            plt.plot(loss[log_name], label=log_name)
        # plt.legend(handles=[l1, l2], labels=['dec_c', 'dec_z'], loc='best')

        plt.legend()
        plt.show()
