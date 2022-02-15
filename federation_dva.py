import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
# torch.cuda.set_device(0)
import torchvision.utils as vutils

from models.DualEncoderMNIST import DualEncoderMNIST
from models.funcs import FocalLoss
from visualize_tools import generate_coordinates
from fed_server import aggregate
from fed_client_dva import FedClient

# create logger with 'spam_application'
logger = logging.getLogger('federation')
logger.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)

d_z = 1
d_c = 1
shared_list = {'backbone_z', 'backbone_c', 'encoder_c', 'encoder_z', 'embedding_z', 'embedding_x'}
optimizer_func = torch.optim.Adam
criterion = FocalLoss(gamma=5)
# criterion = torch.nn.MSELoss()

xi = 0.5
lbd_dec = 1.
lbd_z = 1.
lbd_c = 1.
lbd_cc = 1.

use_cuda = True
device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
batch = 256
lr = 0.001

n_rounds = 2
n_snapshot = 2
epoch_encoder_z = 2
epoch_encoder_c = 2
epoch_decoder = 2
n_resamples = 0

n_clients = 2

client_root = "clients/clients_balance_y_10"
client_ids = [c_id for c_id in range(n_clients)]
path_to_data = "data/Images/MNIST.pt"
model_name = "fed_dva_mnist"

tr_datasets = [torch.load(os.path.join(client_root, str(c_id), path_to_data)).get_fed_dataset(True, False, 'x')
               for c_id in client_ids]

ts_datasets = [torch.load(os.path.join(client_root, str(c_id), path_to_data)).get_fed_dataset(False, False, 'x')
               for c_id in client_ids]

tr_loaders = [torch.utils.data.DataLoader(tr_dataset, batch_size=batch, shuffle=True)
              for tr_dataset in tr_datasets]
n_tr_samples = [len(tr_dataset) for tr_dataset in tr_datasets]

ts_loaders = [torch.utils.data.DataLoader(ts_dataset, batch_size=1000, shuffle=True)
              for ts_dataset in ts_datasets]

# init models
global_model = FedClient(-1, shared_list.union({"decoder"}), optimizer_func, criterion
                           , client_root, DualEncoderMNIST(d_z, d_c)
                           , xi, lbd_dec, lbd_z, lbd_c, lbd_cc)
global_model.model = global_model.model.to(device)

client_models = [FedClient(int(c_id), shared_list.union({"decoder"}), optimizer_func, criterion
                           , client_root, DualEncoderMNIST(d_z, d_c)
                           , xi, lbd_dec, lbd_z, lbd_c, lbd_cc)
                 for c_id in client_ids]

for client_model in client_models:
    client_model.update_model(global_model)
    client_model.shared_list = shared_list

# communication rounds
for r in range(n_rounds):
    logger.info("round: " + str(r))

    for c_id, (tr_loader, client_model) in enumerate(zip(tr_loaders, client_models)):
        client_model.fit(device, r, tr_loader
                         , epoch_encoder_z, epoch_encoder_c, epoch_decoder
                         , lr, n_resamples)

        if r % n_snapshot == (n_snapshot - 1) or r == (n_rounds - 1):
            client_model.save_model(model_name + "_r{:d}".format(r))

    for para in global_model.model.parameters():
        para.data = torch.zeros_like(para.data)

    global_model = aggregate(global_model, client_models, n_tr_samples)

    for client_model in client_models:
        client_model.update_model(global_model)

# check savings
for client_model in client_models:
    client_model.load_model(model_name + "_r{:d}".format(n_rounds - 1))

n_row = 8
n_col = 8

for c_id in client_ids:
    md = client_models[c_id]
    tl = ts_loaders[c_id]

    img, img_hat, z, c, mu_z, log_var_z, mu_c, log_var_c = md.evaluate(device, tl, 0)

    plt.figure()

    # display original images
    plt.subplot(1, 2, 1)
    plt.imshow(
        np.transpose(vutils.make_grid(img[:n_row * n_col].cpu(), nrow=n_col, padding=2, normalize=True), (1, 2, 0)))
    plt.axis("off")
    plt.title("Client {:d} Orig".format(c_id))

    # display rec images
    plt.subplot(1, 2, 2)
    plt.imshow(
        np.transpose(vutils.make_grid(img_hat[:n_row * n_col].detach().cpu(), nrow=n_col, padding=2, normalize=True),
                     (1, 2, 0)))
    plt.axis("off")
    plt.title("Client {:d} Rec".format(c_id))

    """
    plt.figure()
    # display latent distribution
    plt.scatter(mu_z.detach().cpu().squeeze(), mu_c.detach().cpu().squeeze())
    # plt.axis("off")
    plt.title("Client {:d} Distribution".format(c_id))
    """

    plt.show()

# setup
n_row = 8
n_col = 8

min_z, max_z = -2., 2.
min_c, max_c = -4., 4.

z, c, z_labels, c_labels = generate_coordinates(n_row, n_col, min_z, max_z, min_c, max_c)


for c_id in client_ids:
    # generate images
    imgs = client_models[c_id].generate(device, z, c).detach().to("cpu")

    # display images
    plt.figure(figsize=(8,8))

    plt.imshow(np.transpose(vutils.make_grid(imgs, nrow=n_col, padding=2, normalize=True),(1,2,0)))

    axes = plt.gca()
    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()

    x_steps = (x_max - x_min) / n_col
    y_steps = (y_max - y_min) / n_row

    plt.xticks(np.arange(x_min + 0.5 * x_steps, x_max, x_steps), c_labels, rotation=0)
    plt.yticks(np.arange(y_min + 0.5 * y_steps, y_max, y_steps), z_labels, rotation=0)

    # plt.axis("off")
    plt.title("Client {:d} - Generated Images".format(c_id))
    plt.ylabel('Representation z')
    plt.xlabel('Representation c')

    plt.show()

