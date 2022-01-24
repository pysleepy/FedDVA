import os

import numpy as np
import matplotlib.pyplot as plt

import torch

from fed_server import aggregate
from fed_client_dva_digits import FedClient

from utils.focal_loss import FocalLoss


class args:
    use_cuda = True
    device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'

    batch = 64
    learning_rate = 0.001
    n_rounds = 1
    epoch_encoder = 1
    epoch_decoder = 1

    d_z = 1
    d_c = 1
    xi = 0.5
    lbd_dec_z = 1.
    lbd_dec_c = 1.
    lbd_z = 1.
    lbd_c = 0.
    lbd_cc = 1.
    n_resamples = 1
    percent = 0.1

    shared_modules = {'backbone_z', 'backbone_c', 'encoder_c', 'encoder_z'}
    optimizer_func = torch.optim.Adam
    # criterion = torch.nn.MSELoss()
    criterion = FocalLoss(gamma=5)


# init data loaders
# tr_loaders, ts_loaders = prepare_digits(args)
# n_tr_samples = [len(tr.dataset) for tr in tr_loaders]

client_root = "clients/clients_balance_y_10"
client_ids = [c_id for c_id in range(5)]
path_to_data = "data/Images/MNIST.pt"

tr_datasets = [torch.load(os.path.join(client_root, str(c_id), path_to_data)).get_fed_dataset(True, False, 'x')
               for c_id in client_ids]

ts_datasets = [torch.load(os.path.join(client_root, str(c_id), path_to_data)).get_fed_dataset(False, False, 'x')
               for c_id in client_ids]

tr_loaders = [torch.utils.data.DataLoader(tr_dataset, batch_size=args.batch, shuffle=True)
              for tr_dataset in tr_datasets]
n_tr_samples = [len(tr_dataset) for tr_dataset in tr_datasets]

ts_loaders = [torch.utils.data.DataLoader(ts_dataset, batch_size=1000)
              for ts_dataset in ts_datasets]


# init models
global_model = FedClient(-1, args.shared_modules.union('decoder_z', 'decoder_c')
                         , args.optimizer_func, args.criterion
                         , args.d_z, args.d_c, args.xi
                         , args.lbd_dec_z, args.lbd_dec_c, args.lbd_z, args.lbd_c, args.lbd_cc
                         , args.n_resamples, 1)
global_model.model = global_model.model.to(args.device)

client_models = [FedClient(c_id, args.shared_modules.union('decoder_z', 'decoder_c')
                           , args.optimizer_func, args.criterion
                           , args.d_z, args.d_c, args.xi
                           , args.lbd_dec_z, args.lbd_dec_c, args.lbd_z, args.lbd_c, args.lbd_cc
                           , args.n_resamples, 1)
                 for c_id in client_ids]

for client_model in client_models:
    client_model.update_model(global_model)
    client_model.shared_list = args.shared_modules

# communication rounds
for r in range(args.n_rounds):
    print("round: " + str(r))
    for client_id, (tr_loader, ts_loader, n_sample, client_model) in enumerate(
            zip(tr_loaders, ts_loaders, n_tr_samples, client_models)):
        client_model.fit(args.device, r, tr_loader, args.epoch_encoder, args.epoch_decoder, args.learning_rate)
    for para in global_model.model.parameters():
        para.data = torch.zeros_like(para.data)
    global_model = aggregate(global_model, client_models, n_tr_samples)
    for client_model in client_models:
        client_model.update_model(global_model)

for r in range(1):
    print("round: " + str(r))
    for client_id, (tr_loader, ts_loader, n_sample, client_model) in enumerate(
            zip(tr_loaders, ts_loaders, n_tr_samples, client_models)):
        client_model.fit(args.device, r, tr_loader, args.epoch_encoder, args.epoch_decoder, args.learning_rate)
    for para in global_model.model.parameters():
        para.data = torch.zeros_like(para.data)
    global_model = aggregate(global_model, client_models, n_tr_samples)

ts_loader = ts_loaders[0]
md = client_models[0]
i = np.random.randint(64)
x, x_hat_z, z,  mu_z, log_var_z, x_hat_c, c, mu_c, log_var_c = md.evaluate(args.device, ts_loader)

x = x.to("cpu")
rec_z = x_hat_z.detach().to("cpu")
rec_c = x_hat_c.detach().to("cpu")
img, rec_z, rec_c = x[i].numpy().squeeze()\
    , rec_z[i].numpy().squeeze()\
    , rec_c[i].numpy().squeeze()
plt.figure()
plt.subplot(2, 2, 1)
plt.title("origin")
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.title("rec_z")
plt.imshow(rec_z)
plt.subplot(2, 2, 3)
plt.title("rec_c")
plt.imshow(rec_c)
plt.subplot(2, 2, 4)
plt.title("rec")
plt.imshow(rec_z+rec_c)
plt.show()
