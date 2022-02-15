import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.functional import F

from FedServer import aggregate
from fed_client_dva_domainnet import FedClient

from data_preprocessor.FocalLoss import FocalLoss


class args:
    shared_list = {'backbone_z', 'backbone_c', 'encoder_c', 'encoder_z', 'embedding_z', 'embedding_x'}
    # criterion = torch.nn.MSELoss()
    optimizer_func = torch.optim.Adam
    criterion = FocalLoss(gamma=5)
    in_h = 64
    in_w = 64
    in_channels = 3
    hidden_dims = [32, 64, 64 * 2, 64 * 4, 64*8]
    d_z = 1
    d_c = 1
    xi = 0.5
    lbd_dec = 1.
    lbd_z = 1.
    lbd_c = 1.
    lbd_cc = 1.

    use_cuda = True
    device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
    batch = 256
    lr = 0.001
    n_rounds = 1
    epoch_encoder_z = 1
    epoch_encoder_c = 1
    epoch_decoder = 1
    n_resamples = 1

    n_clients = 5

# init data loaders
# tr_loaders, ts_loaders = prepare_digits(args)
# n_tr_samples = [len(tr.dataset) for tr in tr_loaders]

client_root = "clients/DomainNet"
client_ids = [c_id for c_id in range(5)]
path_to_data = "data/Images/DomainNet.pt"

tr_datasets = [torch.load(os.path.join(client_root, str(c_id), path_to_data)).get_fed_dataset(True)
               for c_id in client_ids]

ts_datasets = [torch.load(os.path.join(client_root, str(c_id), path_to_data)).get_fed_dataset(False)
               for c_id in client_ids]

tr_loaders = [torch.utils.data.DataLoader(tr_dataset, batch_size=args.batch, shuffle=True)
              for tr_dataset in tr_datasets]
n_tr_samples = [len(tr_dataset) for tr_dataset in tr_datasets]

ts_loaders = [torch.utils.data.DataLoader(ts_dataset, batch_size=1000)
              for ts_dataset in ts_datasets]


# init models
global_model = FedClient(-1, args.shared_list.union({"decoder"}), args.optimizer_func, args.criterion
                           , args.in_h, args.in_w, args.in_channels, args.hidden_dims
                           , args.d_z, args.d_c, args.xi, args.lbd_dec, args.lbd_z, args.lbd_c, args.lbd_cc)
global_model.model = global_model.model.to(args.device)

client_models = [FedClient(int(c_id), args.shared_list.union({"decoder"}), args.optimizer_func, args.criterion
                           , args.in_h, args.in_w, args.in_channels, args.hidden_dims
                           , args.d_z, args.d_c, args.xi, args.lbd_dec, args.lbd_z, args.lbd_c, args.lbd_cc)
                 for c_id in client_ids]

for client_model in client_models:
    client_model.update_model(global_model)
    client_model.shared_list = args.shared_list

# communication rounds
for r in range(args.n_rounds):
    print("round: " + str(r))
    for client_id, (tr_loader, ts_loader, n_sample, client_model) in enumerate(
            zip(tr_loaders, ts_loaders, n_tr_samples, client_models)):
        client_model.fit(args.device, r, tr_loader, args.epoch_encoder_z, args.epoch_encoder_c, args.epoch_decoder
                         , args.lr, args.n_resamples)
    for para in global_model.model.parameters():
        para.data = torch.zeros_like(para.data)
    global_model = aggregate(global_model, client_models, n_tr_samples)
    for client_model in client_models:
        client_model.update_model(global_model)

for r in range(1):
    print("round: " + str(r))
    for client_id, (tr_loader, ts_loader, n_sample, client_model) in enumerate(
            zip(tr_loaders, ts_loaders, n_tr_samples, client_models)):
        client_model.fit(args.device, r, tr_loader, 0, args.epoch_encoder_z, args.epoch_encoder_c
                         , args.lr, args.n_resamples)

tl = ts_loaders[0]
md = client_models[0]
i = np.random.randint(64)
x, x_hat, z,  c, mu_z, log_var_z, mu_c, log_var_c = md.evaluate(args.device, tl, args.n_resamples)

origin = x.to("cpu")
rec = x_hat.detach().to("cpu")
origin, rec, = F.relu(origin[i]).numpy().squeeze(), F.relu(rec[i]).numpy().squeeze()
plt.figure()
plt.subplot(1, 2, 1)
plt.title("origin")
plt.imshow(np.transpose(origin, [1, 2, 0]))
plt.subplot(1, 2, 2)
plt.title("rec")
plt.imshow(np.transpose(rec, [1, 2, 0]))
plt.show()
