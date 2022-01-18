import os

import numpy as np

import matplotlib.pyplot as plt

import torch

from utils.digit_loaders import prepare_digits
from Server import aggregate
from models.DualEncodersDigits import DualEncodersDigits


class args:
    use_cuda = True
    device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'

    percent = 0.1
    batch = 64
    learning_rate = 0.001
    n_rounds = 10
    epoch_encoder = 1
    epoch_decoder = 1

    shared_modules = {'backbone', 'encoder_c', 'encoder_z'}
    optimizer_func = torch.optim.Adam
    criterion = torch.nn.MSELoss()


# init data loaders
tr_loaders, ts_loaders = prepare_digits(args)
n_tr_samples = [len(tr.dataset) for tr in tr_loaders]

client_root = "clients/digits/"
client_ids = [c_id for c_id, _ in enumerate(tr_loaders)]

# init models
global_model = DualEncodersDigits(-1, args.shared_modules, args.optimizer_func, args.criterion)
global_model.model = global_model.model.to(args.device)
client_models = [DualEncodersDigits(client, args.shared_modules, args.optimizer_func, args.criterion)
                 for client in client_ids]
# allocate global_model to each client to ensure the encoders are trained from the same initialization
for client_model in client_models:
    client_model.update_model(global_model)

# communication rounds
for r in range(args.n_rounds):
    print("round: " + str(r))
    for client_id, (tr_loader, ts_loader, n_sample, client_model) in \
            enumerate(zip(tr_loaders, ts_loaders, n_tr_samples, client_models)):
        print("client: " + str(client_id))
        epoch_loss = client_model.fit(args.device
                                      , r
                                      , tr_loader
                                      , args.epoch_encoder
                                      , args.epoch_decoder
                                      , args.learning_rate)
        outputs = client_model.evaluate(client_id, ts_loader, args.criterion)
    for para in global_model.model.parameters():
        para.data = torch.zeros_like(para.data)
    global_model = aggregate(global_model, client_models, n_tr_samples)
    for client_model in client_models:
        client_model.update_model(global_model)

"""
# evaluate
ts_loader = ts_loaders[0]
DEMnist = client_models[0]
i = np.random.randint(1000)
for data in ts_loader:
    x, y = data
    x_hat = DEMnist.model(x.to(args.device))
    rec = x_hat[0].detach().to("cpu")
    img, rec = x[i], rec[i]
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("origin")
    plt.imshow(img.squeeze(dim=0).numpy())
    plt.subplot(1, 2, 2)
    plt.title("rec")
    plt.imshow(rec.squeeze(dim=0).numpy())
    plt.show()
    break
"""
