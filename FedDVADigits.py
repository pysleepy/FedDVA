import os
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
global_model = DualEncodersDigits(args.device, args.shared_modules, args.optimizer_func)
client_models = [DualEncodersDigits(args.device, args.shared_modules, args.optimizer_func) for client in client_ids]
# allocate global_model to each client to ensure the encoders are trained from the same initialization
for client_model in client_models:
    client_model.update_model(global_model)

# communication rounds
for r in range(args.n_rounds):
    print("round: " + str(r))
    for client_id, (tr_loader, ts_loader, n_sample, client_model) in \
            enumerate(zip(tr_loaders, ts_loaders, n_tr_samples, client_models)):
        print("client: " + str(client_id))
        epoch_loss = client_model.fit(client_id
                                      , tr_loader
                                      , args.epoch_encoder
                                      , args.epoch_decoder
                                      , args.criterion
                                      , args.learning_rate)
        outputs = client_model.evaluate(client_id, ts_loader, args.criterion)
    for para in global_model.model.parameters():
        para.data = torch.zeros_like(para.data)
    global_model.model = global_model.model.to(args.device)
    global_model = aggregate(global_model, client_models, n_tr_samples)
    for client_model in client_models:
        client_model.update_model(global_model)

