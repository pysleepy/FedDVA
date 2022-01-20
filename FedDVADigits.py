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

    batch = 64
    learning_rate = 0.001
    n_rounds = 10
    epoch_encoder = 5
    epoch_decoder = 5

    d_z = 2
    d_c = 2
    xi = 0.5
    lbd_dec = 28 * 28
    lbd_z = 0.001
    lbd_c = 0.001
    lbd_cc = 1
    n_resamples = 1

    shared_modules = {'backbone_z', 'backbone_c', 'encoder_c', 'encoder_z'}
    optimizer_func = torch.optim.Adam
    criterion = torch.nn.MSELoss()


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


local_batch_size = 64
tr_loaders = [torch.utils.data.DataLoader(tr_dataset, batch_size=local_batch_size, shuffle=True)
              for tr_dataset in tr_datasets]
n_tr_samples = [len(tr_dataset) for tr_dataset in tr_datasets]

ts_loaders = [torch.utils.data.DataLoader(ts_dataset, batch_size=1000)
              for ts_dataset in ts_datasets]


# init models
global_model = DualEncodersDigits(-1, args.shared_modules.union('decoder_z', 'decoder_c'), args.optimizer_func, args.criterion
                                  , args.d_z, args.d_c, args.xi
                                  , args.lbd_dec, args.lbd_z, args.lbd_c, args.lbd_cc, args.n_resamples)
global_model.model = global_model.model.to(args.device)
client_models = [DualEncodersDigits(client, args.shared_modules.union('decoder_z', 'decoder_c'), args.optimizer_func, args.criterion
                                    , args.d_z, args.d_c, args.xi
                                    , args.lbd_dec, args.lbd_z, args.lbd_c, args.lbd_cc, args.n_resamples)
                 for client in client_ids]
# allocate global_model to each client to ensure the encoders are trained from the same initialization
for client_model in client_models:
    client_model.update_model(global_model)
    client_model.shared_list = args.shared_modules

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


ts_loader = ts_loaders[0]
DEMnist = client_models[0]
i = np.random.randint(64)
for data in ts_loader:
    x, y = data
    x_given_z, x_given_c, z, c, mu_z, log_var_z, mu_c, log_var_c = DEMnist.model(x.to(args.device))
    rec_z = x_given_z.detach().to("cpu")
    rec_c = x_given_c.detach().to("cpu")
    img, rec_z, rec_c = np.transpose(x[i].numpy(), [1, 2, 0])\
        , np.transpose(rec_z[i].numpy(), [1, 2, 0])\
        , np.transpose(rec_c[i].numpy(), [1, 2, 0])
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.title("origin")
    plt.imshow(img)
    plt.subplot(1, 4, 2)
    plt.title("rec_z")
    plt.imshow(rec_z)
    plt.subplot(1, 4, 3)
    plt.title("rec_c")
    plt.imshow(rec_c)
    plt.subplot(1, 4, 4)
    plt.title("rec")
    plt.imshow(rec_z+rec_c)
    plt.show()
    break
