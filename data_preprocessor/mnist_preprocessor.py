import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets

from models.ImageDatasets import ImageDatasetName, MNISTGenerator
from data_preprocessor.display_utilities import rand_cmap, plot_client_sample_rate
from data_preprocessor.allocate_utilities import allocate_supervised_data

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

logger = logging.getLogger("root")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logger.addHandler(ch)

dataset_name = ImageDatasetName.MNIST
n_total_clients = 20
alpha = 10000  # the larger the alpha is, the balance the label distributed on different clients
heter_x = True
heter_y = False
data_root = os.path.join(base_path, "./data/Images")
client_root = os.path.join(base_path, "./clients/mnist_balanced_y_hetero_x")

logger.info("loading dataset: MNIST")
tr_set = datasets.MNIST(data_root, train=True, download=True)
ts_set = datasets.MNIST(data_root, train=False, download=True)
tr_data, tr_label = tr_set.data.unsqueeze(3).permute([0, 3, 1, 2]), tr_set.targets  # N x H x W x C -> N x C x H x W, N
ts_data, ts_label = ts_set.data.unsqueeze(3).permute([0, 3, 1, 2]), ts_set.targets  # N x H x W x C -> N x C x H x W, N

logger.info("allocate samples")
client_idx_tr_samples, client_idx_ts_samples = allocate_supervised_data(alpha, n_total_clients, tr_label, ts_label)
# [[[sample index of label 0],[],...,[]], [clietn 2], ..., []]

logger.info("distribute client datasets")
# distributing samples to clients and create the dataloader
for client_id, (cur_client_tr, cur_client_ts) in enumerate(zip(client_idx_tr_samples, client_idx_ts_samples)):
    logger.info("client: {:d}".format(client_id))
    cur_client = str(client_id)
    cur_client_data_path = os.path.join(client_root, cur_client, "data")
    logger.info("data path: " + cur_client_data_path)

    cur_client_tr_data = []
    cur_client_tr_labels = []
    cur_client_ts_data = []
    cur_client_ts_labels = []

    for cls_id, (tr_samples, ts_samples) in enumerate(zip(cur_client_tr, cur_client_ts)):
        cur_client_tr_data.append(tr_data[tr_label == cls_id][tr_samples])
        cur_client_tr_labels.append(tr_label[tr_label == cls_id][tr_samples])

        cur_client_ts_data.append(ts_data[ts_label == cls_id][ts_samples])
        cur_client_ts_labels.append(ts_label[ts_label == cls_id][ts_samples])

    cur_client_tr_data = torch.cat(cur_client_tr_data, dim=0)
    cur_client_tr_labels = torch.cat(cur_client_tr_labels, dim=0)
    cur_client_ts_data = torch.cat(cur_client_ts_data, dim=0)
    cur_client_ts_labels = torch.cat(cur_client_ts_labels, dim=0)

    mnistGenerators = MNISTGenerator(client_id=client_id
                                     , dataset=dataset_name
                                     , client_tr_data=cur_client_tr_data
                                     , client_tr_labels=cur_client_tr_labels
                                     , client_ts_data=cur_client_ts_data
                                     , client_ts_labels=cur_client_ts_labels)

    client_tr_set, client_ts_set = mnistGenerators.get_fed_dataset(heter_x, heter_y)
    if not os.path.exists(cur_client_data_path):
        os.makedirs(cur_client_data_path)

    torch.save(client_tr_set, os.path.join(cur_client_data_path, dataset_name.value+'_tr.pt'))
    torch.save(client_ts_set, os.path.join(cur_client_data_path, dataset_name.value+'_ts.pt'))

vecs = [(c_id, [len(s) for l, s in enumerate(sample_idx)])
        for c_id, sample_idx in enumerate(client_idx_tr_samples)]
# [(client_0, [n_class_0, n_class_1,...]), (client_1, [n_class_0, n_class_1,...])]

cmap, fig1, ax1 = rand_cmap(tr_label.unique().shape[0] + 2)
fig2, ax2 = plot_client_sample_rate(vecs, cmap)
fig1.savefig(client_root + "/" + dataset_name.value + "_colormap.eps")
fig2.savefig(client_root + "/" + dataset_name.value + "_class_distribution.eps")
# fig1.show()
# fig2.show()
plt.show()

client_tr_sets = [torch.load(os.path.join(client_root, str(c_id), "data", dataset_name.value+'_tr.pt'))
                  for c_id in range(n_total_clients)]

for c_id in range(5):
    idx = np.random.randint(0, 1000)
    plt.figure("client: {:d}".format(c_id))
    img, label = client_tr_sets[c_id].data[idx], client_tr_sets[c_id].labels[idx]
    plt.imshow(img.squeeze(0).numpy())
    plt.title("client {:d}:, class label: {:d}".format(c_id, label.item()))
    plt.show()
