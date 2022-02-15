import os
import logging
import numpy as np
import torch
from torchvision import datasets, transforms

from image_datasets import ImageDatasetName, ClientDataset
from display_utilities import rand_cmap, plot_client_sample_rate

logging.getLogger().setLevel(logging.INFO)


def load_image_data(dataset_name):
    """
    load an image dataset
    :param dataset_name: dataset name. enumerator DatasetName
    :return: tr_data[n_sample, n_row, n_col, n_channel], tr_label[n_samples], ts_data, ts_label
    """
    data_root = "./data/Images"
    logging.info("loading dataset: " + dataset_name.value + " directory: " + "./data/Images/" + dataset_name.value)
    tr_data, tr_label, ts_data, ts_label = None, None, None, None
    if dataset_name == ImageDatasetName.MNIST:
        tr_set = datasets.MNIST(data_root, train=True, download=True)
        ts_set = datasets.MNIST(data_root, train=False, download=True)
        tr_data, tr_label = tr_set.data.unsqueeze(3), tr_set.targets
        ts_data, ts_label = ts_set.data.unsqueeze(3), ts_set.targets
    elif dataset_name == ImageDatasetName.FashionMNIST:
        tr_set = datasets.FashionMNIST(data_root, train=True, download=True)
        ts_set = datasets.FashionMNIST(data_root, train=False, download=True)
        tr_data, tr_label = tr_set.data.unsqueeze(3), tr_set.targets
        ts_data, ts_label = ts_set.data.unsqueeze(3), ts_set.targets
    elif dataset_name == ImageDatasetName.CIFAR10:
        tr_set = datasets.CIFAR10(os.path.join(data_root, dataset_name.value),
                                  train=True, download=True)
        ts_set = datasets.CIFAR10(os.path.join(data_root, dataset_name.value),
                                  train=False, download=True)
        tr_data, tr_label = torch.tensor(tr_set.data), torch.tensor(tr_set.targets)
        ts_data, ts_label = torch.tensor(ts_set.data), torch.tensor(ts_set.targets)
    elif dataset_name == ImageDatasetName.CIFAR100:
        tr_set = datasets.CIFAR100(os.path.join(data_root, dataset_name.value),
                                   train=True, download=True)
        ts_set = datasets.CIFAR100(os.path.join(data_root, dataset_name.value),
                                   train=False, download=True)
        tr_data, tr_label = torch.tensor(tr_set.data), torch.tensor(tr_set.targets)
        ts_data, ts_label = torch.tensor(ts_set.data), torch.tensor(ts_set.targets)
    logging.info("dataset loaded")
    return tr_data, tr_label, ts_data, ts_label


def allocate_image_data(alpha, n_total_clients, tr_label, ts_label):
    """
    allocate data to each client through Dirichlet distribution
    :param alpha: parameter of the Dirichlet distribution. the larger, the more uniform the # of samples per label
    on each client.
    :param n_total_clients: # of clients
    :param tr_label: a tensor of training data. [n_sample]
    :param ts_label: a tensor of test data. [n_sample]
    :return: client_idx_tr_samples: a list of tr samples index: [[[sample index of label 0],[],...,[]]
                                                               , [clietn 2]
                                                               , ...
                                                               , []]
            , client_idx_ts_samples: a list of tr samples index
    """
    # count samples for each class
    n_class = tr_label.unique().shape[0]
    logging.info(str(n_class) + " classes detected")

    n_tr_samples = np.zeros(n_class)
    n_ts_samples = np.zeros(n_class)
    for c in tr_label.unique().tolist():
        n_tr_samples[c] = tr_label[tr_label == c].shape[0]
        n_ts_samples[c] = ts_label[ts_label == c].shape[0]

    log_text = ["class " + str(c_id) + ": n_tr/n_ts: " + str(n_tr) + "/" + str(n_ts)
                for c_id, (n_tr, n_ts) in enumerate(zip(n_tr_samples, n_ts_samples))]
    logging.info("scale of samples per class:\n" + "\n".join(log_text))

    # generate sample distributions
    logging.info("generate class distributions over clients")
    prior = 1.0 * np.ones(n_total_clients) / n_total_clients
    class_dist = np.random.dirichlet(prior * alpha, n_class)  # row * col: n_class * n_client
    n_client_tr_samples = np.floor(class_dist.transpose() * n_tr_samples)  # row * col: n_client * n_class
    n_client_ts_samples = np.floor(class_dist.transpose() * n_ts_samples)  # row * col: n_client * n_class

    # allocate sample index to clients
    logging.info("allocate sample index to clients")
    idx_tr_samples = [np.random.permutation(range(int(n))) for n in n_tr_samples]  # permute tr sample index per class
    idx_ts_samples = [np.random.permutation(range(int(n))) for n in n_ts_samples]  # permute ts sample index per class

    client_idx_tr_samples = []
    client_idx_ts_samples = []
    for client_id, (tr_set, ts_set) in enumerate(zip(n_client_tr_samples, n_client_ts_samples)):
        cur_client_tr = []  # list store idx for current client  [[idx for class 0], ..., [], []]
        cur_client_ts = []  # list store idx for current client
        for cls_id, (n_sample_tr, n_sample_ts) in enumerate(zip(tr_set, ts_set)):
            cur_client_tr.append(idx_tr_samples[cls_id][:int(n_sample_tr)])  # allocate tr samples
            idx_tr_samples[cls_id] = idx_tr_samples[cls_id][int(n_sample_tr):]  # remove allocated samples from the list

            cur_client_ts.append(idx_ts_samples[cls_id][:int(n_sample_ts)])  # allocate ts samples
            idx_ts_samples[cls_id] = idx_ts_samples[cls_id][int(n_sample_ts):]  # remove allocated samples from the list

        client_idx_tr_samples.append(cur_client_tr)
        client_idx_ts_samples.append(cur_client_ts)

    return client_idx_tr_samples, client_idx_ts_samples


def distribute_allocated_data(client_root, dataset_name
                              , tr_data, tr_label, ts_data, ts_label
                              , client_idx_tr_samples, client_idx_ts_samples):
    """
    distribute data for clients, data augmentation and save allocated data to
    ./[client_root]]/[client_idx]/image/[dataset_name.pt]
    :param client_root: root to save clients
    :param dataset_name: dataset name. enumerator DatasetName
    :param tr_data: a tensor of training data. [n_sample, n_row, n_col (, n_channel)]
    :param tr_label: a tensor of training data. [n_sample]
    :param ts_data: a tensor of test data. [n_sample, n_row, n_col (, n_channel)]
    :param ts_label: a tensor of test data. [n_sample]
    :param client_idx_tr_samples: a list of tr samples index: [[[sample index of label 0],[],...,[]]
                                                               , [client 2]
                                                               , ...
                                                               , []]
    :param client_idx_ts_samples: a list of ts samples index: [[[sample index of label 0],[],...,[]]
                                                               , [client 2]
                                                               , ...
                                                               , []]
    :return: client_sets: a list of ClientDataset
    """
    client_dataset_root = "data/Images/"
    client_sets = []
    # distributing samples to clients and create the dataloader
    logging.info("distributing samples")
    for client_id, (cur_client_tr, cur_client_ts) in enumerate(zip(client_idx_tr_samples, client_idx_ts_samples)):
        cur_client = str(client_id)
        cur_client_path = os.path.join(client_root, cur_client, client_dataset_root)

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

        client_dataset = ClientDataset(client_id=client_id
                                       , dataset=dataset_name
                                       , client_tr_idx=cur_client_tr
                                       , client_ts_idx=cur_client_ts
                                       , client_tr_data=cur_client_tr_data
                                       , client_tr_labels=cur_client_tr_labels
                                       , client_ts_data=cur_client_ts_data
                                       , client_ts_labels=cur_client_ts_labels)
        client_sets.append(client_dataset)
        if not os.path.exists(cur_client_path):
            os.makedirs(cur_client_path)
        torch.save(client_dataset, os.path.join(cur_client_path, dataset_name.value+'.pt'))

    return client_sets


if __name__ == '__main__':
    import shutil
    import matplotlib.pyplot as plt
    client_pth = "./clients/clients_balance_y_10"
    alpha = 10000
    n_clients = 20
    dataset = ImageDatasetName.MNIST

    # if os.path.exists(client_pth):
    #     shutil.rmtree(client_pth)

    tr_data, tr_label, ts_data, ts_label = load_image_data(dataset)
    print(tr_data.shape)
    print(type(tr_data))
    print(tr_label.shape)
    print(type(tr_label))

    client_idx_tr_samples, client_idx_ts_samples = allocate_image_data(alpha, n_clients, tr_label, ts_label)

    client_sets = distribute_allocated_data(client_pth, dataset, tr_data, tr_label, ts_data, ts_label
                                            , client_idx_tr_samples, client_idx_ts_samples)

    vecs = [(c_id, [len(s) for l, s in enumerate(sample_idx)]) for c_id, sample_idx in enumerate(client_idx_tr_samples)]

    cmap, fig1, ax1 = rand_cmap(tr_label.unique().shape[0]+2)
    fig2, ax2 = plot_client_sample_rate(vecs, cmap)
    fig1.savefig(client_pth+"/" + dataset.value + "_colormap.eps")
    fig2.savefig(client_pth+"/" + dataset.value + "_class_distribution.eps")
    fig1.show()
    fig2.show()

    plt.figure()
    i, l = next(iter(client_sets[0].get_fed_dataset(True, False, False)))
    dim = i.shape[0]
    if dim == 1:
        plt.imshow(i.squeeze(0).numpy())
        plt.show()
    else:
        plt.imshow(np.transpose(i.numpy(), (1, 2, 0)))
        plt.show()
    print("heterogeneous type: False")
    print("label:{:d} ".format(l.item()))

    plt.figure()
    i, l = next(iter(client_sets[1].get_fed_dataset(True, True, 'x')))
    dim = i.shape[0]
    if dim == 1:
        plt.imshow(i.squeeze(0).numpy())
        plt.show()
    else:
        plt.imshow(np.transpose(i.numpy(), (1, 2, 0)))
        plt.show()
    print("heterogeneous type: x")
    print("label:{:d} ".format(l.item()))

    plt.figure()
    i, l = next(iter(client_sets[2].get_fed_dataset(True, True, 'x')))
    dim = i.shape[0]
    if dim == 1:
        plt.imshow(i.squeeze(0).numpy())
        plt.show()
    else:
        plt.imshow(np.transpose(i.numpy(), (1, 2, 0)))
        plt.show()
    print("heterogeneous type: x")
    print("label:{:d} ".format(l.item()))

    plt.figure()
    i, l = next(iter(client_sets[3].get_fed_dataset(True, True, 'x')))
    dim = i.shape[0]
    if dim == 1:
        plt.imshow(i.squeeze(0).numpy())
        plt.show()
    else:
        plt.imshow(np.transpose(i.numpy(), (1, 2, 0)))
        plt.show()
    print("heterogeneous type: x")
    print("label:{:d} ".format(l.item()))

    plt.figure()
    i, l = next(iter(client_sets[4].get_fed_dataset(True, True, 'x')))
    dim = i.shape[0]
    if dim == 1:
        plt.imshow(i.squeeze(0).numpy())
        plt.show()
    else:
        plt.imshow(np.transpose(i.numpy(), (1, 2, 0)))
        plt.show()
    print("heterogeneous type: x")
    print("label:{:d} ".format(l.item()))
