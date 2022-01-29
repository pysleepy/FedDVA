import io
import os
import zipfile
import logging

import numpy as np

import matplotlib.pyplot as plt

import torch

from image_datasets import ImageDatasetName, ClientDatasetCelebA

logging.getLogger().setLevel(logging.INFO)


def allocate_celebA(path_list_attrs, selected_attrs, n_tr, n_ts):
    img_id_list = []
    img_attr_list = []

    client_attr_map = dict()

    client_tr_list = []
    client_ts_list = []

    with open(path_list_attrs, 'r') as f:
        cnt = int(f.readline().strip())
        attr_list = [a for a in f.readline().strip().split()]
        attr_id_list = [attr_list.index(a) for a in selected_attrs]

        for line in f.readlines():
            line = line.strip().split()
            img_id, img_attrs = line[0], np.maximum(np.array(line[1:], dtype=int)[attr_id_list], 0).tolist()
            if sum(img_attrs) == 1:
                img_id_list.append(img_id)
                img_attr_list.append(img_attrs)

    img_id_list = np.array(img_id_list)
    img_attr_list = np.array(img_attr_list)
    for a_id, att in enumerate(attr_id_list):
        cur_list = np.random.permutation(img_id_list[img_attr_list[:, a_id] == 1])
        if len(cur_list) >= n_tr + n_ts:
            client_tr_list.append(cur_list[: n_tr].tolist())
            client_ts_list.append(cur_list[n_tr: n_tr+n_ts].tolist())
            client_attr_map[a_id] = attr_list[att]
        else:
            raise IndexError("Insufficient Images. Try to reduce the number of samples or select other attributes")
    print(client_attr_map)
    return client_attr_map, client_tr_list, client_ts_list


if __name__ == '__main__':
    path_list_attrs = "../data/Images/CelebA/Anno/list_attr_celeba.txt"
    root_dataset = "../data/Images/CelebA/img_align_celeba"

    root_images = "data/Images"
    root_clients = "../clients/CelebA"

    client_attrs = {'Bald'
                    , 'Wearing_Hat'
                    , 'Eyeglasses'
                    , 'Blond_Hair'
                    , 'Mustache'}

    n_tr = 2000
    n_ts = 500

    dataset_name = ImageDatasetName.CelebA

    client_attr_map, tr_list, ts_list = allocate_celebA(path_list_attrs, client_attrs, n_tr, n_ts)

    for c_id in range(len(tr_list)):
        cur_client_path = os.path.join(root_clients, str(c_id), root_images)

        plt.figure(c_id)
        client_dataset = ClientDatasetCelebA(c_id, dataset_name, root_dataset, tr_list[c_id], ts_list[c_id], 64)
        if not os.path.exists(cur_client_path):
            os.makedirs(cur_client_path)
        torch.save(client_dataset, os.path.join(cur_client_path, dataset_name.value+'.pt'))


        ds = client_dataset.get_fed_dataset(True)
        x = ds[0]
        plt.imshow(np.transpose(x.numpy(), [1, 2, 0]))
        plt.title(client_attr_map[c_id])

    with open(os.path.join(root_clients, "summary.txt"), 'w+') as f:
        f.write("n_tr: {:d}, n_ts: {:d}, n_clients: {:d}\n".format(n_tr, n_ts, len(tr_list)))
        for c_id in client_attr_map:
            f.write("client: {:d}, attribute: {:s}\n".format(c_id, client_attr_map[c_id]))
    plt.show()
