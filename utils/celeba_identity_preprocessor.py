import io
import os
import zipfile
import logging

import numpy as np

import matplotlib.pyplot as plt

import torch

from image_datasets import ImageDatasetName, ClientDatasetCelebA

logging.getLogger().setLevel(logging.INFO)


def allocate_celebA_identity(path_list_identity, path_list_attrs, n_clients, n_tr, n_ts, n_image_perperson):
    person_img_list = dict()

    client_tr_list = []
    client_ts_list = []

    with open(path_list_identity, 'r') as f:
        for line in f:
            img_id, person_id = line.strip().split()
            img_id, person_id = img_id.strip(), person_id.strip()
            if person_id not in person_img_list:
                person_img_list[person_id] = []
            person_img_list[person_id].append(img_id)

    person_ids = []
    for p_id in person_img_list:
        if len(person_img_list[p_id]) >= n_image_perperson:
            person_ids.append(p_id)
    person_ids = np.random.permutation(person_ids)

    ratio_tr = n_tr / (n_tr + n_ts)
    person_per_client = len(person_ids) / n_clients
    for i, p_id in enumerate(person_ids):
        if i % person_per_client == 0:
            client_tr_list.append([])
            client_ts_list.append([])

        cur_img_list = np.random.permutation(person_img_list[p_id])
        tr_images = int(len(cur_img_list) * ratio_tr)
        client_tr_list[-1].extend(cur_img_list[:tr_images].tolist())
        client_ts_list[-1].extend(cur_img_list[tr_images:].tolist())

    with open(path_list_attrs, 'r') as f:
        cnt = int(f.readline().strip())
        attr_list = [a for a in f.readline().strip().split()]
        person_attr_list = dict()

        for line in f.readlines():
            line = line.strip().split()
            img_id, img_attrs = line[0], np.maximum(np.array(line[1:], dtype=int), 0)
            person_attr_list[img_id] = img_attrs

    client_attr_summary = np.zeros([n_clients, len(attr_list)])
    for c_id in range(n_clients):
        for img_id in client_tr_list[c_id]:
            client_attr_summary[c_id] += person_attr_list[img_id]
    return client_tr_list, client_ts_list, attr_list, client_attr_summary


if __name__ == '__main__':
    path_list_identity = "../data/Images/CelebA/Anno/identity_CelebA.txt"
    path_list_attrs = "../data/Images/CelebA/Anno/list_attr_celeba.txt"
    root_dataset = "../data/Images/CelebA/img_align_celeba"

    root_images = "data/Images"
    root_clients = "../clients/CelebA_Identity"

    n_clients = 10
    n_tr = 5000
    n_ts = 1000
    n_image_perperson = 30

    dataset_name = ImageDatasetName.CelebA

    client_tr_list, client_ts_list, attr_list, client_attr_summary = allocate_celebA_identity(path_list_identity
                                                                                              , path_list_attrs
                                                                                              , n_clients
                                                                                              , n_tr
                                                                                              , n_ts
                                                                                              , n_image_perperson)

    for c_id in range(n_clients):
        cur_client_path = os.path.join(root_clients, str(c_id), root_images)

        client_dataset = ClientDatasetCelebA(c_id, dataset_name, root_dataset
                                             , client_tr_list[c_id], client_ts_list[c_id], 64)
        if not os.path.exists(cur_client_path):
            os.makedirs(cur_client_path)
        torch.save(client_dataset, os.path.join(cur_client_path, dataset_name.value+'.pt'))

        ds = client_dataset.get_fed_dataset(True)
        x = ds[0]
        plt.figure(c_id)
        plt.title("client: " + str(c_id))
        plt.imshow(np.transpose(x.numpy(), [1, 2, 0]))

    with open(os.path.join(root_clients, "summary.txt"), 'w+') as f:
        f.write("n_tr: {:d}, n_ts: {:d}, n_clients: {:d}\n".format(n_tr, n_ts, n_clients))
        f.write(" ".join(attr_list) + "\n")
        for c_attr in client_attr_summary:
            f.write(" ".join([str(int(a)) for a in c_attr]))
    plt.show()
