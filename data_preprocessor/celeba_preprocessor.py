import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from models.ImageDatasets import ImageDatasetName, CelebAGenerator

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

logger = logging.getLogger("root")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logger.addHandler(ch)


def load_CelebA_index(path_list_annos):

    """
    load annoed attributes
    :param path_list_annos:
    :return:
    """
    df = []
    col = ["image_id"]
    with open(path_list_annos, 'r') as f:
        n_total_images = int(f.readline().strip())
        attr_list = [attr for attr in f.readline().strip().split()]
        col.extend(attr_list)
        for line in f.readlines():
            line = line.strip().split()
            df.append(line)
    df = pd.DataFrame(df)
    df.columns = col
    df.replace('-1', 0, inplace=True)
    df.replace('1', 1, inplace=True)
    return df


def allocate_samples(df_image_index, target_attrs, n_clients):
    attr_images = dict()
    n_images_per_client = dict()

    clients_each_attr = np.ceil(n_clients / len(target_attrs))
    for attr in target_attrs:
        attr_images[attr] = np.random.permutation(df_image_index["image_id"][df_image_index[attr] == 1].tolist())
        attr_images[attr] = attr_images[attr][:5000]  # 5000 samples at most
        n_images_per_client[attr] = int(np.floor(len(attr_images[attr]) / clients_each_attr))

    client_tr_list = []
    client_ts_list = []
    client_attr_map = []

    for c_id in range(n_clients):
        attr = target_attrs[c_id % len(target_attrs)]
        offset = int(c_id / len(target_attrs))
        imgs = attr_images[attr][offset * n_images_per_client[attr]: (offset+1) * n_images_per_client[attr]]
        client_tr_list.append(imgs[: int(np.round(len(imgs) * 0.75))])
        client_ts_list.append(imgs[int(np.round(len(imgs) * 0.75)):])
        client_attr_map.append([c_id, attr, len(client_tr_list[-1]), len(client_ts_list[-1])])

    return client_attr_map, client_tr_list, client_ts_list


dataset_name = ImageDatasetName.CelebA
data_root = os.path.join(base_path, "data/Images")
path_list_annos = os.path.join(data_root, "CelebA/Anno/list_attr_celeba.txt")
path_to_data = os.path.join(data_root, "CelebA/img_align_celeba")

root_clients = os.path.join(base_path, "clients/CelebA_heter_4")

# target_attr_list = {'Bald', 'Wearing_Hat', 'Eyeglasses', 'Blond_Hair', 'Mustache'}
# target_attr_list = ['Bald', 'Wearing_Hat', 'Receding_Hairline', 'Blond_Hair', 'Black_Hair']  # let clients vary on hairstyle
target_attr_list = ['Bald', 'Wearing_Hat', 'Eyeglasses', 'Blond_Hair']
n_clients = 8


df_image_index = load_CelebA_index(path_list_annos)
client_attr_map, client_tr_list, client_ts_list = allocate_samples(df_image_index, target_attr_list, n_clients)

for c_id in range(n_clients):
    path_to_client_data = os.path.join(root_clients, str(c_id), "data")
    celebAGenerator = CelebAGenerator(c_id
                                      , dataset_name
                                      , path_to_data
                                      , client_tr_list[c_id]
                                      , client_ts_list[c_id]
                                      , 64)

    client_tr_set, client_ts_set = celebAGenerator.get_fed_dataset()
    if not os.path.exists(path_to_client_data):
        os.makedirs(path_to_client_data)

    torch.save(client_tr_set, os.path.join(path_to_client_data, dataset_name.value+'_tr.pt'))
    torch.save(client_ts_set, os.path.join(path_to_client_data, dataset_name.value+'_ts.pt'))

with open(os.path.join(root_clients, "summary.txt"), 'w+') as f:
    for c_id, attr, n_tr, n_ts in client_attr_map:
        f.write("client: {:d}, attribute: {:s}, n_tr: {:d}, n_ts: {:d}\n".format(c_id, attr, n_tr, n_ts))


client_tr_sets = [torch.load(os.path.join(root_clients, str(c_id), "data", dataset_name.value+'_tr.pt'))
                  for c_id in range(n_clients)]


tr_loaders = [torch.utils.data.DataLoader(tr_dataset, batch_size=64, shuffle=True)
              for tr_dataset in client_tr_sets]

client_attr_map = []
with open(os.path.join(root_clients, "summary.txt"), 'r') as f:
    for line in f:
        client_attr_map.append(line.strip().split())

for c_id, tr_set in enumerate(tr_loaders):
    plt.figure(c_id)
    x = next(iter(tr_loaders[c_id]))
    x = x[0]
    plt.imshow(np.transpose(x.numpy(), [1, 2, 0]))
    plt.title(client_attr_map[c_id])
    plt.show()

