import io
import os
import zipfile
import logging

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

from image_datasets import ImageDatasetName, ClientDatasetDomainNet

logging.getLogger().setLevel(logging.INFO)


def load_domainnet_data(path_img_list, path_zipped_file, label_dict):
    img_list = []
    label_list = []
    with open(path_img_list) as f:
        for line in f:
            img = line.strip().split()[0].strip()
            label_text = img.split("/")[1].strip()
            if label_text in label_dict:
                img_list.append(img)
                label_list.append(label_dict[label_text])

    feature_list = []
    with zipfile.ZipFile(path_zipped_file) as thezip:
        for img, label in zip(img_list, label_list):
            with thezip.open(img, mode='r') as thefile:
                x = thefile.read()
                imageStream = io.BytesIO(x)
                imageFile = Image.open(imageStream)
                # pil_to_tensor = transformer(imageFile)
                # feature_list.append(pil_to_tensor)
                feature_list.append(imageFile)
    # feature_list = torch.cat(feature_list, dim=0)
    return feature_list, label_list


if __name__ == '__main__':
    path_img_list= '../data/Images/DomainNet'
    path_zipped_file = '../data/Images/DomainNet'

    root_clients = "../clients/DomainNet"
    root_images = "data/Images"

    dataset_name = ImageDatasetName.DomainNet

    domain_names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

    label_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5, 'whale': 6
                  , 'windmill': 7, 'wine_glass': 8, 'zebra': 9}

    n_clients = 6
    image_size = 64

    feature_list_tr, label_list_tr = [], []
    feature_list_ts, label_list_ts = [], []
    for c_id in range(n_clients):
        cur_client_path = os.path.join(root_clients, str(c_id), root_images)

        path_client_img_list_tr = os.path.join(path_img_list, domain_names[c_id % 6] + "_train.txt")
        path_client_img_list_ts = os.path.join(path_img_list, domain_names[c_id % 6] + "_test.txt")
        path_client_zipped_file = os.path.join(path_zipped_file, domain_names[c_id % 6] + ".zip")

        client_tr_list, client_tr_label = load_domainnet_data(path_client_img_list_tr, path_client_zipped_file, label_dict)
        client_ts_list, client_ts_label = load_domainnet_data(path_client_img_list_ts, path_client_zipped_file, label_dict)

        print(len(client_tr_list))
        print(len(client_ts_list))
        client_dataset = ClientDatasetDomainNet(c_id, dataset_name
                                                , client_tr_list, client_tr_label
                                                , client_ts_list, client_ts_label
                                                , image_size)
        if not os.path.exists(cur_client_path):
            os.makedirs(cur_client_path)
        torch.save(client_dataset, os.path.join(cur_client_path, dataset_name.value+'.pt'))

        ds = client_dataset.get_fed_dataset(True)
        x = ds[0]
        plt.figure(c_id)
        plt.title("client: " + str(c_id))
        plt.imshow(np.transpose(x[0].numpy(), [1, 2, 0]))
        print("label " + str(x[1]))

    with open(os.path.join(root_clients, "summary.txt"), 'w+') as f:
        f.write("n_classes: {:d}, n_clients: {:d}\n".format(len(label_dict), n_clients))
        f.write(" ".join(list(label_dict.keys())) + "\n")
    plt.show()


    """
    for im in ptt:
        plt.figure()
        plt.imshow(np.transpose(F.relu(im).numpy(), [1, 2, 0]))
    plt.show()
    """