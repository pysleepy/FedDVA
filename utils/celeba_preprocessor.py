import io
import zipfile

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import numpy as np

import logging


def preprocess_domain_net():
    with zipfile.ZipFile('data/Images/DimainNet/clipart.zip') as thezip:
        with thezip.open('clipart/aircraft_carrier/clipart_001_000018.jpg', mode='r') as thefile:
            x = thefile.read()
    imageStream = io.BytesIO(x)
    imageFile = Image.open(imageStream)
    print(imageFile.size)


path_list_attrs = "./data/Images/CelebA/Anno/list_attr_celeba.txt"
root_images = "./clients/data/Images/CelebA/img_align_celeba"
root_clients = "./clients/CelebA"

client_attrs = {'Wavy_Hair'
                , 'Bald'
                , 'Wearing_Hat'
                , 'Eyeglasses'
                , 'Bushy_Eyebrows'
                , 'Blond_Hair'
                , 'Wearing_Earrings'}
n_tr = 2000
n_ts = 500


def allocate_celebA(path_list_attrs, client_attrs, n_tr, n_ts):
    img_id_list = []
    img_attr_list = []
    client_tr_list = []
    client_ts_list = []
    with open(path_list_attrs, 'r') as f:
        cnt = int(f.readline().strip())
        attr_list = [a for a in f.readline().strip().split()]
        attr_id_list = [attr_list.index(a) for a in client_attrs]
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

    return client_tr_list, client_ts_list


class FedDataset(Dataset):
    def __init__(self, c_id, dataset, tensor_data, img_size, is_training):
        """

        :param c_id:
        :param dataset: dataset name. enumerator DatasetName
        :param tensor_data: a tensor of samples [n_sample * h * w * channel]
        :param img_size: (h * w * c)
        :param is_training: is training set
        """
        self.c_id = c_id
        self.dataset = dataset
        self.tensor_data = tensor_data
        self.is_training = is_training
        self.img_size = img_size
        self.transformer = transforms.Compose([transforms.Resize(self.img_size)
                                               , transforms.CenterCrop(self.img_size)
                                               , transforms.ToTensor()
                                               , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img = self.tensor_data[index]
        img = self.transformer(img)
        return img

    def __len__(self):
        return self.tensor_data.shape[0]


class ClientDataset:
    def __init__(self, client_id, dataset
                 , client_tr_idx, client_ts_idx
                 , client_tr_data, client_ts_data):
        """

        :param client_id:
        :param dataset: dataset name. enumerator DatasetName
        :param client_tr_idx: a list of tr samples index: [img_id_list]
        :param client_ts_idx: a list of ts samples index: [img_id_list]
        :param client_tr_data: a tensor of tr samples [n_sample * h * w * channel]
        :param client_ts_data: a tensor of ts samples [n_sample * h * w * channel]
        """

        self.client_id = client_id
        logging.info("distribute data to client: {:d}".format(self.client_id))

        self.dataset = dataset
        logging.info("dataset: " + self.dataset.value)

        self.client_tr_idx = client_tr_idx
        self.client_ts_idx = client_ts_idx

        self.client_tr_data = client_tr_data
        self.client_ts_data = client_ts_data

        self.n_tr = len(self.client_tr_data)
        self.n_ts = len(self.client_ts_data)
        self.n_samples = self.n_tr + self.n_ts

        logging.info("n_tr: {:d}, n_ts: {:d}".format(self.n_tr, self.n_ts))

        self.image_size = self.client_tr_data[0].shape

        logging.info("image size: ({:d} * {:d} * {:d})".format(
            self.image_size[0], self.image_size[1], self.image_size[2]))

    def get_fed_dataset(self, is_training):
        if is_training:
            return FedDataset(self.client_id, self.dataset
                              , self.client_tr_data
                              , self.image_size, is_training)
        else:
            return FedDataset(self.client_id, self.dataset
                              , self.client_ts_data
                              , self.image_size, is_training)

