import sys
import os
import numpy as np
from enum import Enum, unique
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as F
import logging

from functools import partial

from data_preprocessor.allocate_utilities import generate_ellipse_marks, generate_sin_marks, generate_line_marks


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

logger = logging.getLogger("root")
logger.setLevel(logging.INFO)


@unique
class ImageDatasetName(Enum):
    MNIST = "MNIST"
    FashionMNIST = "FashionMNIST"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    DomainNet = "DomainNet"
    CelebA = "CelebA"


class FedDataset(Dataset):
    def __init__(self, c_id, dataset_name
                 , tensor_data, tensor_labels
                 , img_size, data_mean, data_std
                 , is_training):
        """

        :param c_id:
        :param dataset: dataset name. enumerator DatasetName
        :param tensor_data: a tensor of samples [N x H x W x C]
        :param tensor_labels: a tensor of labels [N]
        :param img_size: (H x W x C)
        :param data_mean: (mean_c1, mean_c2, ...)
        :param data_std: (std_c1, std_c2, ...)
        :param is_training: is training set
        """

        self.c_id = c_id
        self.dataset_name = dataset_name

        self.data = tensor_data
        self.labels = tensor_labels
        self.n_samples = self.data.shape[0]
        self.shape = self.data.shape

        self.img_size = img_size
        self.data_mean = data_mean
        self.data_std = data_std

        self.is_training = is_training

    def __getitem__(self, index):
        img = self.data[index]
        img = F.normalize(img, self.data_mean, self.data_std)
        if self.labels is None:
            return img
        else:
            label = self.labels[index]
            return img, label

    def __len__(self):
        return self.data.shape[0]


class MNISTGenerator:
    def __init__(self, client_id, dataset_name
                 , client_tr_data, client_tr_labels
                 , client_ts_data, client_ts_labels):
        """

        :param client_id:
        :param dataset_name: dataset name. enumerator DatasetName
        :param client_tr_data: a tensor of tr samples [N x C x H x W]
        :param client_tr_labels: a tensor of tr labels [N]
        :param client_ts_data: a tensor of ts samples [N x C x H x W]
        :param client_ts_labels: a tensor of ts labels [N]
        """

        self.client_id = client_id
        logger.info("distribute data to client: {:d}".format(self.client_id))

        self.dataset_name = dataset_name
        logger.info("dataset: " + self.dataset_name.value)

        self.client_tr_data = client_tr_data
        self.client_tr_labels = client_tr_labels
        self.client_ts_data = client_ts_data
        self.client_ts_labels = client_ts_labels

        self.n_tr = self.client_tr_labels.shape[0]
        self.n_ts = self.client_ts_labels.shape[0]
        self.n_classes = self.client_tr_labels.unique().shape[0]
        logger.info("n_tr: {:d}".format(self.n_tr))
        logger.info("n_ts: {:d}".format(self.n_ts))
        logger.info("n_classes: {:d}".format(self.n_classes))

        tr_mean = (client_tr_data.float() / 255.).mean(dim=[0, 2, 3])
        tr_std = (client_tr_data.float() / 255.).std(dim=[0, 2, 3])
        self.tr_mean = tuple(tr_mean.tolist())
        self.tr_std = tuple(tr_std.tolist())
        self.image_size = self.client_tr_data[0].shape

        logger.info("tr_mean: " + str(self.tr_mean))
        logger.info("tr_std: " + str(self.tr_std))
        logger.info("image size: ({:d} * {:d} * {:d})".format(
            self.image_size[0], self.image_size[1], self.image_size[2]))

    def __generate_marks__(self):
        m_type = int(self.client_id) % (5-1)
        tr_marks = torch.zeros_like(self.client_tr_data)
        ts_marks = torch.zeros_like(self.client_ts_data)

        # if m_type == 0:
        #     g_mark = partial(generate_line_marks
        #                      , image_size=self.image_size
        #                      , rate=0.25)
        # if m_type == 1:
        #     g_mark = partial(generate_line_marks
        #                      , image_size=self.image_size)
        if m_type == 2-1:
            g_mark = partial(generate_sin_marks
                             , image_size=self.image_size
                             , A=1., phase=None, period=1.)
        elif m_type == 3-1:
            g_mark = partial(generate_sin_marks
                             , image_size=self.image_size
                             , A=1., phase=None, period=1., vertical=True)
        elif m_type == 4-1:
            g_mark = partial(generate_ellipse_marks
                             , image_size=self.image_size, )
        else:
            return tr_marks, ts_marks

        for idx, coords in map(g_mark, range(self.n_tr)):
            tr_marks[idx, :, coords[:, 0], coords[:, 1]] = 255
        for idx, coords in map(g_mark, range(self.n_ts)):
            ts_marks[idx, :, coords[:, 0], coords[:, 1]] = 255

        return tr_marks, ts_marks

    def get_fed_dataset(self, heter_x=False, heter_y=False):
        if heter_x:
            logger.info("generate marks")
            tr_marks, ts_marks = self.__generate_marks__()
            self.client_tr_data += tr_marks
            self.client_ts_data += ts_marks
            self.client_tr_data.clamp_max_(255)
            self.client_ts_data.clamp_max_(255)
            self.client_tr_data = self.client_tr_data.float() / 255.
            self.client_ts_data = self.client_ts_data.float() / 255.

        if heter_y:
            logger.info("offeset labels")
            label_offset = int(self.client_id) % (5-1)
            self.client_tr_labels = (self.client_tr_labels + label_offset) % self.n_classes
            self.client_ts_labels = (self.client_ts_labels + label_offset) % self.n_classes

        tr_set = FedDataset(self.client_id, self.dataset_name
                            , self.client_tr_data, self.client_tr_labels
                            , self.image_size, self.tr_mean, self.tr_std
                            , True)
        ts_set = FedDataset(self.client_id, self.dataset_name
                            , self.client_ts_data, self.client_ts_labels
                            , self.image_size, self.tr_mean, self.tr_std
                            , False)
        return tr_set, ts_set


class CelebAGenerator:
    def __init__(self, client_id, dataset_name, path_to_data
                 , client_tr_list, client_ts_list
                 , resize):
        """

        :param client_id:
        :param dataset_name: dataset name. enumerator DatasetName
        :param path_to_data: path to the directory of image data
        :param client_tr_list: a list of ts samples index: [sample index of the client]
        :param client_ts_list: a list of ts samples index: [sample index of the client]
        :param resize: resize the image
        """

        self.client_id = client_id
        logger.info("distribute data to client: {:d}".format(self.client_id))

        self.dataset_name = dataset_name
        logger.info("dataset: " + self.dataset_name.value)

        self.path_to_data = path_to_data
        logger.info("directory of image data: " + self.path_to_data)

        logger.info("loading data")
        self.client_tr_list = client_tr_list
        self.client_ts_list = client_ts_list

        self.client_tr_data = []
        self.client_ts_data = []

        self.resize = resize
        transformerPIL = transforms.Compose([transforms.Resize(self.resize)
                                            , transforms.CenterCrop(self.resize)
                                            , transforms.ToTensor()])

        for img_idx in self.client_tr_list:
            img = transformerPIL(Image.open(os.path.join(self.path_to_data, img_idx))).unsqueeze(0)
            self.client_tr_data.append(img)

        for img_idx in self.client_ts_list:
            img = transformerPIL(Image.open(os.path.join(self.path_to_data, img_idx))).unsqueeze(0)
            self.client_ts_data.append(img)

        self.client_tr_data = torch.cat(self.client_tr_data, dim=0)
        self.client_ts_data = torch.cat(self.client_ts_data, dim=0)

        self.n_tr = self.client_tr_data.shape[0]
        self.n_ts = self.client_ts_data.shape[0]
        logger.info("n_tr: {:d}".format(self.n_tr))
        logger.info("n_ts: {:d}".format(self.n_ts))

        tr_mean = self.client_tr_data.mean(dim=[0, 2, 3])
        tr_std = self.client_tr_data.std(dim=[0, 2, 3])
        self.tr_mean = tuple(tr_mean.tolist())
        self.tr_std = tuple(tr_std.tolist())
        self.image_size = self.client_tr_data[0].shape

        logger.info("tr_mean: " + str(self.tr_mean))
        logger.info("tr_std: " + str(self.tr_std))
        logger.info("image size: ({:d} * {:d} * {:d})".format(
            self.image_size[0], self.image_size[1], self.image_size[2]))

    def get_fed_dataset(self):
        tr_set = FedDataset(self.client_id, self.dataset_name
                            , self.client_tr_data, None
                            , self.image_size, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                            , True)
        ts_set = FedDataset(self.client_id, self.dataset_name
                            , self.client_ts_data, None
                            , self.image_size, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                            , False)
        return tr_set, ts_set



class ClientDatasetDomainNet:
    def __init__(self, client_id, dataset
                 , client_tr_data, client_tr_label
                 , client_ts_data, client_ts_label
                 , image_size):
        """

        :param client_id:
        :param dataset: name of the dataset
        :param client_tr_data: a tensor of tr samples index: [sample]
        :param client_tr_label: a tensor of tr samples index: [sample]
        :param client_ts_data: a tensor of ts samples index: [sample index of the client]
        :param client_ts_label: a tensor of tr samples index: [sample]
        :param image_size
        """

        self.client_id = client_id
        logging.info("distribute data to client: {:d}".format(self.client_id))

        self.dataset = dataset
        logging.info("dataset: " + self.dataset.value)

        self.client_tr_data = client_tr_data
        self.client_tr_label = torch.tensor(client_tr_label)
        self.client_ts_data = client_ts_data
        self.client_ts_label = torch.tensor(client_ts_label)

        self.image_size = image_size

        self.n_tr = len(self.client_tr_data)
        self.n_ts = len(self.client_ts_data)
        self.n_samples = self.n_tr + self.n_ts

    def get_fed_dataset(self, is_training):
        if is_training:
            return FedDatasetDomainNet(self.client_id, self.dataset
                                       , self.client_tr_data
                                       , self.client_tr_label
                                       , self.image_size, is_training)
        else:
            return FedDatasetDomainNet(self.client_id, self.dataset
                                       , self.client_ts_data
                                       , self.client_ts_label
                                       , self.image_size, is_training)


class FedDatasetDomainNet(Dataset):
    def __init__(self, c_id, dataset, tensor_data, tensor_label, img_size, is_training):
        """

        :param c_id:
        :param dataset: dataset name. enumerator DatasetName
        :param tensor_data: a tensor of samples [n_sample]
        :param list_label
        :param img_size:
        :param is_training: is training set
        """

        self.c_id = c_id
        self.dataset = dataset
        self.is_training = is_training

        self.data = tensor_data
        self.label = tensor_label

        self.img_size = img_size

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
        return img, label

    def __len__(self):
        return self.data.shape[0]
