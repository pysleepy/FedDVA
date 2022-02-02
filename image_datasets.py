import random

import os
import numpy as np
from enum import Enum, unique
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import logging
import functools as f


@unique
class ImageDatasetName(Enum):
    MNIST = "MNIST"
    FashionMNIST = "FashionMNIST"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    DomainNet = "DomainNet"
    CelebA = "CelebA"


class ClientDataset:
    def __init__(self, client_id, dataset
                 , client_tr_idx, client_ts_idx
                 , client_tr_data, client_tr_labels, client_ts_data, client_ts_labels):
        """

        :param client_id:
        :param dataset: dataset name. enumerator DatasetName
        :param client_tr_idx: a list of tr samples index: [[sample index of label 0],[],...,[]]
        :param client_ts_idx: a list of ts samples index: [[sample index of label 0],[],...,[]]
        :param client_tr_data: a tensor of tr samples [n_sample * h * w * channel]
        :param client_tr_labels: a tensor of tr labels [n_sample]
        :param client_ts_data: a tensor of ts samples [n_sample * h * w * channel]
        :param client_ts_labels: a tensor of ts labels [n_sample]
        """

        self.client_id = client_id
        logging.info("distribute data to client: {:d}".format(self.client_id))

        self.dataset = dataset
        logging.info("dataset: " + self.dataset.value)

        self.client_tr_idx = client_tr_idx
        self.client_ts_idx = client_ts_idx

        self.client_tr_data = client_tr_data
        self.client_tr_labels = client_tr_labels
        self.client_ts_data = client_ts_data
        self.client_ts_labels = client_ts_labels

        self.n_samples = [(len(sample_tr), len(sample_ts)) for sample_tr, sample_ts in zip(self.client_tr_idx, self.client_ts_idx)]
        self.n_tr = [x[0] for x in self.n_samples]
        self.n_ts = [x[1] for x in self.n_samples]
        self.n_class = len(self.n_samples)
        logging.info("n_tr: {:d}, n_ts: {:d}, n_class:{:d}".format(sum(self.n_tr), sum(self.n_ts), self.n_class))

        tr_mean = (client_tr_data.float() / 255).mean(dim=list(range(client_tr_data.dim()-1)))
        tr_std = (client_tr_data.float() / 255).std(dim=list(range(client_tr_data.dim()-1)))
        self.tr_mean = tuple(tr_mean.tolist())
        self.tr_std = tuple(tr_std.tolist())
        self.image_size = self.client_tr_data[0].shape

        logging.info("tr_mean: " + str(self.tr_mean))
        logging.info("tr_std: " + str(self.tr_std))
        logging.info("image size: ({:d} * {:d} * {:d})".format(
            self.image_size[0], self.image_size[1], self.image_size[2]))

        self.m_type = client_id % 5
        self.tr_marks, self.ts_marks = self.generate_marks()
        self.tr_labels_shifted = (self.client_tr_labels + self.m_type) % self.n_class
        self.ts_labels_shifted = (self.client_ts_labels + self.m_type) % self.n_class

    def generate_marks(self):
        tr_marks = torch.zeros_like(self.client_tr_data)
        ts_marks = torch.zeros_like(self.client_ts_data)
        if self.m_type == 1:
            tr_cords = [generate_triangle_marks(idx, self.image_size) for idx in range(sum(self.n_tr))]
            ts_cords = [generate_triangle_marks(idx, self.image_size) for idx in range(sum(self.n_ts))]
        elif self.m_type == 2:
            tr_cords = [generate_sin_marks(idx, self.image_size, bias=np.random.rand())
                        for idx in range(sum(self.n_tr))]
            ts_cords = [generate_sin_marks(idx, self.image_size, bias=np.random.rand())
                        for idx in range(sum(self.n_ts))]
        elif self.m_type == 3:
            tr_cords = [generate_sin_marks(idx, self.image_size, bias=np.random.rand(), vertical=True)
                        for idx in range(sum(self.n_tr))]
            ts_cords = [generate_sin_marks(idx, self.image_size, bias=np.random.rand(), vertical=True)
                        for idx in range(sum(self.n_ts))]
        elif self.m_type == 4:
            tr_cords = [generate_ellipse_marks(idx, self.image_size, rot_angle=np.random.rand())
                        for idx in range(sum(self.n_tr))]
            ts_cords = [generate_ellipse_marks(idx, self.image_size, rot_angle=np.random.rand())
                        for idx in range(sum(self.n_ts))]
        else:
            return tr_marks, ts_marks

        for cor in tr_cords:
            for (idx, h, w) in cor:
                tr_marks[idx, h, w, :] = 255
        for cor in ts_cords:
            for (idx, h, w) in cor:
                ts_marks[idx, h, w, :] = 255
        return tr_marks, ts_marks

    def get_class_dist(self):
        return self.client_id, self.n_tr

    def get_fed_dataset(self, is_training, require_augmentation, type_heter):
        if is_training:
            return FedDataset(self.client_id, self.dataset
                              , self.client_tr_data, self.tr_marks
                              , self.client_tr_labels, self.tr_labels_shifted
                              , self.image_size, self.tr_mean, self.tr_std
                              , is_training, require_augmentation, type_heter)
        else:
            return FedDataset(self.client_id, self.dataset
                              , self.client_ts_data, self.ts_marks
                              , self.client_ts_labels, self.ts_labels_shifted
                              , self.image_size, self.tr_mean, self.tr_std
                              , is_training, require_augmentation, type_heter)


class FedDataset(Dataset):
    def __init__(self, c_id, dataset
                 , tensor_data, tensor_marks
                 , tensor_labels, tensor_labels_shifted
                 , img_size, data_mean, data_std
                 , is_training, require_augmentation, type_heter):
        """

        :param c_id:
        :param dataset: dataset name. enumerator DatasetName
        :param tensor_data: a tensor of samples [n_sample * h * w * channel]
        :param tensor_labels: a tensor of labels [n_sample]
        :param img_size: (h * w *c)
        :param data_mean: (mean_c1, mean_c2, ...)
        :param data_std: (std_c1, std_c2, ...)
        :param is_training: is training set
        :param require_agument:
        :param type_heter: is x or y need be heterogeneous ('x', 'y' or others)
        """

        self.c_id = c_id
        self.dataset = dataset
        self.type_heter = type_heter
        self.is_training = is_training
        self.require_augmentation = require_augmentation
        # self.require_augmentation = True if dataset.value.startswith("CIFAR") else False

        if type_heter == 'x':
            self.data = torch.max(tensor_data, tensor_marks).numpy()
        else:
            self.data = tensor_data.numpy()

        if type_heter == 'y':
            self.labels = tensor_labels_shifted
        else:
            self.labels = tensor_labels

        self.img_size = img_size
        self.data_mean = data_mean
        self.data_std = data_std

        self.transformer = self.get_transformer()

    def get_transformer(self):
        trans = [transforms.ToPILImage()]
        if self.require_augmentation and self.is_training:
            trans.extend([transforms.RandomCrop(self.img_size[0], padding=4), transforms.RandomHorizontalFlip(0.1)])
        trans.extend([transforms.ToTensor(), transforms.Normalize(mean=self.data_mean, std=self.data_std)])
        return transforms.Compose(trans)

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.data[index]
        img = self.transformer(img)
        return img, label

    def __len__(self):
        return self.data.shape[0]


def generate_triangle_marks(idx, image_size, padding=1):
    n_col, n_row = image_size[0], image_size[1]

    x_max, y_max = n_col - 1 - padding, n_row - 1 - padding
    x_min, y_min = 0 + padding, 0 + padding

    x_center = np.random.randint(x_min+x_max/4, x_max-x_max/4)
    y_center = np.random.randint(y_min+y_max/4, y_max-y_max/4)
    r = min(x_center - x_min, x_max - x_center, y_center - y_min, y_max - y_center)

    degree = np.random.rand()
    degree_a = np.pi * degree
    degree_b = np.pi * (degree + 2./3.)
    degree_c = np.pi * (degree + 4./3.)

    a = (r * np.cos(degree_a)+x_center, r * np.sin(degree_a)+y_center)
    b = (r * np.cos(degree_b)+x_center, r * np.sin(degree_b)+y_center)
    c = (r * np.cos(degree_c)+x_center, r * np.sin(degree_c)+y_center)
    a, c, b = sorted([a, b, c], key=lambda x: x[0])

    rate_a_b = (b[1] - a[1]) / (b[0] - a[0])
    rate_a_c = (c[1] - a[1]) / (c[0] - a[0])
    rate_c_b = (b[1] - c[1]) / (b[0] - c[0])

    line_a_b = [(x, a[1]+(x-a[0])*rate_a_b) for x in np.arange(a[0], b[0], 0.2)]
    line_a_c = [(x, a[1]+(x-a[0])*rate_a_c) for x in np.arange(a[0], c[0], 0.2)]
    line_c_b = [(x, c[1]+(x-c[0])*rate_c_b) for x in np.arange(c[0], b[0], 0.2)]

    cor = line_a_b + line_a_c + line_c_b
    cor = [(idx, int(c[0]), int(c[1])) for c in cor]
    return cor


def generate_sin_marks(idx, image_size, padding=1, period=1, bias=1/3, vertical=False):
    """
    draw sin marks
    :param image_size
    :param padding: the padding of x and y
    :param period: how many periods of the sin
    :param bias: bias of the Asin(wx+bias*2*pi)
    :param vertical:
    :return: image of the marks
    """

    n_col, n_row = image_size[0], image_size[1]
    x_max, y_max = n_col - 1 - padding, n_row - 1 - padding
    x_span = x_max - padding
    y_span = y_max - padding

    x_period = np.arange(0, 2*np.pi*period, 0.2) + 2 * np.pi * bias
    x_len = len(x_period)

    x = np.arange(padding, x_max, x_span/x_len)
    y = np.array([y_span / 2 + y_span / 2 * np.sin(z) for z in x_period]) + padding

    if not vertical:
        cor = [(idx, int(b), int(a)) for a, b in zip(x, y)]
    else:
        cor = [(idx, int(a), int(b)) for a, b in zip(x, y)]

    return cor


def generate_ellipse_marks(idx, image_size, padding=1, e=0.9, rot_angle=0.25):
    """
    draw ellipse marks
    :param
    :param padding: the padding of x and y
    :param e: eccentricity
    :param rot_angle: rotation angle
    :return: image with marks, image of the marks
    """

    n_col, n_row = image_size[0], image_size[1]
    x_max, y_max = n_col - 1 - padding, n_row - 1 - padding

    x_span = x_max - padding
    y_span = y_max - padding

    rot_angle = np.pi * rot_angle
    x_center = padding + int(x_span / 2)
    y_center = padding + int(y_span / 2)

    a = x_span / 2
    c = a * e
    b = np.sqrt(a ** 2 - c ** 2)

    # draw the ellipse
    t = np.arange(0, 2 * np.pi, 0.2)
    x = np.cos(t) * a
    y = np.sin(t) * b

    # rotate
    xx = np.cos(rot_angle) * x - np.sin(rot_angle) * y
    yy = np.sin(rot_angle) * x + np.cos(rot_angle) * y

    # move the center
    xx += x_center
    yy += y_center

    cor = [(idx, int(b), int(a)) for a, b in zip(xx, yy)]

    return cor


class ClientDatasetCelebA:
    def __init__(self, client_id, dataset, root_dataset
                 , client_tr_idx, client_ts_idx, image_size):
        """

        :param client_id:
        :param dataset: dataset name. enumerator DatasetName
        :param root_dataset: root of the dataset
        :param client_tr_idx: a list of tr samples index: [sample index of the client]
        :param client_ts_idx: a list of ts samples index: [sample index of the client]
        # :param client_tr_data: a tensor of tr samples [n_sample * h * w * channel]
        # :param client_ts_data: a tensor of ts samples [n_sample * h * w * channel]
        """

        self.client_id = client_id
        logging.info("distribute data to client: {:d}".format(self.client_id))

        self.dataset = dataset
        logging.info("dataset: " + self.dataset.value)

        self.root_dataset = root_dataset
        logging.info("dataset: " + self.root_dataset)

        self.client_tr_idx = client_tr_idx
        self.client_ts_idx = client_ts_idx

        self.image_size = image_size

        self.n_tr = len(self.client_tr_idx)
        self.n_ts = len(self.client_ts_idx)
        self.n_samples = self.n_tr + self.n_ts

        self.transformer = transforms.Compose([transforms.Resize(self.image_size)
                                              , transforms.CenterCrop(self.image_size)
                                              , transforms.ToTensor()
                                              , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.client_tr_data = []
        self.client_ts_data = []

        for img_idx in self.client_tr_idx:
            img = self.transformer(Image.open(os.path.join(root_dataset, img_idx))).unsqueeze(0)
            self.client_tr_data.append(img)

        for img_idx in self.client_ts_idx:
            img = self.transformer(Image.open(os.path.join(root_dataset, img_idx))).unsqueeze(0)
            self.client_ts_data.append(img)

        self.client_tr_data= torch.cat(self.client_tr_data, dim=0)
        self.client_ts_data = torch.cat(self.client_ts_data, dim=0)

    def get_fed_dataset(self, is_training):
        if is_training:
            return FedDatasetCelebA(self.client_id, self.dataset
                              , self.client_tr_data, self.image_size, is_training)
        else:
            return FedDatasetCelebA(self.client_id, self.dataset
                              , self.client_ts_data, self.image_size, is_training)


class FedDatasetCelebA(Dataset):
    def __init__(self, c_id, dataset, tensor_data, img_size, is_training):
        """

        :param c_id:
        :param dataset: dataset name. enumerator DatasetName
        :param tensor_data: a tensor of samples [n_sample * h * w * channel]
        :param img_size: (h * w *c)
        :param is_training: is training set
        """

        self.c_id = c_id
        self.dataset = dataset
        self.is_training = is_training

        self.data = tensor_data
        self.img_size = img_size

    def __getitem__(self, index):
        img = self.data[index]
        return img

    def __len__(self):
        return self.data.shape[0]


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
