import io
import os
import zipfile
import logging

import numpy as np

from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

from image_datasets import ImageDatasetName, ClientDataset
from display_utilities import rand_cmap, plot_client_sample_rate

logging.getLogger().setLevel(logging.INFO)


def preprocess_domain_net():
    with zipfile.ZipFile('data/Images/DimainNet/clipart.zip') as thezip:
        with thezip.open('clipart/aircraft_carrier/clipart_001_000018.jpg', mode='r') as thefile:
            x = thefile.read()
    imageStream = io.BytesIO(x)
    imageFile = Image.open(imageStream)
    print(imageFile.size)
