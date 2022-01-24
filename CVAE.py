import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
% matplotlib
inline

DEVICE = 'cuda'
SEED = 0
CLASS_SIZE = 10
BATCH_SIZE = 256
ZDIM = 16
NUM_EPOCHS = 50

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class CVAE(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self._zdim = zdim
        self._in_units = 28 * 28
        hidden_units = 512
        self._encoder = nn.Sequential(
            nn.Linear(self._in_units + CLASS_SIZE, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=True),
        )
        self._to_mean = nn.Linear(hidden_units, zdim)
        self._to_lnvar = nn.Linear(hidden_units, zdim)
        self._decoder = nn.Sequential(
            nn.Linear(zdim + CLASS_SIZE, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, self._in_units),
            nn.Sigmoid()
        )

    def encode(self, x, labels):
        in_ = torch.empty((x.shape[0], self._in_units + CLASS_SIZE), device=DEVICE)
        in_[:, :self._in_units] = x
        in_[:, self._in_units:] = labels
        h = self._encoder(in_)
        mean = self._to_mean(h)
        lnvar = self._to_lnvar(h)
        return mean, lnvar

    def decode(self, z, labels):
        in_ = torch.empty((z.shape[0], self._zdim + CLASS_SIZE), device=DEVICE)
        in_[:, :self._zdim] = z
        in_[:, self._zdim:] = labels
        return self._decoder(in_)


def to_onehot(label):
    return torch.eye(CLASS_SIZE, device=DEVICE, dtype=torch.float32)[label]


# Train
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

model = CVAE(ZDIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for e in range(NUM_EPOCHS):
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        labels = to_onehot(labels)
        # Reconstruction images
        # Encode images
        x = images.view(-1, 28 * 28 * 1).to(DEVICE)
        mean, lnvar = model.encode(x, labels)
        std = lnvar.exp().sqrt()
        epsilon = torch.randn(ZDIM, device=DEVICE)

        # Decode latent variables
        z = mean + std * epsilon
        y = model.decode(z, labels)

        # Compute loss
        kld = 0.5 * (1 + lnvar - mean.pow(2) - lnvar.exp()).sum(axis=1)
        bce = F.binary_cross_entropy(y, x, reduction='none').sum(axis=1)
        loss = (-1 * kld + bce).mean()

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.shape[0]

    print(f'epoch: {e + 1} epoch_loss: {train_loss / len(train_dataset)}')