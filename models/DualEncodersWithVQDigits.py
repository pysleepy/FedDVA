import math
import torch
import torch.nn as nn
from torch.functional import F


class LinearWithChannel(nn.Module):
    # reference https://github.com/pytorch/pytorch/issues/36591
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.channel_size = channel_size

        # initialize weights
        # [C x I x O]
        self.w = torch.nn.Parameter(torch.zeros(self.channel_size, self.input_size, self.output_size))
        # [C x 1 x O]
        self.b = torch.nn.Parameter(torch.zeros(self.channel_size, 1, self.output_size))

        # change weights to kaiming
        self.reset_parameters(self.w, self.b)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):  # [C x N x I]
        return torch.bmm(x, self.w) + self.b  # [D x N x O]


class Backbone(nn.Module):
    def __init__(self, in_channel, hidden_dims):
        super(Backbone, self).__init__()

        self.hidden_dims = hidden_dims
        self.in_channel = in_channel
        self.layers = nn.ModuleList()

        # for images of 28 x 28 padding=3. for images of 32 x 32 padding = 1.
        first_layer = nn.Sequential(nn.Conv2d(in_channels=self.in_channel, out_channels=self.hidden_dims[0]
                                              , kernel_size=4, stride=1, padding=3)
                                    , nn.BatchNorm2d(self.hidden_dims[-1])
                                    , nn.LeakyReLU())
        cur_channel = self.in_channel
        self.layers.append(first_layer)
        for h_dim in self.hidden_dims[1:-1]:
            cur_layer = nn.Sequential(nn.Conv2d(in_channels=cur_channel, out_channels=h_dim
                                                , kernel_size=4, stride=2, padding=1)
                                      , nn.BatchNorm2d(h_dim)
                                      , nn.LeakyReLU())
            cur_channel = h_dim
            self.layers.append(cur_layer)
        last_layer = nn.Sequential(nn.Conv2d(in_channels=cur_channel, out_channels=self.hidden_dims[-1]
                                             , kernel_size=4, stride=1, padding=0)
                                   , nn.BatchNorm2d(self.hidden_dims[-1])
                                   , nn.LeakyReLU())
        self.layers.append(last_layer)

    def forward(self, x):
        for layer in self.layers:  # [N x C x H x W] -> [N x hidden_dims[-1] x H' x W']
            x = layer(x)
        return x


class VQEncoder(nn.Module):
    def __init__(self, in_features, out_channels, h_latent, w_latent):
        super(VQEncoder, self).__init__()
        self.in_features = in_features
        self.out_channels = out_channels
        self.h_latent = h_latent
        self.w_latent = w_latent
        self.d_latent = self.h_latent * self.w_latent

        self.fc_1 = LinearWithChannel(self.in_features, int(self.in_features / 2), self.out_channels)
        self.fc_2 = LinearWithChannel(int(self.in_features / 2), self.d_latent, self.out_channels)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=0).repeat(self.out_channels, 1, 1)  # [N x I -> C x N x I]
        x = F.relu(self.fc_1(x))  # [C x N x I] -> [C x N x I/2]
        x = F.relu(self.fc_2(x))  # [C x N x I/2] -> [C x N x D]
        x = x.permute(1, 0, 2).contiguous()  # [N x C x D]
        return x


class Codebook(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    [2] https://github.com/AntixK/PyTorch-VAE/
    """
    def __init__(self, k_embeddings, d_embeddings, beta=0.25):
        super(Codebook, self).__init__()
        self.K = k_embeddings
        self.D = d_embeddings
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [NC x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [NC x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [NC, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [NC x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [NC x D]
        quantized_latents = quantized_latents.view(latents_shape)  # [N x C x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        # vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents, commitment_loss, embedding_loss  # [N x C x D]


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = list(reversed(hidden_dims))
        self.out_channels = out_channels

        self.layers = nn.ModuleList()
        cur_channel = self.hidden_dims[0]
        first_layer = nn.Sequential(nn.ConvTranspose2d(self.in_channels, cur_channel
                                                       , kernel_size=4, stride=1, padding=0)
                                    , nn.BatchNorm2d(cur_channel)
                                    , nn.LeakyReLU())
        self.layers.append(first_layer)
        for h_dim in self.hidden_dims[1:]:
            cur_layer = nn.Sequential(nn.ConvTranspose2d(cur_channel, h_dim
                                                         , kernel_size=4, stride=2, padding=1)
                                      , nn.BatchNorm2d(h_dim)
                                      , nn.LeakyReLU())
            cur_channel = h_dim
            self.layers.append(cur_layer)
        last_layer = nn.Sequential(nn.ConvTranspose2d(cur_channel, self.out_channels
                                                      , kernel_size=4, stride=2, padding=1)
                                   , nn.Tanh())
        self.layers.append(last_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DualVQEncoder(nn.Module):
    def __init__(self, in_channel, hidden_dims, d_z, d_c, n_code_z, n_code_c):
        super(DualVQEncoder, self).__init__()
        self.in_channel = in_channel
        self.hidden_dims = hidden_dims
        self.d_z = d_z
        self.d_c = d_c
        self.n_code_z = n_code_z
        self.n_code_c = n_code_c

        self.backbone_z = Backbone(self.in_channel, self.hidden_dims)
        self.encoder_z = VQEncoder(self.hidden_dims[-1], self.d_z)
        self.codebook_z = Codebook(self.n_code_z, self.d_z)

        self.backbone_c = Backbone(self.in_channel + self.d_z, self.hidden_dims)
        self.encoder_c = VQEncoder(self.hidden_dims[-1], self.d_c)
        self.codebook_c = Codebook(self.n_code_c, self.d_c)

        self.decoder = Decoder(self.d_z + self.d_c, self.hidden_dims, self.in_channel)

    def forward(self, x, conditioned_on_c):
        x_z = self.backbone_z(x)

        x = torch.flatten(x, start_dim=1)  # N x (hidden_dims[-1] x H' x W')

        z = self.encoder_z(x_z)
        z = z.reshape(-1, self.d_z, x.size(2), x.size(3))
        z, commitment_loss, embedding_loss = self.codebook_z(z)
        if not conditioned_on_c:
            random_c = torch.zeros([x.shape(0), self.d_z, x.shape(2), x.shape(3)], dtype=torch.float)
            c = reparameter(random_c, random_c)
        else:
            x_c = self.backbone_c(torch.stack([x, z], dim=1))
            c = self.encoder_c(x_c)
            c = c.reshape(-1, self.d_c, x.shape(2), x.shape(3))
            c, commitment_loss, embedding_loss = self.codebook_c(c)

        x_hat = self.decoder(torch.stack([z, c], dim=1))
        return x_hat, z, c, commitment_loss, embedding_loss

    def generate(self, z, c):
        output = self.decoder(torch.cat([z, c], dim=1))
        return output
