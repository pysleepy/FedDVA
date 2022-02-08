import math
import torch
import torch.nn as nn
from torch.functional import F


class LinearWithChannel(nn.Module):
    # reference https://github.com/pytorch/pytorch/issues/36591
    def __init__(self, input_size, output_size, n_channels):
        super(LinearWithChannel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_channels = n_channels

        # initialize weights
        # [C x I x O]
        self.w = torch.nn.Parameter(torch.zeros(self.n_channels, self.input_size, self.output_size))
        # [C x 1 x O]
        self.b = torch.nn.Parameter(torch.zeros(self.n_channels, 1, self.output_size))

        # change weights to kaiming
        self.reset_parameters(self.w, self.b)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):  # [C x N x I]
        return torch.bmm(x, self.w) + self.b  # [C x N x O]


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    [2] https://github.com/AntixK/PyTorch-VAE
    """
    def __init__(self, k_latents, latent_channel, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.K = k_latents
        self.D = latent_channel
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [N x C x H x W] -> [N x H x W x C]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [NHW x C]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [NHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [NHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [NHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [NHW, C]
        quantized_latents = quantized_latents.view(latents_shape)  # [N x H x W x C]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [N x C x H x W]


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input):
        return input + self.resblock(input)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels

        self.layers = nn.ModuleList()

        # building blocks
        cur_channel = self.in_channels
        for h_dim in hidden_dims:
            cur_layer = nn.Sequential(nn.Conv2d(cur_channel, out_channels=h_dim, kernel_size=4, stride=2, padding=1)
                                      , nn.LeakyReLU())
            self.layers.append(cur_layer)
            cur_channel = h_dim

        last_layer = nn.Sequential(nn.Conv2d(cur_channel, cur_channel, kernel_size=3, stride=1, padding=1)
                                   , nn.LeakyReLU())
        self.layers.append(last_layer)

        # residual blocks
        for _ in range(6):
            self.layers.append(ResidualLayer(cur_channel, cur_channel))
        self.layers.append(nn.LeakyReLU())

        last_layer = nn.Sequential(nn.Conv2d(cur_channel, self.out_channels, kernel_size=1, stride=1)
                                   , nn.LeakyReLU())
        self.layers.append(last_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = list(reversed(hidden_dims))
        self.out_channels = out_channels

        self.layers = nn.ModuleList()

        # residual blocks
        first_layer = nn.Sequential(nn.Conv2d(self.in_channels, hidden_dims[0], kernel_size=3, stride=1, padding=1)
                                    , nn.LeakyReLU())
        self.layers.append(first_layer)

        for _ in range(6):
            self.layers.append(ResidualLayer(hidden_dims[0], hidden_dims[0]))
        self.layers.append(nn.LeakyReLU())

        # building blocks
        for i in range(len(hidden_dims) - 1):
            cur_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1]
                                                         , kernel_size=4, stride=2, padding=1)
                                      , nn.LeakyReLU())
            self.layers.append(cur_layer)

        last_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1]
                                                      , out_channels=3, kernel_size=4, stride=2, padding=1)
                                   , nn.Tanh())
        self.layers.append(last_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DualEncoderWithVQ(nn.Module):
    def __init__(self, in_h, in_w, in_channel, hidden_dims
                 , z_w, z_h, z_channel
                 , c_w, c_h, c_channel
                 , k_hidden_z, k_hidden_c
                 , beta_z, beta_c):
        super(DualEncoderWithVQ, self).__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.in_channel = in_channel
        self.hidden_dims = hidden_dims

        self.z_w = z_w
        self.z_h = z_h
        self.z_channel = z_channel

        self.c_w = c_w
        self.c_h = c_h
        self.c_channel = c_channel

        self.k_hidden_z = k_hidden_z
        self.k_hidden_c = k_hidden_c

        self.beta_z = beta_z
        self.beta_c = beta_c

        self.encoder_z = Encoder(self.in_channel, self.hidden_dims, self.z_channel)
        self.vq_layer_z = VectorQuantizer(self.k_hidden_z, self.z_channel, self.beta_z)

        self.embedding_z = LinearWithChannel(self.z_w * self.z_w, self.in_h * self.in_w, self.z_channel)
        self.embedding_x = nn.Conv2d(self.in_channel, self.in_channel, 1)

        self.encoder_c = Encoder(self.in_channel + self.z_channel, self.hidden_dims, self.c_channel)
        self.vq_layer_c = VectorQuantizer(self.k_hidden_c, self.c_channel, self.beta_c)

        self.decoder = Decoder(self.z_channel+self.c_channel, hidden_dims, self.in_channel)

    def forward(self, x, on_c):
        x_z = self.encoder_z(x)
        z, vq_loss_z = self.vq_layer_z(x_z)
        if not on_c:
            x_c = torch.rand([x.shape[0], self.c_channel, self.c_h, self.c_w])  # [N x C x H x W]
            c, vq_loss_c = self.vq_layer_c(x_c)  # [N x C x H x W]
        else:
            z_c = z.detach().permute(1, 0, 2, 3).contiguous()  # [N x C x Z_H x Z_W] -> [C x N x Z_H x Z_W]
            z_c = z_c.view(z.shape[1], z.shape[0], -1)  # [C x N x Z_H x Z_W] -> [C x N x Z_HZ_W]
            # [C x N x Z_HZ_W] -> [C x N x IN_H x IN_W]
            e_z = self.embedding_z(z_c).view(z.shape[1], z.shape[0], self.in_h, self.in_w)
            e_z = e_z.permute(1, 0, 2, 3).contiguous()  # [C x N x IN_H x IN_W] -> [N x C x IN_H x IN_W]
            e_x = self.embedding_x(x)
            x_c = torch.cat([e_x, e_z], dim=1)
            x_c = self.encoder_c(x_c)  # [N x C x H x W]
            c, vq_loss_c = self.vq_layer_c(x_c)  # [N x C x H x W]

        x_hat = self.decoder(torch.cat([z, c], dim=1))
        return x_hat, z,  c, vq_loss_z, vq_loss_c

    def generate(self, z, c):
        z, vq_loss_z = self.vq_layer_c(z)
        c, vq_loss_c = self.vq_layer_c(c)
        output = self.decoder(torch.cat([z, c], dim=1))
        return output, z, c, vq_loss_z, vq_loss_c
