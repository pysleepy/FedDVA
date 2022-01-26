import torch
import torch.nn as nn
from torch.functional import F

def reparameter(mu, log_var):
    # reparameterization trick refers to https://arxiv.org/abs/1312.6114
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


def loss_dkl(mu, log_var, mu_prior, log_var_prior):
    # Equation (24)
    var_p = log_var_prior.exp()
    var = log_var.exp()
    loss = (mu - mu_prior) ** 2 / var_p - (log_var - log_var_prior) + var / var_p - 1
    loss = 0.5 * torch.sum(loss, dim=1)
    return loss


def loss_reg_c(mu_c, log_var_c):
    # Equation (26)
    n_sample = mu_c.shape[0]
    d_sample = mu_c.shape[1]

    # Descartes operations through increasing tensor dimension rather 'for' loop
    mu_c_expand = mu_c.expand(n_sample, n_sample, d_sample)
    log_var_c_expand = log_var_c.expand(n_sample, n_sample, d_sample)
    var_c = log_var_c.exp()
    var_c_expand = var_c.expand(n_sample, n_sample, d_sample)

    term_1 = (mu_c_expand.permute(1, 0, 2) - mu_c_expand) ** 2 / var_c_expand

    term_2 = - (log_var_c_expand.permute(1, 0, 2) - log_var_c_expand)

    term_3 = var_c_expand.permute(1, 0, 2) / var_c_expand

    loss = term_1 + term_2 + term_3 - 1

    loss = torch.mean(0.5 * (torch.sum(loss, dim=2)), dim=1)

    return loss


class Backbone(nn.Module):
    def __init__(self, in_channel, hidden_dims):
        super(Backbone, self).__init__()

        self.hidden_dims = hidden_dims
        self.in_channel = in_channel
        self.layers = nn.ModuleList()

        cur_channel = self.in_channel
        for h_dim in self.hidden_dims:
            cur_layer = nn.Sequential(nn.Conv2d(cur_channel, out_channels=h_dim,
                                                kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(h_dim),
                                      nn.ReLU())
            cur_channel = h_dim
            self.layers.append(cur_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        return x


class VQEncoder(nn.Module):
    def __init__(self, d_in, d_out):
        super(VQEncoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.fc = nn.Linear(self.d_in, self.d_out)

    def forward(self, x):
        z = self.fc(x)
        return z


class GEncoder(nn.Module):
    def __init__(self, d_in, d_out):
        super(GEncoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.fc_mu_z = nn.Linear(self.d_in, self.d_out)
        self.fc_log_var_z = nn.Linear(self.d_in, self.d_out)

    def forward(self, x):
        mu_z = self.fc_mu_z(x)
        log_var_z = self.fc_log_var_z(x)
        return mu_z, log_var_z


class Decoder(nn.Module):
    def __init__(self, d_encoding, hidden_dims, out_channels):
        super(Decoder, self).__init__()
        self.d_encoding = d_encoding
        self.out_channels = out_channels
        self.hidden_dims = list(reversed(hidden_dims))

        self.fc_input = nn.Linear(self.d_encoding, self.hidden_dims[0])
        self.layers = nn.ModuleList()
        cur_channel = self.hidden_dims[0]
        for h_dim in self.hidden_dims[1:]:
            cur_layer = nn.Sequential(nn.ConvTranspose2d(cur_channel, h_dim
                                                         , kernel_size=3, stride=2
                                                         , padding=1, output_padding=1)
                                      , nn.BatchNorm2d(h_dim)
                                      , nn.ReLU())
            cur_channel = h_dim
            self.layers.append(cur_layer)
        last_layer = nn.Sequential(nn.ConvTranspose2d(cur_channel, self.out_channels
                                                      , kernel_size=3, stride=2, padding=1, output_padding=1)
                                   , nn.BatchNorm2d(self.out_channels)
                                   , nn.Tanh())
        self.layers.append(last_layer)

    def forward(self, encoding):
        x = F.relu(self.fc_input(encoding))
        for layer in self.layers:
            x = layer(x)
        return x


class Codebook(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self, n_embeddings, d_embedding, beta = 0.25):
        super(Codebook, self).__init__()
        self.K = n_embeddings
        self.D = d_embedding
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        # vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), commitment_loss, embedding_loss  # [B x D x H x W]


class DualVQEncoder(nn.Module):
    def __init__(self, in_channel, hidden_dims, d_z, d_c, n_code):
        super(DualVQEncoder, self).__init__()
        self.in_channel = in_channel
        self.hidden_dims = hidden_dims
        self.d_z = d_z
        self.d_c = d_c
        self.n_code = n_code

        self.backbone_z = Backbone(self.in_channel, self.hidden_dims)
        self.backbone_c = Backbone(self.in_channel + self.d_z, self.hidden_dims)
        self.encoder_z = VQEncoder(self.hidden_dims[-1], self.d_z)
        self.encoder_c = GEncoder(self.hidden_dims[-1], self.d_c)
        self.vq_encoder = Codebook(self.n_code, self.d_z)
        # self.decoder_z = Decoder(self.d_z, self.hidden_dims, self.in_channel)
        self.decoder = Decoder(self.d_z + self.d_c, self.hidden_dims, self.in_channel)

    def forward(self, x, z=None):
        x = x.view(-1, 1 * 28 * 28)
        if z is None:
            x_z = self.backbone_z(x)
            z = self.encoder_z(x_z)
            z.reshape(-1, x.size(1), x.size(2), x.size(3))
            z, commitment_loss, embedding_loss = self.vq_encoder(z)
            x_hat = self.decoder_z(z)
            return x_hat, z, commitment_loss, embedding_loss
        else:
            x_c = self.backbone_c(torch.stack([x, z], dim=1))
            mu_c, log_var_c = self.encoder_c(x_c)
            c = reparameter(mu_c, log_var_c)
            x_hat = self.decoder_c(torch.cat([z.detach(), c], dim=1))
            return x_hat, c, mu_c, log_var_c

    def generate(self, z, c):
        output_z = self.decoder_z(z).view(-1, 1, 28, 28)
        output_c = self.decoder_c(torch.cat([z, c], dim=1)).view(-1, 1, 28, 28)
        return output_z, output_c


