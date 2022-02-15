
import math

import torch
import torch.nn as nn


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
        """
        C: n_channel, N: n_samples, I: n_features, O: n_outputs
        :param x: [C x N x I]
        :return: [C x N x O]
        """
        return torch.bmm(x, self.w) + self.b  # [C x N x O]
