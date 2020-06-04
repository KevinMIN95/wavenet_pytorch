# -*- coding: utf-8 -*-

import logging
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn


def initialize(m):
    """INITILIZE CONV WITH XAVIER.

    Arg:
        m (torch.nn.Module): Pytorch nn module instance.

    """
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class OneHot(nn.Module):
    """CONVERT TO ONE-HOT VECTOR.

    Args:
        depth (int): Dimension of one-hot vector

    """

    def __init__(self, depth):
        super(OneHot, self).__init__()
        self.depth = depth

    def forward(self, x):
        """FORWARD CALCULATION.

        Arg:
            x (LongTensor): Long tensor variable with the shape (B, T).

        Returns:
            Tensor: Float tensor variable with the shape (B, T, depth).

        """
        x = x % self.depth
        x = torch.unsqueeze(x, 2)
        x_onehot = x.new_zeros(x.size(0), x.size(1), self.depth).float()

        return x_onehot.scatter_(2, x, 1)


class CausalConv1d(nn.Module):
    """1D DILATED CAUSAL CONVOLUTION."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        """FORWARD CALCULATION.

        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).

        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).

        """
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x


class UpSampling(nn.Module):
    """UPSAMPLING LAYER WITH DECONVOLUTION.

    Args:
        upsampling_factor (int): Upsampling factor.

    """

    def __init__(self, upsampling_factor, bias=True):
        super(UpSampling, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.bias = bias
        self.conv = nn.ConvTranspose2d(1, 1,
                                       kernel_size=(1, self.upsampling_factor),
                                       stride=(1, self.upsampling_factor),
                                       bias=self.bias)

    def forward(self, x):
        """FORWARD CALCULATION.

        Args:
            x (Tensor): Float tensor variable with the shape (B, C, T).

        Returns:
            Tensor: Float tensor variable with the shape (B, C, T'),
                where T' = T * upsampling_factor.

        """
        x = x.unsqueeze(1)  # B x 1 x C x T
        x = self.conv(x)  # B x 1 x C x T'
        return x.squeeze(1)


class WaveNet(nn.Module):
    """CONDITIONAL WAVENET.

    Args:
        n_quantize (int): Number of quauntization
        n_aux (int): Number of aux feature dimension.
        n_resch (int): Number of filter channels for residual block.
        n_skipch (int): Number of filter channels for skip connection.
        dilation_depth (int): Number of dilation depth (e.g. if set 10, max dilation = 2^(10-1)).
        dilation_repeat (int): Number of dilation repeat.
        kernel_size (int): Filter size of dilated causal convolution.
        upsampling_factor (int): Upsampling factor.

    """

    def __init__(self,n_quantize=256, n_aux=28, n_resch=512, n_skipch=256,
                 dilation_depth=10, dilation_repeat=3, kernel_size=2, upsampling_factor=0):
        super(WaveNet, self).__init__()
        self.n_quantize = n_quantize
        self.n_aux = n_aux
        self.n_resch = n_resch
        self.n_skipch = n_skipch

        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.dilation_repeat = dilation_repeat
        self.upsampling_factor = upsampling_factor

        self.dilations = [2 ** i for i in range(self.dilation_depth)] * self.dilation_repeat
        self.receptive_field = (self.kernel_size - 1) * sum(self.dilations) + 1

        # for preprocessing
        self.onehot = OneHot(self.n_quantize)
        self.causal = CausalConv1d(self.n_quantize, self.n_resch, self.kernel_size)
        if self.upsampling_factor > 0:
            self.upsampling = UpSampling(self.upsampling_factor)

        # for residual blocks
        self.dil_sigmoid = nn.ModuleList()
        self.dil_tanh = nn.ModuleList()
        self.aux_1x1_sigmoid = nn.ModuleList()
        self.aux_1x1_tanh = nn.ModuleList()
        self.skip_1x1 = nn.ModuleList()
        self.res_1x1 = nn.ModuleList()
        for d in self.dilations:
            self.dil_sigmoid += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.dil_tanh += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.aux_1x1_sigmoid += [nn.Conv1d(self.n_aux, self.n_resch, 1)]
            self.aux_1x1_tanh += [nn.Conv1d(self.n_aux, self.n_resch, 1)]
            self.skip_1x1 += [nn.Conv1d(self.n_resch, self.n_skipch, 1)]
            self.res_1x1 += [nn.Conv1d(self.n_resch, self.n_resch, 1)]

        # for postprocessing
        self.conv_post_1 = nn.Conv1d(self.n_skipch, self.n_skipch, 1)
        self.conv_post_2 = nn.Conv1d(self.n_skipch, self.n_quantize, 1)

    def forward(self, x, h):
        """FORWARD CALCULATION.

        Args:
            x (Tensor): Long tensor variable with the shape (B, T).
            h (Tensor): Float tensor variable with the shape (B, n_aux, T),

        Returns:
            Tensor: Float tensor variable with the shape (B, T, n_quantize).

        """
        # preprocess
        output = self._preprocess(x)
        if self.upsampling_factor > 0:
            h = self.upsampling(h)

        # residual block
        skip_connections = []
        for l in range(len(self.dilations)):
            output, skip = self._residual_forward(
                output, h, self.dil_sigmoid[l], self.dil_tanh[l],
                self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                self.skip_1x1[l], self.res_1x1[l])
            skip_connections.append(skip)

        # skip-connection part
        output = sum(skip_connections)
        output = self._postprocess(output)

        return output

    def generate(self, x, h, n_samples, intervals=None, mode="sampling"):
        """GENERATE WAVEFORM WITH NAIVE CALCULATION.

        Args:
            x (Tensor): Long tensor variable with the shape (1, T).
            h (Tensor): Float tensor variable with the shape (1, n_aux, n_samples + T).
            n_samples (int): Number of samples to be generated.
            intervals (int): Log interval.
            mode (str): "sampling" or "argmax".

        Returns:
            ndarray: Generated quantized wavenform (n_samples,).

        """
        # upsampling
        if self.upsampling_factor > 0:
            h = self.upsampling(h)

        #padding if the length less than receptive field size
        n_pad = self.receptive_field - x.size(1)
        if n_pad > 0:
            x = F.pad(x, (n_pad, 0), "constant", self.n_quantize // 2)
            h = F.pad(h, (n_pad, 0), "replicate")

        # generate
        samples = x[0].tolist()
        start = time.time()
        for i in range(n_samples):
            current_idx = len(samples)
            x = torch.tensor(samples[-self.receptive_field:]).long().view(1, -1)
            h_ = h[:, :, current_idx - self.receptive_field: current_idx]

            # calculate output
            output = self._preprocess(x)
            skip_connections = []
            for l in range(len(self.dilations)):
                output, skip = self._residual_forward(
                    output, h_, self.dil_sigmoid[l], self.dil_tanh[l],
                    self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                    self.skip_1x1[l], self.res_1x1[l])
                skip_connections.append(skip)
            output = sum(skip_connections)
            output = self._postprocess(output)[0]  # T x n_quantize

            # get waveform
            if mode == "sampling":
                # get last sample
                posterior = F.softmax(output[-1], dim=0)
                dist = torch.distributions.Categorical(posterior)
                sample = dist.sample()
            elif mode == "argmax":
                sample = output[-1].argmax()
            else:
                logging.error("mode should be sampling or argmax")
                sys.exit(1)
            samples.append(sample)

            # show progress
            if intervals is not None and (i + 1) % intervals == 0:
                logging.info("%d/%d estimated time = %.3f sec (%.3f sec / sample)" % (
                    i + 1, n_samples,
                    (n_samples - i - 1) * ((time.time() - start) / intervals),
                    (time.time() - start) / intervals))
                start = time.time()

        return np.array(samples[-n_samples:])

    def _preprocess(self, x):
        x = self.onehot(x).transpose(1, 2)

        output = self.causal(x)

        return output

    def _postprocess(self, x):
        output = F.relu(x)
        output = self.conv_post_1(output)
        output = F.relu(output)  # B x C x T
        output = self.conv_post_2(output).transpose(1, 2)  # B x T x C
        return output

    def _residual_forward(self, x, h, dil_sigmoid, dil_tanh,
                          aux_1x1_sigmoid, aux_1x1_tanh, skip_1x1, res_1x1):
        output_sigmoid = dil_sigmoid(x)
        output_tanh = dil_tanh(x)
        aux_output_sigmoid = aux_1x1_sigmoid(h)
        aux_output_tanh = aux_1x1_tanh(h)
        output = torch.sigmoid(output_sigmoid + aux_output_sigmoid) * \
            torch.tanh(output_tanh + aux_output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + x
        return output, skip

    def _generate_residual_forward(self, x, h, dil_sigmoid, dil_tanh,
                                   aux_1x1_sigmoid, aux_1x1_tanh, skip_1x1, res_1x1):
        output_sigmoid = dil_sigmoid(x)[:, :, -1:]
        output_tanh = dil_tanh(x)[:, :, -1:]
        aux_output_sigmoid = aux_1x1_sigmoid(h)
        aux_output_tanh = aux_1x1_tanh(h)
        output = torch.sigmoid(output_sigmoid + aux_output_sigmoid) * \
            torch.tanh(output_tanh + aux_output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + x[:, :, -1:]  # B x C x 1
        return output, skip
        

