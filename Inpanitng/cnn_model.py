#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import math
import copy
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride):
        super(ConvBlock, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=pad_size, bias=False),
                                       # nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class DeConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(DeConvBlock, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel, stride=2, padding=pad_size, output_padding=1, bias=False),
                                       # nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ConvBlock_last(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock_last, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, padding=pad_size, bias=False))
                                       # nn.BatchNorm2d(out_size),
                                       # nn.Tanh())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class DeConvBlock_last(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(DeConvBlock_last, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel, padding=pad_size, bias=False))
                                       # nn.BatchNorm2d(out_size),
                                       # nn.Tanh())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ConvBlockTranspose(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlockTranspose, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel, padding=pad_size, bias=False),
                                       # nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ConvBlock_cat(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock_cat, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, padding=pad_size, bias=False),
                                       # nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ConvBlock_s(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride):
        super(ConvBlock_s, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=pad_size, bias=False),
                                       # nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ConvBlock_sum(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock_sum, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, padding=pad_size, bias=False),
                                       # nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = self.conv1(inputs1)
        in_data = [outputs, inputs2]
        # check of the channel size
        small_ch_id, large_ch_id = (0, 1) if in_data[0].size(1) < in_data[1].size(1) else (1, 0)
        offset = int(in_data[large_ch_id].size()[1] - in_data[small_ch_id].size()[1])
        if offset != 0:
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat([in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return self.relu(out)

class DeConvBlock_sum(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(DeConvBlock_sum, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel, stride=2, padding=pad_size, output_padding=1, bias=False),
                                       # nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs1 = self.conv1(inputs1)
        offset = outputs1.size()[2] - inputs2.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs2 = F.pad(inputs2, padding)
        out = torch.add(outputs1, outputs2)
        return self.relu(out)




class CGP2CNN_autoencoder(nn.Module):
    def __init__(self, cgp, in_channel, imgSize):
        super(CGP2CNN_autoencoder, self).__init__()
        self.cgp = cgp
        self.pool_size = 2
        self.arch = OrderedDict()
        self.encode = []
        self.decode = []
        self.channel_num = [None for _ in range(len(self.cgp))]
        self.size = [None for _ in range(len(self.cgp))]
        self.channel_num[0] = in_channel
        self.size[0] = imgSize
        # encoder
        i = 0
        for name, in1 in self.cgp:
            if name == 'input' or 'DeConv' in name:
                i += 1
                continue
            key = name.split('_')
            down =     key[0]
            func =     key[1]
            out_size = int(key[2])
            kernel   = int(key[3])
            if down == 'S':
                self.channel_num[i] = out_size
                self.size[i] = self.size[in1]
                if func == 'ConvBlock':
                    self.encode.append(ConvBlock(self.channel_num[in1], out_size, kernel, stride=1))
                else:
                    self.encode.append(ConvBlock_s(self.channel_num[in1], out_size, kernel, stride=1))
            else:
                self.channel_num[i] = out_size
                self.size[i] = self.size[in1]
                if func == 'ConvBlock':
                    self.encode.append(ConvBlock(self.channel_num[in1], out_size, kernel, stride=2))
                else:
                    self.encode.append(ConvBlock_s(self.channel_num[in1], out_size, kernel, stride=2))
            i += 1

        # decoder
        self.channel_num_d = [None for _ in range(len(self.cgp))]
        i -= 2 # skip the last layer
        self.channel_num_d[0] = self.channel_num[i]
        self.channel_num_d[1] = self.channel_num[i]
        i = 0
        self.cgp_inverse = copy.deepcopy(self.cgp)
        self.cgp_inverse.reverse()
        for j in range(len(self.cgp_inverse)):
            self.cgp_inverse[j][1] = int(math.fabs(self.cgp_inverse[j][1]-(len(self.cgp_inverse)-3)))
        for j in range(len(self.cgp_inverse)-1):
            if j == 0:
                i += 1
                continue
            name = self.cgp_inverse[j][0]
            key = name.split('_')
            down =     key[0]
            func =     key[1]
            out_size = int(key[2])
            kernel   = int(key[3])
            self.channel_num_d[i] = out_size
            if down == 'S':
                if func == 'ConvBlock':
                    self.decode.append(ConvBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, kernel, stride=1))
                else:
                    self.decode.append(ConvBlock_sum(self.channel_num_d[self.cgp_inverse[j][1]], out_size, kernel))
            else:
                if func == 'ConvBlock':
                    self.decode.append(DeConvBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, kernel))
                else:
                    self.decode.append(DeConvBlock_sum(self.channel_num_d[self.cgp_inverse[j][1]], out_size, kernel))
            i += 1
        # the last layer
        for j in range(1):
            name = self.cgp_inverse[j][0]
            key = name.split('_')
            down =     key[0]
            func =     key[1]
            out_size = int(key[2])
            kernel   = int(key[3])
            self.channel_num_d[-1] = out_size
            if down == 'S':
                if func == 'DeConvBlock':
                    self.decode.append(DeConvBlock_last(self.channel_num_d[i-1], out_size, kernel))
                else:
                    self.decode.append(ConvBlock_last(self.channel_num_d[i-1], out_size, kernel))
            else:
                if func == 'DeConvBlock':
                    self.decode.append(DeConvBlock_last(self.channel_num_d[i-1], out_size, kernel))
                else:
                    self.decode.append(ConvBlock_last(self.channel_num_d[i-1], out_size, kernel))

        self.network = self.encode + self.decode
        self.layer_module = nn.ModuleList(self.network)
        self.train = True
        self.loss = None
        self.accuracy = None
        self.outputs = [None for _ in range(len(self.cgp)-1)]
        self.outputs_d = [None for _ in range(len(self.cgp_inverse))]
        self.outputs_sum = [None for _ in range(len(self.cgp_inverse))]
        self.param_num = 0

    def main(self,x):
        out = x
        outputs = self.outputs
        outputs[0] = x    # input image
        outputs_d = self.outputs_d
        outputs_sum = self.outputs_sum
        nodeID = 1
        sumID = 0
        decodeID = 1
        flag = True
        for layer in self.layer_module:
            # encoder
            if nodeID <= len(self.encode):
                if isinstance(layer, ConvBlock_s):
                    outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
                    outputs_sum[sumID] = outputs[nodeID]
                    sumID += 1
                else:
                    outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            # decoder
            elif nodeID < (len(self.decode)+len(self.encode)):
                if flag:
                    outputs_d[0] = outputs[nodeID-1]
                    outputs_d[1] = outputs[nodeID-1]
                    del outputs
                    flag = False
                    if isinstance(layer, ConvBlock_sum):
                        sumID -= 1
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]], outputs_sum[sumID])
                    elif isinstance(layer, DeConvBlock_sum):
                        sumID -= 1
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]], outputs_sum[sumID])
                    else:
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                else:
                    if isinstance(layer, ConvBlock_sum):
                        sumID -= 1
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]], outputs_sum[sumID])
                    elif isinstance(layer, DeConvBlock_sum):
                        sumID -= 1
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]], outputs_sum[sumID])
                    else:
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                decodeID += 1
            nodeID += 1
        # the last layer
        layer = self.layer_module[-1]
        out = layer(outputs_d[decodeID-1])
        del outputs_d
        return out

    def forward(self, x):
        return self.main(x)
