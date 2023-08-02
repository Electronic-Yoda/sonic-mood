import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch.optim as optim
import numpy as np
import time

class AudioCNN(nn.Module):
    def __init__(self, kernel_dimensions = [5, 3, 3], stride_sizes = [4, 1, 1], padding_sizes = [1, 1, 1]):
        super().__init__()
        self.name = 'AudioCNN'

        input_dimension_x = 64
        input_dimension_y = 292
        pool_kernel_dimension = 3
        pool_stride_size = 2
        out_channels = 512

        self.conv1 = nn.Conv2d(1, 32, kernel_dimensions[0], stride=stride_sizes[0], padding=padding_sizes[0])
        self.conv2 = nn.Conv2d(32, 64,  kernel_dimensions[1], stride=stride_sizes[1], padding=padding_sizes[1])
        self.conv3 = nn.Conv2d(64, 128,  kernel_dimensions[2], stride=stride_sizes[2], padding=padding_sizes[2])
        self.conv4 = nn.Conv2d(128, 256,  kernel_dimensions[2], stride=stride_sizes[2], padding=padding_sizes[2])
        self.conv5 = nn.Conv2d(256, out_channels,  kernel_dimensions[2], stride=stride_sizes[2], padding=padding_sizes[2])
        self.pool = nn.MaxPool2d(pool_kernel_dimension, pool_stride_size)

        oc1_x = math.floor((input_dimension_x + 2 * padding_sizes[0]- kernel_dimensions[0]) / stride_sizes[0]) + 1
        op1_x = math.floor((oc1_x - pool_kernel_dimension) / pool_stride_size) + 1
        oc2_x = math.floor((op1_x + 2 * padding_sizes[1] - kernel_dimensions[1]) / stride_sizes[1]) + 1
        op2_x = math.floor((oc2_x - pool_kernel_dimension) / pool_stride_size) + 1
        oc3_x = math.floor((op2_x + 2 * padding_sizes[2] - kernel_dimensions[2]) / stride_sizes[2]) + 1
        oc4_x = math.floor((oc3_x + 2 * padding_sizes[2] - kernel_dimensions[2]) / stride_sizes[2]) + 1
        oc5_x = math.floor((oc4_x + 2 * padding_sizes[2] - kernel_dimensions[2]) / stride_sizes[2]) + 1
        o_x = math.floor((oc5_x - pool_kernel_dimension) / pool_stride_size) + 1

        oc1_y = math.floor((input_dimension_y + 2 * padding_sizes[0]- kernel_dimensions[0]) / stride_sizes[0]) + 1
        op1_y = math.floor((oc1_y - pool_kernel_dimension) / pool_stride_size) + 1
        oc2_y = math.floor((op1_y + 2 * padding_sizes[1] - kernel_dimensions[1]) / stride_sizes[1]) + 1
        op2_y = math.floor((oc2_y - pool_kernel_dimension) / pool_stride_size) + 1
        oc3_y = math.floor((op2_y + 2 * padding_sizes[2] - kernel_dimensions[2]) / stride_sizes[2]) + 1
        oc4_y = math.floor((oc3_y + 2 * padding_sizes[2] - kernel_dimensions[2]) / stride_sizes[2]) + 1
        oc5_y = math.floor((oc4_y + 2 * padding_sizes[2] - kernel_dimensions[2]) / stride_sizes[2]) + 1
        o_y = math.floor((oc5_y - pool_kernel_dimension) / pool_stride_size) + 1
        self.fc_input_size = out_channels * o_x * o_y

        self.fc1 = nn.Linear(self.fc_input_size, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 8)

    def forward(self, x):
        # c1 + p1
        x = self.pool(torch.relu(self.conv1(x)))
        # c2 + p2
        x = self.pool(torch.relu(self.conv2(x)))
        # c3, c4
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        # c5 + p3
        x = self.pool(torch.relu(self.conv5(x)))
        # print(self.fc_input_size)
        x = x.view(-1, self.fc_input_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class ModAudioCNN8(nn.Module):
    def __init__(self, kernel_dimensions = [5, 3, 3], stride_sizes = [4, 1, 1], padding_sizes = [1, 1, 1]):
        super().__init__()
        self.name = 'ModAudioCNN8'

        input_dimension_x = 64
        input_dimension_y = 292
        pool_kernel_dimension = 3
        pool_stride_size = 2
        out_channels = 100

        self.conv1 = nn.Conv2d(1, 10, kernel_dimensions[0], stride=stride_sizes[0], padding=padding_sizes[0])
        self.conv2 = nn.Conv2d(10, 30,  kernel_dimensions[1], stride=stride_sizes[1], padding=padding_sizes[1])
        self.conv3 = nn.Conv2d(30, 50,  kernel_dimensions[2], stride=stride_sizes[2], padding=padding_sizes[2])
        self.conv4 = nn.Conv2d(50, 75,  kernel_dimensions[2], stride=stride_sizes[2], padding=padding_sizes[2])
        self.conv5 = nn.Conv2d(75, out_channels,  kernel_dimensions[2], stride=stride_sizes[2], padding=padding_sizes[2])
        self.pool = nn.MaxPool2d(pool_kernel_dimension, pool_stride_size)

        oc1_x = math.floor((input_dimension_x + 2 * padding_sizes[0]- kernel_dimensions[0]) / stride_sizes[0]) + 1
        op1_x = math.floor((oc1_x - pool_kernel_dimension) / pool_stride_size) + 1
        oc2_x = math.floor((op1_x + 2 * padding_sizes[1] - kernel_dimensions[1]) / stride_sizes[1]) + 1
        op2_x = math.floor((oc2_x - pool_kernel_dimension) / pool_stride_size) + 1
        oc3_x = math.floor((op2_x + 2 * padding_sizes[2] - kernel_dimensions[2]) / stride_sizes[2]) + 1
        oc4_x = math.floor((oc3_x + 2 * padding_sizes[2] - kernel_dimensions[2]) / stride_sizes[2]) + 1
        oc5_x = math.floor((oc4_x + 2 * padding_sizes[2] - kernel_dimensions[2]) / stride_sizes[2]) + 1
        o_x = math.floor((oc5_x - pool_kernel_dimension) / pool_stride_size) + 1

        oc1_y = math.floor((input_dimension_y + 2 * padding_sizes[0]- kernel_dimensions[0]) / stride_sizes[0]) + 1
        op1_y = math.floor((oc1_y - pool_kernel_dimension) / pool_stride_size) + 1
        oc2_y = math.floor((op1_y + 2 * padding_sizes[1] - kernel_dimensions[1]) / stride_sizes[1]) + 1
        op2_y = math.floor((oc2_y - pool_kernel_dimension) / pool_stride_size) + 1
        oc3_y = math.floor((op2_y + 2 * padding_sizes[2] - kernel_dimensions[2]) / stride_sizes[2]) + 1
        oc4_y = math.floor((oc3_y + 2 * padding_sizes[2] - kernel_dimensions[2]) / stride_sizes[2]) + 1
        oc5_y = math.floor((oc4_y + 2 * padding_sizes[2] - kernel_dimensions[2]) / stride_sizes[2]) + 1
        o_y = math.floor((oc5_y - pool_kernel_dimension) / pool_stride_size) + 1
        self.fc_input_size = out_channels * o_x * o_y

        self.fc1 = nn.Linear(self.fc_input_size, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 8)

    def forward(self, x):
        # c1 + p1
        x = self.pool(torch.relu(self.conv1(x)))
        # c2 + p2
        x = self.pool(torch.relu(self.conv2(x)))
        # c3, c4
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        # c5 + p3
        x = self.pool(torch.relu(self.conv5(x)))
        # print(self.fc_input_size)
        x = x.view(-1, self.fc_input_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x