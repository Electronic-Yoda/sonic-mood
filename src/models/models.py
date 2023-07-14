import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch.optim as optim
import numpy as np
import time

# Initial model - primary model (for progress report)
class AudioCNN(nn.Module):
    def __init__(self, kernel_dimensions = [5, 2], stride_sizes = [1, 1], padding_sizes = [1, 1]):
        super().__init__()
        self.name = 'AudioCNN'

        input_dimension_x = 64
        input_dimension_y = 292
        pool_kernel_dimension = 2
        pool_stride_size = 2

        self.conv1 = nn.Conv2d(1, 6, kernel_dimensions[0], stride=stride_sizes[0], padding=padding_sizes[0])
        self.conv2 = nn.Conv2d(6, 12,  kernel_dimensions[1], stride=stride_sizes[1], padding=padding_sizes[1])
        self.pool = nn.MaxPool2d(2, 2)

        a_x = math.floor((input_dimension_x + 2 * padding_sizes[0]- kernel_dimensions[0]) / stride_sizes[0] + 1)
        b_x = math.floor((a_x - pool_kernel_dimension) / pool_stride_size + 1)
        c_x = math.floor((b_x + 2 * padding_sizes[1] - kernel_dimensions[1]) / stride_sizes[1] + 1)
        d_x = math.floor((c_x - pool_kernel_dimension) / pool_stride_size + 1)

        a_y = math.floor((input_dimension_y + 2 * padding_sizes[0]- kernel_dimensions[0]) / stride_sizes[0] + 1)
        b_y = math.floor((a_y - pool_kernel_dimension) / pool_stride_size + 1)
        c_y = math.floor((b_y + 2 * padding_sizes[1] - kernel_dimensions[1]) / stride_sizes[1] + 1)
        d_y = math.floor((c_y - pool_kernel_dimension) / pool_stride_size + 1)
        self.fc_input_size = 12 * d_x * d_y

        self.fc1 = nn.Linear(self.fc_input_size, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 8)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # print(self.fc_input_size)
        x = x.view(-1, self.fc_input_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
