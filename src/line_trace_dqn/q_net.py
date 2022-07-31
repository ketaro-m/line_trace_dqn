#! /usr/bin/env python

"""
https://qiita.com/poorko/items/c151ff4a827f114fe954
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# for debug
import os
import cv2

class QNet(nn.Module):
    def __init__(self, action_size, seed=0):
        super(QNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 6, 5) # input_shape = (batch_size, 3, 36, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 6, 120) # be careful, (batch_size, 16, 13, 6)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, action_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_image = cv2.imread(os.path.join(os.path.dirname(__file__), "lenna.jpeg"))
    input_image = cv2.resize(input_image, (36,64))
    net = QNet(action_size=5)
    input_image = input_image.transpose(2,0,1)
    state = torch.from_numpy(input_image).float().unsqueeze(0)
    print(state.shape)
    x = net.forward(state)
    print(x.shape)
