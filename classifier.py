#!/usr/bin/env python3

# Imports
from PIL import Image
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Convolutional neural net class
# This needs work. I think the brunt of the work here is getting from our, currently 3x120x120, jgps and 4x120x120 pngs to
# a 151 length result vector through convolution. I think the LeNet5 diagram has a good indication of what
# to do but if any of you have more of an idea let me know and put something in the README.md file.
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 6)
        self.fc1 = nn.Linear(729, 151)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x))
        return x


# Initializations
cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

# Paths to images for training data in our repo
labels = sorted([int(x) for x in os.listdir("train/")])
images = {
    label: [
        "train/" + str(label) + "/" + image_name
        for image_name in os.listdir("train/" + str(label) + "/")
    ]
    for label in labels
}

# Normalizing so that we can process images easily
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([120, 120])])

for i in [2, 4]:
    inputs, labels = (
        [transform(Image.open(path)) for path in images[i]],
        [i for x in images[i]],
    )

    optimizer.zero_grad()

    outputs = [cnn(i) for i in iter(inputs)]
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
