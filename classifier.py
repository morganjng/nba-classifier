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

# Normalizing so that we can process images easily
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize([112, 112]),
    ]
)


def one_hot_tensor(x):
    l = [0.0 for i in range(151)]
    l[x - 1] = 1.0
    return torch.tensor(l)


# Convolutional neural net class
# This needs work. I think the brunt of the work here is getting from our, currently 3x120x120, jgps and 4x120x120 pngs to
# a 151 length result vector through convolution. I think the LeNet5 diagram has a good indication of what
# to do but if any of you have more of an idea let me know and put something in the README.md file.
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(3, 2)
        self.conv1 = nn.Conv2d(3, 48, 11, stride=4)
        self.conv2 = nn.Conv2d(48, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 192, 3, padding=1)
        self.conv4 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 128, 3, padding=1)
        self.lin1 = nn.Linear(512, 151)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(self.pool(x))
        x = self.conv3(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.pool(x))
        x = torch.flatten(x)
        x = F.relu(self.lin1(x))
        return x


def train():
    images = {
        label: [
            "train/" + str(label) + "/" + image_name
            for image_name in os.listdir("train/" + str(label) + "/")
        ]
        for label in pdns
    }

    for i in pdns:
        print("Training PDN: " + str(i))
        inputs, labels = (
            [transform(Image.open(path).convert("RGB")) for path in images[i]],
            [one_hot_tensor(i) for x in images[i]],
        )

        optimizer.zero_grad()

        for j in range(len(inputs)):
            x = inputs[j]
            y = labels[j]
            out = cnn(x)
            opt = loss(out, y)
            opt.backward()
            optimizer.step()

    torch.save(cnn.state_dict(), "cnn.pth")


def test():
    cnn.load_state_dict(torch.load("cnn.pth"))
    test_images = {
        label: [
            "test/" + str(label) + "/" + image_name
            for image_name in os.listdir("test/" + str(label) + "/")
        ]
        for label in pdns
    }

    count = 0
    correct = 0
    for i in pdns:
        inputs, labels = (
            [transform(Image.open(path).convert("RGB")) for path in test_images[i]],
            [one_hot_tensor(i) for x in test_images[i]],
        )

        print("Testing PDN: " + str(i))

        prev_count = count
        prev_correct = correct
        for j in range(len(inputs)):
            x = inputs[j]
            y = labels[j]
            res = cnn(x)
            print(res)
            print(y)
            print("++++++++++++++++++++++++++==")
            print("True " + str(torch.argmax(y)) + " Guess " + str(torch.argmax(res)))
            if torch.argmax(res) == torch.argmax(y):
                correct += 1
            count += 1

        print(
            "For PDN: "
            + str(i)
            + " got "
            + str(correct - prev_correct)
            + " out of "
            + str(count - prev_count)
            + " correct"
        )


# Initializations
cnn = CNN()
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

# Paths to images for training data in our repo
pdns = sorted([int(x) for x in os.listdir("train/")])

train()
test()
