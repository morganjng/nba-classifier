#!/usr/bin/env python3

# Imports
from PIL import Image
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Normalizing so that we can process images easily
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize([224, 224]),
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
        self.sm = nn.Softmax()
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.pool3 = nn.MaxPool2d(3, stride=2)
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.flat = nn.Flatten(0)
        self.lin1 = nn.Linear(6400, 3200)
        self.lin2 = nn.Linear(3200, 1600)
        self.lin3 = nn.Linear(1600, 151)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.relu(x))
        x = self.conv2(x)
        x = self.pool2(F.relu(x))
        x = self.conv3(x)
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.pool3(F.relu(x))
        x = self.flat(x)
        x = self.lin1(x)
        x = self.lin2(F.relu(x))
        x = self.lin3(x)
        return x


def train():
    images = {
        label: [
            "train/" + str(label) + "/" + image_name
            for image_name in os.listdir("train/" + str(label) + "/")
        ]
        for label in pdns
    }

    count = 0
    rl = 0.0
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
            # print(cnn.conv1.parameters())
            print(out, torch.argmax(out))
            opt = loss(out, y)
            opt.backward()
            optimizer.step()
            count += 1
            rl += opt.item()
            if count % 50 == 0:
                print(str(count) + " with loss of " + str(rl))
                rl = 0.0

    torch.save(cnn.state_dict(), "cnn.pth")


def neo_train():
    test_images = {
        label: [
            "train/" + str(label) + "/" + image_name
            for image_name in os.listdir("train/" + str(label) + "/")
        ]
        for label in pdns
    }

    testx = []
    for i in pdns:
        for j in range(len(test_images[i])):
            testx.append([test_images[i][j], torch.LongTensor([i - 1.0])])

    print(len(testx))
    random.shuffle(testx)
    print(len(testx))

    count = 0
    rl = 0.0
    for XY in testx:
        # print(XY[0])
        res = cnn(transform(Image.open(XY[0]).convert("RGB")))
        # print(res, torch.argmax(res), XY[1][0])
        l = loss(res, XY[1][0])
        l.backward()
        optimizer.step()
        count += 1
        rl += l.item()
        if count % 200 == 0:
            print(str(count) + " with loss of " + str(rl))
            rl = 0.0

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
        guesses = []
        for j in range(len(inputs)):
            x = inputs[j]
            y = labels[j]
            res = cnn(x)
            print(res)
            print("True " + str(torch.argmax(y)) + " Guess " + str(torch.argmax(res)))
            guesses.append(torch.argmax(res))
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
            + " correct",
            guesses,
        )


# Initializations
cnn = CNN()
# cnn = torchvision.models.AlexNet(num_classes=151)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# Paths to images for training data in our repo
pdns = sorted([int(x) for x in os.listdir("train/")])

# neo_train()
test()
neo_train()
test()
neo_train()
test()
neo_train()
test()
neo_train()
test()
