
import pandas as pd
import os
import numpy as np
import torch
import random

MNIST_SIZE = 28

raw_path = os.path.join("./data", "mnist_raw.csv")
raw_data = pd.read_csv(raw_path, header=None)

x_raw = np.array(raw_data)
x_raw = torch.FloatTensor(x_raw)

# create label.csv

label = x_raw.clone()
label = label.numpy()
x_df = pd.DataFrame(label[:, 0])
x_df.to_csv('label.csv')


# create mnist_raw.csv

x_before = x_raw.clone()
x_before = x_before.numpy()
x_df = pd.DataFrame(x_before[:, 1:])
x_df.to_csv('mnist_before.csv')


# create transfrom.csv
x_after = np.copy(x_before[:, 1:])
MNIST_SIZE = 28

# left Translation
curIndex = 0
batchSize = 6000
for index in range(curIndex, curIndex + batchSize):
    randomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if j + randomShift < MNIST_SIZE - 1:
                x_after[index, pixel] = x_after[index, pixel + randomShift]
            else:
                x_after[index, pixel] = 0

curIndex += batchSize

# right Translation
for index in range(curIndex, curIndex + batchSize):
    randomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if j < randomShift:
                x_after[index, pixel] = 0
            else:
                x_after[index, pixel] = x_after[index, pixel - randomShift]

curIndex += batchSize

# up Translation
for index in range(curIndex, curIndex + batchSize):
    randomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if i + randomShift < MNIST_SIZE - 1:
                newPixel = (i+randomShift)*MNIST_SIZE + j
                x_after[index, pixel] = x_after[index, newPixel]
            else:
                x_after[index, pixel] = 0

curIndex += batchSize

# down Translation
for index in range(curIndex, curIndex + batchSize):
    randomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if i < randomShift:
                x_after[index, pixel] = 0
            else:
                newPixel = (i-randomShift)*MNIST_SIZE + j
                x_after[index, pixel] = x_after[index, newPixel]

curIndex += batchSize

# up left Translation
for index in range(curIndex, curIndex + batchSize):
    iRandomShift = random.randint(1, 7)
    jRandomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if j + jRandomShift > MNIST_SIZE - 1 or i + iRandomShift > MNIST_SIZE - 1:
                x_after[index, pixel] = 0
            else:
                newPixel = (i+iRandomShift)*MNIST_SIZE + j + jRandomShift
                x_after[index, pixel] = x_after[index, newPixel]

curIndex += batchSize

# down left Translation
for index in range(curIndex, curIndex + batchSize):
    iRandomShift = random.randint(1, 7)
    jRandomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if j + jRandomShift > MNIST_SIZE - 1 or i < iRandomShift:
                x_after[index, pixel] = 0
            else:
                newPixel = (i-iRandomShift)*MNIST_SIZE + j + jRandomShift
                x_after[index, pixel] = x_after[index, newPixel]

curIndex += batchSize

# up right Translation
for index in range(curIndex, curIndex + batchSize):
    iRandomShift = random.randint(1, 7)
    jRandomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if j < jRandomShift or i + iRandomShift > MNIST_SIZE - 1:
                x_after[index, pixel] = 0
            else:
                newPixel = (i+iRandomShift)*MNIST_SIZE + j - jRandomShift
                x_after[index, pixel] = x_after[index, newPixel]

curIndex += batchSize

# down right Translation
for index in range(curIndex, curIndex + batchSize):
    iRandomShift = random.randint(1, 7)
    jRandomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if j < jRandomShift or i < iRandomShift:
                x_after[index, pixel] = 0
            else:
                newPixel = (i-iRandomShift)*MNIST_SIZE + j - jRandomShift
                x_after[index, pixel] = x_after[index, newPixel]

curIndex += batchSize

# noise added
mu, sigma = 0, 10  # mean and standard deviation
noise = np.random.normal(mu, sigma, 784)
for index in range(curIndex, curIndex + batchSize):
    for i in range(MNIST_SIZE * MNIST_SIZE):
        x_after[index, pixel] = x_after[index, pixel] + noise[i]

curIndex += batchSize

# invert contrast
MNIST_MAX = 255
for index in range(curIndex, curIndex + batchSize):
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            x_after[index, pixel] = -x_after[index, pixel] + MNIST_MAX

curIndex += batchSize


x_df = pd.DataFrame(x_after)
x_df.to_csv('mnist_after.csv')
