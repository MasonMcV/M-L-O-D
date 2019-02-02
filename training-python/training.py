import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D

import os
import sys
import numpy as np
import random


# np.set_printoptions(threshold=np.nan)

def dataGenerator(dir, batchSize):
    while True:
        inputTensor = np.empty((0, 1081, 2))  # batch size, data width, channels
        outputTensor = np.empty((0, 1081, 2))  # batch size, data width, classes
        for i in range(0, batchSize):
            filename = random.choice(os.listdir(dir))  # random.sample() would pick unique files
            path = os.path.join(dir, filename)
            r, intensity, label = np.loadtxt(path, delimiter=',', usecols=(0, 2, 3), unpack=True)

            # to flip, or not to flip
            # if random.choice((True, False)):
            #     r         = np.flip(r, 0)
            #     intensity = np.flip(intensity, 0)
            #     label     = np.flip(label, 0)

            # add gaussian noise
            # r         += np.random.normal(0, 1, r.shape) # median, std dev, size
            # intensity += np.random.normal(0, 1, intensity.shape)

            # normalize intensity
            intensity = (intensity - np.min(intensity)) / np.ptp(intensity)

            obstacle = label.astype(int)
            notObstacle = 1 - label.astype(int)

            inputTensor = np.append(inputTensor, np.dstack((r, intensity)), axis=0)
            outputTensor = np.append(outputTensor, np.dstack((obstacle, notObstacle)), axis=0)

        yield inputTensor, outputTensor


inputTensor = np.empty((0, 1081, 2))
outputTensor = np.empty((0, 1081, 2))

filename = random.choice(os.listdir('data/'))  # random.sample() would pick unique files
path = os.path.join(dir, filename)
r, intensity, label = np.loadtxt(path, delimiter=',', usecols=(0, 2, 3), unpack=True)

# to flip, or not to flip
# if random.choice((True, False)):
#     r         = np.flip(r, 0)
#     intensity = np.flip(intensity, 0)
#     label     = np.flip(label, 0)

# add gaussian noise
# r         += np.random.normal(0, 1, r.shape) # median, std dev, size
# intensity += np.random.normal(0, 1, intensity.shape)

# normalize intensity
intensity = (intensity - np.min(intensity)) / np.ptp(intensity)

obstacle = label.astype(int)
notObstacle = 1 - label.astype(int)

inputTensor = np.append(inputTensor, np.dstack((r, intensity)), axis=0)
outputTensor = np.append(outputTensor, np.dstack((obstacle, notObstacle)), axis=0)

# Generate dummy data
# x_train = np.random.random((1000, 20, 1))
# y_train = np.random.randint(1, size=(1000, 1))
# x_test = np.random.random((100, 20))
# y_test = np.random.randint(1, size=(100, 1))

model = Sequential()
model.add(Conv1D(8, 3, padding='same', dilation_rate=3, activation='relu'))
model.add(Conv1D(16, 3, padding='same', dilation_rate=3, activation='relu'))
model.add(Conv1D(8, 3, padding='same', dilation_rate=3, activation='relu'))
model.add(Conv1D(1, 3, padding='same', dilation_rate=3, activation='relu'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

dataset = dataGenerator('data/', 15)

model.fit(, epochs = 20)

# score = model.evaluate(dataGenerator('data/', 5), batch_size=5)

# print score
