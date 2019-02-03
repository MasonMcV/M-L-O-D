from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, BatchNormalization
from keras import backend as K
from keras import optimizers
import tensorflow as tf
#from sklearn.preprocessing import StandardScaler
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from keras.optimizers import adam
from MeanIoU import MeanIoU

def dataGenerator(directory, batchSize, flip=False, noise=False, shift=False):
    while True:
        inputTensor = np.empty((0, 1081, 2))  # batch size, data width, channels
        outputTensor = np.empty((0, 1081, 2))  # batch size, data width, classes
        for i in range(0, batchSize):
            filename = random.choice(os.listdir(directory))  # random.sample() would pick unique files
            path = os.path.join(directory, filename)
            r, intensity, label = np.loadtxt(path, delimiter=',', usecols=(0, 2, 3), unpack=True)

            # to flip, or not to flip
            if flip and random.choice((True, False)):
                r         = np.flip(r, 0)
                intensity = np.flip(intensity, 0)
                label     = np.flip(label, 0)

            # add gaussian noise
            if noise:
                r         += np.random.normal(0, 0.01, r.shape) # median, std dev, size
                intensity += np.random.normal(0, 0.01, intensity.shape)

            if shift:
                indexShift = random.choice(range(-150, 150))
                np.roll(r, indexShift)
                np.roll(intensity, indexShift)

            # normalize
            # r = (r - np.min(r)) / np.ptp(r)
            intensity = intensity / 262144.0#(intensity - np.min(intensity)) / np.ptp(intensity)

            # Standardize
            # r = (r - np.mean(r)) / np.std(r)
            # intensity = (intensity - np.mean(intensity)) / np.std(intensity)


            obstacle = label.astype(int)
            notObstacle = 1 - label.astype(int)

            inputTensor = np.append(inputTensor, np.dstack((r, intensity)), axis=0)
            outputTensor = np.append(outputTensor, np.dstack((obstacle, notObstacle)), axis=0)

       # print(outputTensor)

        yield inputTensor, outputTensor


def create_model():
    model = Sequential()
    model.add(Conv1D(32, 3, input_shape=(1081, 2), padding='same', dilation_rate=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(32, 3, padding='same', dilation_rate=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(32, 3, padding='same', dilation_rate=4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(32, 3, padding='same', dilation_rate=8, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(32, 3, padding='same', dilation_rate=16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(32, 3, padding='same', dilation_rate=32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(32, 3, padding='same', dilation_rate=64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv1D(2, 3, padding='same', dilation_rate=1, activation='softmax'))

    num_classes = 2
    miou_metric = MeanIoU(num_classes)

    model.compile(optimizer=adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', miou_metric.mean_iou])
    return model


model = create_model()
print('params:')
print(model.count_params())

model.fit_generator(dataGenerator('data/', batchSize=16, flip=True, noise=True, shift=True),
                    epochs=100,
                    steps_per_epoch=20)



for i in range(1, 10):
    filename = random.choice(os.listdir('test/'))  # random.sample() would pick unique files
    path = os.path.join('test/', filename)
    r, theta, intensity, label = np.loadtxt(path, delimiter=',', usecols=(0, 1, 2, 3), unpack=True)
    inputTensor = np.empty((0, 1081, 2))


    r = np.flip(r, 0)
    intensity = np.flip(intensity, 0)
    label = np.flip(label, 0)

    # add gaussian noise
    # r += np.random.normal(0, 0.01, r.shape)  # median, std dev, size
    # intensity += np.random.normal(0, 0.01, intensity.shape)

    indexShift = random.choice(range(-150, 150))
    np.roll(r, indexShift)
    np.roll(intensity, indexShift)

    # Normalize
    #intensity = (intensity - np.min(intensity)) / np.ptp(intensity)
    intensity = intensity / 262144
    # r = (r - np.min(r)) / np.ptp(r)

    # Standardize
    # r = (r - np.mean(r)) / np.std(r)
    # intensity = (intensity - np.mean(intensity)) / np.std(intensity)


    obstacle = label.astype(int)
    notObstacle = 1 - label.astype(int)

    inputTensor = np.append(inputTensor, np.dstack((r, intensity)), axis=0)


    predict = model.predict(inputTensor)


    fig = plt.figure()

    X = np.array([])
    for point, thetaPoint in zip(inputTensor[0], theta):
        value = point[0] * math.cos(thetaPoint)
        X = np.append(X, value)

    Y = np.array([])
    for point, thetaPoint in zip(inputTensor[0], theta):
        value = point[0] * math.sin(thetaPoint)
        Y = np.append(Y, value)

    for i in range(len(X)):
        #print(predict[0, i, 0])
        if predict[0, i, 0] > 0.1:
            plt.plot(X[i], Y[i], color='yellow', marker='x', markersize=1, picker=5)
        else:
            plt.plot(X[i], Y[i], color='blue', marker='+', markersize=1, picker=5)
    ax = plt.gca()
    ax.set_title('test')
    ax.set_facecolor('black')

    plt.show()
