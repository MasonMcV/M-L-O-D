import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D

import os
import sys
import numpy as np
import random



def dataGenerator(directory, batchSize):
    while True:
        inputTensor = np.empty((0, 1081, 2))  # batch size, data width, channels
        outputTensor = np.empty((0, 1081, 2))  # batch size, data width, classes
        for i in range(0, batchSize):
            filename = random.choice(os.listdir(directory))  # random.sample() would pick unique files
            path = os.path.join(directory, filename)
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

        #print(np.shape(inputTensor))
        yield inputTensor, outputTensor


model = Sequential()
model.add(Conv1D(8, 3, input_shape=(1081, 2), padding='same', dilation_rate=3, activation='relu'))
model.add(Conv1D(16, 3, padding='same', dilation_rate=3, activation='relu'))
model.add(Conv1D(8, 3, padding='same', dilation_rate=3, activation='relu'))
model.add(Conv1D(2, 3, padding='same', dilation_rate=3, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(dataGenerator('data/', 13), epochs=2, steps_per_epoch=13)


#model.summary()

test = model.predict_generator(dataGenerator('test/', 1), steps=1)

print model.metrics_names

np.set_printoptions(threshold=sys.maxsize)

print test

# score = model.evaluate(dataGenerator('data/', 5), batch_size=5)

# print score
