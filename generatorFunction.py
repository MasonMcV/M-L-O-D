import os
import sys
import numpy as np
import random

# np.set_printoptions(threshold=np.nan)

def dataGenerator(dir, batchSize):
    while True:
        inputTensor = np.empty((0, 1081, 2)) # batch size, data width, channels
        outputTensor = np.empty((0, 1081, 2)) # batch size, data width, classes
        for j in range(0, batchSize):
            filename = random.choice(os.listdir(dir)) # random.sample() would pick unique files
            path = os.path.join(dir, filename)
            r, intensity, label = np.loadtxt(path, delimiter=' ', usecols=(0,2,3), unpack=True)

            # to flip, or not to flip
            if random.choice((True, False)):
                r         = np.flip(r, 0)
                intensity = np.flip(intensity, 0)
                label     = np.flip(label, 0)

            # add gaussian noise
            r         += np.random.normal(0, 1, r.shape) # median, std dev, size
            intensity += np.random.normal(0, 1, intensity.shape)

            # normalize intensity
            intensity = (intensity - np.min(intensity)) / np.ptp(intensity)

            obstacle = label.astype(int)
            notObstacle = 1 - label.astype(int)

            inputTensor = np.append(inputTensor, np.dstack((r, intensity)), axis=0)
            outputTensor = np.append(outputTensor, np.dstack((obstacle, notObstacle)), axis=0)

        yield inputTensor, outputTensor

for tensor in dataGenerator("11-17/labeled", 2):
    print(tensor)
    break