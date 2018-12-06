import os
import sys
import numpy as np
import random

# np.set_printoptions(threshold=np.nan)

def dataGenerator(dir, batchSize):
    while True:
        inputTensor = np.empty((0, 1081, 2))
        outputTensor = np.empty((0, 1081, 2))
        for j in range(0, batchSize):
            filename = random.choice(os.listdir(dir)) #random.sample() would pick unique files
            path = os.path.join(dir, filename)
            r, intensity, label = np.loadtxt(path, delimiter=' ', usecols=(0,2,3), unpack=True)

            # normalize intensity
            intensity = (intensity - np.min(intensity)) / np.ptp(intensity)

            obstacle = label.astype(int)
            notObstacle = 1 - label.astype(int)

            inputTensor = np.append(inputTensor, np.dstack((r, intensity)), axis=0)
            outputTensor = np.append(outputTensor, np.dstack((obstacle, notObstacle)), axis=0)

        yield inputTensor, outputTensor

for tensor in dataGenerator("/Users/matthew/astrobot/rmc-ml-od/11-17/labeled", 2):
    print(tensor)
    break