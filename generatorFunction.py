import os
import sys
import numpy as np
import random

np.set_printoptions(threshold=np.nan)

def dataGenerator(dir, batchSize):
    filename = random.choice(os.listdir(dir))
    path = os.path.join(dir,filename)
    r, intensity, label = np.loadtxt(path, delimiter=' ', usecols=(0,2,3), unpack=True)

    # normalize intensity
    intensity = (intensity - np.min(intensity)) / np.ptp(intensity)

    print("Chose file {}".format(filename))
    print("R:")
    print(r)
    print("INTENSITY:")
    print(intensity)
    print("LABEL:")
    print(label)

dataGenerator("/Users/matthew/Box/Astrobotics Software/ml-od/11-17/labeled", 1)