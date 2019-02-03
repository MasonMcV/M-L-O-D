import numpy as np
import sys
import time
from hokuyolx import HokuyoLX
import keras

laser = HokuyoLX(addr=('192.168.11.40', 10940))

model = keras.models.load_model("model.h5")

model.load_weights("Model.model")

np.set_printoptions(threshold=sys.maxsize)
while 1:
    # timestamp, scan = laser.get_dist()
    intens = laser.get_intens()
    angles = laser.get_angles()
    # print intens[0]
    # print intens[1]
    inputTensor = np.empty((0, 1081, 2))
    print len(intens[1])
    inputTensor = np.append(inputTensor, [intens[1]], axis=0)

    prediction = model.predict(inputTensor)

    print prediction

    # print angles
    time.sleep(.33)
