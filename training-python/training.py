from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D
import tensorflow as tf

from keras import backend as K

import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import math

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


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


inputTensor = np.empty((0, 1081, 2))
outputTensor = np.empty((0, 1081, 2))
filename = random.choice(os.listdir('test/'))  # random.sample() would pick unique files
path = os.path.join('test/', filename)
r, theta, intensity, label = np.loadtxt(path, delimiter=',', usecols=(0, 1, 2, 3), unpack=True)


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




prediction = model.predict(inputTensor)

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, "./", "my_model.pb", as_text=False)


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk")

print
print(model.metrics_names)

np.set_printoptions(threshold=sys.maxsize)

# score = model.evaluate(dataGenerator('data/', 5), batch_size=5)

# print score
fig = plt.figure()

X = np.array([])
i = 0
for point, thetaPoint in zip(inputTensor[0], theta):
    value = point[0] * math.cos(thetaPoint)
    X = np.append(X, value)

Y = np.array([])
for point, thetaPoint in zip(inputTensor[0], theta):
    value = point[0] * math.sin(thetaPoint)
    Y = np.append(Y, value)

print(np.shape(outputTensor))
print(np.shape(inputTensor))
for i in range(len(X)):
    if outputTensor[0, i, 0] > outputTensor[0, i, 1]:
        plt.plot(X[i], Y[i], color='yellow', marker='x', markersize=1, picker=5)
    else:
        plt.plot(X[i], Y[i], color='blue', marker='+', markersize=1, picker=5)
ax = plt.gca()
ax.set_title('test')
ax.set_facecolor('black')

plt.show()