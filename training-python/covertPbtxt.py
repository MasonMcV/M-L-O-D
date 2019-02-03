import tensorflow as tf
import numpy as np

graph_filename = "./trained.pb"
with tf.gfile.GFile(graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.train.write_graph(graph_def, './', 'frozen_graph.pbtxt', True)
