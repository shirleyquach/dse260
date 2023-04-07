#!/usr/bin/env python3
# From https://stackoverflow.com/a/59465405

import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1 import graph_util
from tensorflow.python.keras import backend as K

# necessary !!!
tf.compat.v1.disable_eager_execution()

h5_path = "/path/to/model.h5"
model = keras.models.load_model(h5_path)
model.summary()
# save pb
with K.get_session() as sess:
    output_names = [out.op.name for out in model.outputs]
    input_graph_def = sess.graph.as_graph_def()
    for node in input_graph_def.node:
        node.device = ""
    graph = graph_util.remove_training_nodes(input_graph_def)
    graph_frozen = graph_util.convert_variables_to_constants(sess, graph, output_names)
    tf.io.write_graph(graph_frozen, "/path/to/pb/model.pb", as_text=False)
logging.info("save pb successfully！")
