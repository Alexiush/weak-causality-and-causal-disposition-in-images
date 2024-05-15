"""Module implementing a stop gradient layer"""

import tensorflow as tf

class StopLayer(tf.keras.layers.Layer):
  """Layer wrapper for tf.stop_gradient"""

  def __init__(self):
    super(StopLayer, self).__init__()

  def build(self, input_shape):
    pass

  def call(self, inputs):
    return tf.stop_gradient(inputs)