"""Module implementing experimental causal aggregator statcat"""

from utils.ste import StraightThroughEstimator
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
import numpy as np

class CausalityStatisticsExtractor(Layer):
    def __init__(self):
        super(CausalityStatisticsExtractor, self).__init__()
        
        self.ste = StraightThroughEstimator()        
        self.flatten = layers.Flatten()

    def build(self, input_shape):
        self.features_count = math.ceil(math.sqrt(input_shape[1]))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Calculates clearer statistics on asymetrical measures.

        Each pair in causality matrix is transformed to 3 values:
        * base - lower presence exercised by the contributing features
        * asymetry - difference between bigger and lower presence
        * direction - which feature has higher presence (-1 or 1)
        """

        batch_size = tf.shape(inputs)[0]
        
        matrix = tf.experimental.numpy.triu(inputs)
        
        matrix_t = tf.experimental.numpy.tril(inputs, -1)
        matrix_t = tf.transpose(matrix_t, perm=[0, 2, 1])
        
        diff = self.ste(matrix - matrix_t)
        direction = tf.math.sign(diff)
        asymetry = tf.math.abs(diff)
        
        pairs = tf.concat([matrix, matrix_t], -1)
        pairs = tf.reshape(pairs, [batch_size, self.features_count, self.features_count, 2])
        
        base = tf.math.reduce_min(pairs, -1)
        base = tf.reshape(base, [batch_size, self.features_count, self.features_count])
        
        metrics = tf.concat([direction, asymetry, base], -1)
        
        indices_y, indices_x = np.triu_indices(self.features_count)
        indices_y = indices_y * self.features_count
        indices = indices_y + indices_x
        
        metrics = tf.reshape(metrics, [batch_size, self.features_count * self.features_count, 3])
        metrics = tf.gather(metrics, indices=indices, axis=1)
        
        return self.flatten(metrics)