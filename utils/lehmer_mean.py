"""Module containing an implementation of lehmer mean function"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from utils.non_nan import NonNan

def lehmer_mean(values: tf.Tensor, alpha: float = 0., axis: int | list[int] = 0) -> tf.Tensor:
    """
    Calculates the lehmer mean.
    
    Arguments:
    values (tf.Tensor): Tensor for which mean is calculated. Should have numeric type.
    alpha (float): power parameter (default 0.)
    axis (int or list[int]): axis along which lehmer mean is calculated (default 0)

    Returns: tf.Tensor containing lehmer means
    """
    powers_numerator = tf.reduce_sum(tf.pow(values, alpha + 1), axis=axis)
    powers_denominator = tf.reduce_sum(tf.pow(values, alpha), axis=axis)
    
    return powers_numerator / powers_denominator

class LehmerMean(Layer):
    """Manages additional parameters for lehmer mean computation"""

    def __init__(self, initial_value: float = 0., trainable_parameter: bool = True):
        super(LehmerMean, self).__init__()
        
        self.initial_value = initial_value
        self.trainable_parameter = trainable_parameter
        self.epsilon = 1e-9
        
        self.alpha = self.add_weight(
            name="alpha",
            initializer=keras.initializers.Constant(self.initial_value),
            trainable=self.trainable_parameter,
            constraint=NonNan(initial_value),
            shape=[]
        )

    def build(self, input_shape):
        pass

    def call(self, values: tf.Tensor, axis: int | list[int] = 0) -> tf.Tensor:
        return lehmer_mean(values, self.alpha, axis)