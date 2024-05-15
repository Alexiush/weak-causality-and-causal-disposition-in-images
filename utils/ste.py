"""
Module implementing straight through estimator: 
https://discuss.pytorch.org/t/binary-activation-function-with-pytorch/56674/4
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer

def clip_by_value_preserve_gradient(
        tensor: tf.Tensor, 
        clip_value_min: float, 
        clip_value_max: float,
    ) -> tf.Tensor:
    """Clips the tensor without affecting its gradient"""

    clipped_tensor = tf.clip_by_value(tensor, clip_value_min, clip_value_max)
    return tensor + tf.stop_gradient(clipped_tensor - tensor)

@tf.custom_gradient
def binairy_STE(tensor: tf.Tensor, threshold: float) -> tf.Tensor:
    """Calculates binary activation 1 / 0 based on value passing / not passing given threshold"""

    with tf.GradientTape() as tape:
        tape.watch(threshold)
        result = tf.math.maximum(tf.zeros_like(tensor), tensor - threshold)
        result = result / result
        result = tf.where(tf.math.is_nan(result), tf.zeros_like(result), result)
    
    dt = tape.gradient(result, threshold)
    
    def grad(dy):
        return clip_by_value_preserve_gradient(dy, -1, 1), dt
    
    return result, grad

class StraightThroughEstimator(Layer):
    """Layer wrapper for binary_STE function"""

    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def call(self, args: tuple[tf.Tensor, float]) -> tf.Tensor:
        values, thresholds = args
        return binairy_STE(values, thresholds)