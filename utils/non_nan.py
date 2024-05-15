"""Module containing NonNan constraint for tf.Variable"""
import tensorflow as tf

class NonNan(tf.keras.constraints.Constraint):
    def __init__(self, default_value: float = 0.):
        self.default_value = default_value
    
    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        return tf.where(tf.math.is_nan(w), tf.ones_like(w) * self.default_value, w)