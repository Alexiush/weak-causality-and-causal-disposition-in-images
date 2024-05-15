"""Module implementing causality matrix computation"""

from utils.lehmer_mean import LehmerMean
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

class CausalityMatrix(Layer):
    def __init__(self, numerator: float | str = 0., denominator: float | str = 0., trainable_lehmer: bool = True, calculates_grad: bool = False):
      super(CausalityMatrix, self).__init__()
      
      numerators_list = numerator if type(numerator) == list else [numerator]
      if any(numerator not in ["min", "max"] and not isinstance(numerator, float) for numerator in numerators_list):
        raise ValueError("Unknown numerator initializer (only max, min or [-inf; +inf] floats are allowed)")
      self.numerator = numerators_list
      
      denominators_list = denominator if type(denominator) == list else [denominator]
      if any(denominator not in ["sum", "mean", "min", "max"] and not isinstance(denominator, float) for denominator in denominators_list):
        raise ValueError("Unknown denominator initializer (only sum, mean, max, min or [-inf; +inf] floats are allowed)")
      self.denominator = denominators_list
      
      self.trainable_lehmer = trainable_lehmer
      self.calculates_grad = calculates_grad

    def build(self, input_shape):
      self.lehmer_numerators = []
      self.lehmer_denominators = []
          
      input_shape_list = input_shape if type(input_shape) == list else [input_shape]
      inputs_count = len(input_shape_list)
      inputs_count_squared = inputs_count * inputs_count
      
      self.shapes = []
      for i in range(inputs_count):
        self.shapes.append(input_shape_list[i])
      
      self.numerator = self.numerator if len(self.numerator) != 1 else self.numerator * inputs_count_squared
      self.denominator = self.denominator if len(self.denominator) != 1 else self.denominator * inputs_count_squared
      self.trainable_lehmer = self.trainable_lehmer if type(self.trainable_lehmer) == list else [self.trainable_lehmer] * inputs_count_squared
      
      for i in range(inputs_count_squared):
        if self.numerator[i] not in ["min", "max"]:
          self.lehmer_numerators.append(LehmerMean(initial_value=self.numerator[i], trainable_parameter=self.trainable_lehmer[i]))
        
        if self.denominator[i] not in ["min", "max", "sum", "mean"]:
          self.lehmer_denominators.append(LehmerMean(initial_value=self.denominator[i], trainable_parameter=self.trainable_lehmer[i]))

    def stop_check(self, tensor: tf.Tensor) -> tf.Tensor:
      """Helper function that checks whether gradient propagation should be stopped"""
      return tensor if self.calculates_grad else tf.stop_gradient(tensor)
            
    def get_numerator(self, features_i: tf.Tensor, features_j: tf.Tensor, index: int = 0) -> tf.Tensor:
      """
      Calculates numerator part of causality estimation function
      
      Arguments:
      features_i (tf.Tensor): first set of features
      features_j (tf.Tensor): second set of features
      index: 0-based index of the features source (n inputs result in n^2 feature source pairs) 

      Returns: tf.Tensor of computed numerator
      """

      if self.numerator[index] == "min": # -inf
        i_mins = self.stop_check(tf.reduce_min, features_i, axis=2)
        j_mins = self.stop_check(tf.reduce_min(features_j, axis=2))
        mins = tf.einsum('bi,bj->bij', i_mins, j_mins)
        return mins
      elif self.numerator[index] == "max": # +inf
        i_maxes = self.stop_check(tf.reduce_max(features_i, axis=2))
        j_maxes = self.stop_check(tf.reduce_max(features_j, axis=2))
        maxes = tf.einsum('bi,bj->bij', i_maxes, j_maxes)
        return maxes
      else:
        outer = tf.einsum('bmi,bnj->bmnij', features_i, features_j)
        lehmer = self.lehmer_numerators[index](self.stop_check(outer), axis=[3, 4])
        return lehmer

    def get_denominator(self, features, index=0):
      """
      Calculates denominator part of causality estimation function
      
      Arguments:
      features (tf.Tensor): second set of features
      index: 0-based index of the features source (n inputs result in n^2 feature source pairs) 

      Returns: tf.Tensor of computed denominator
      """

      if self.denominator[index] == "sum":
        return self.stop_check(tf.reduce_sum(features, axis=2))
      elif self.denominator[index] == "min": # -inf
        return self.stop_check(tf.reduce_min(features, axis=2))
      elif self.denominator[index] == "mean": # 1
        return self.stop_check(tf.reduce_mean(features, axis=2))
      elif self.denominator[index] == "max": # +inf
        return self.stop_check(tf.reduce_max(features, axis=2))
      else:
        return self.lehmer_denominators[index](self.stop_check(features), axis=2)
    
    def call(self, inputs: tf.Tensor | list[tf.Tensor]) -> list[tf.Tensor]:
      """Calculates causal estimates for each pair of feature tensors (tensors can form a pair with themselves)"""

      inputs_list = inputs if type(inputs) == list else [inputs]
      
      inputs_count = len(inputs_list)
      batch_size = tf.shape(inputs_list[0])[0]

      inputs_preprocessed = []
      
      epsilon = 1e-9
      for i in range(inputs_count):
        input = tf.where(tf.equal(inputs_list[i], 0), tf.ones_like(inputs_list[i]) * epsilon, inputs_list[i])
        input = tf.reshape(input, [batch_size, self.shapes[i][1], np.prod(self.shapes[i][2:])])
        inputs_preprocessed.append(input)
      
      causality_matrices = []

      for first in range(inputs_count):
        for second in range(inputs_count):
          i = inputs_preprocessed[first]
          j = inputs_preprocessed[second]
          index = first * inputs_count + second
          
          numerator = self.get_numerator(i, j, index=index)
          numerator = tf.where(tf.math.is_nan(numerator), tf.zeros_like(numerator), numerator)
          
          denominator = self.get_denominator(j, index=index)
          denominator = tf.tile(denominator, [1, self.shapes[first][1]])
          denominator = tf.reshape(denominator, [batch_size, self.shapes[first][1], self.shapes[second][1]])
          denominator = tf.where(tf.math.is_nan(denominator), tf.zeros_like(denominator), denominator)
          
          causality = numerator / denominator
          causality = tf.where(tf.math.is_nan(causality), tf.zeros_like(causality), causality)
          causality = tf.reshape(causality, [batch_size, self.shapes[first][1], self.shapes[second][1]])
          
          causality_matrices.append(causality)     

      return causality_matrices
