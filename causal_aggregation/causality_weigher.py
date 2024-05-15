"""Module implementing causal weighing of features"""

from utils.ste import StraightThroughEstimator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers

class CausalityWeigher(Layer):
    def __init__(
            self, 
            mulcatbool: bool = False, 
            init_weight: float = 0., 
            direction: str = "cause", 
            weights_shape: str = "global", 
            trainable_threshold: bool = False, 
            weights_mode: str = "threshold"
        ):
        super(CausalityWeigher, self).__init__()
        self.mulcatbool = mulcatbool
        self.init_weight = init_weight
        self.direction = direction
        
        self.weights_shape = weights_shape
        self.trainable_threshold = trainable_threshold
        self.weights_mode = weights_mode
        
        self.ste = StraightThroughEstimator()
        
        self.relu = layers.ReLU()
        self.flatten = layers.Flatten()

    def threshold_counter(self, assymetry: tf.Tensor, features: tf.Tensor) -> tf.Tensor:
        """Counts assymetries passing the independent threshold"""

        return tf.reduce_sum(
            self.ste([assymetry, self.t]), axis = 1
        )
    
    def linear_counter(self, assymetry: tf.Tensor, features: tf.Tensor) -> tf.Tensor:
        """Counts assymetries passing the linear function threshold"""

        t = features * self.w + self.b
        
        return tf.reduce_sum(
            self.ste([assymetry, t]), axis = 1
        )
    
    def add_w(self, name: str) -> tf.Variable:
        if self.weights_shape == "global":
            return self.add_weight(
                name=name,
                initializer=keras.initializers.Constant(self.init_weight),
                trainable=self.trainable_threshold,
                shape=[]
            )
        elif self.weights_shape == "local":
            return self.add_weight(
                name=name,
                initializer=keras.initializers.Constant(self.init_weight),
                trainable=self.trainable_threshold,
                shape=[self.features_count, self.features_count]
            )
        else:
            raise ValueError("Unknown weights type")
    
    def build(self, input_shape):
        self.features_count = input_shape[0][0][1]
    
        if self.weights_mode == "threshold":
            self.t = self.add_w("t")
            self.count_causes = self.threshold_counter
        elif self.weights_mode == "function":
            self.w = self.add_w("w")
            self.b = self.add_w("b")
        
            self.count_causes = self.linear_counter
        else:
            raise ValueError("Unknown weights mode")

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        """Calculates weight according to mulcat method and applies new weights to the features"""

        batch_size = tf.shape(inputs[0][0])[0]
        causality = inputs[0][0]
        features = inputs[1]

        matrix = tf.experimental.numpy.triu(causality)
        
        matrix_t = tf.experimental.numpy.tril(causality, -1)
        matrix_t = tf.transpose(matrix_t, perm=[0, 2, 1])
        
        matrix_pair = tf.concat([matrix[:,:,:,tf.newaxis], matrix_t[:,:,:,tf.newaxis]], -1)
        
        causes = self.count_causes(matrix - matrix_t, tf.reduce_min(matrix_pair, -1))
        effects = self.count_causes(matrix_t - matrix, tf.reduce_min(matrix_pair, -1))
        
        if self.direction == "cause":
            delta = causes - effects
        elif self.direction == "effect":
            delta = effects - causes
        else:
            raise ValueError("Unknown weighing direction")

        w = self.relu(delta) / self.features_count
        w = tf.cast(w, dtype=tf.float32)

        if self.mulcatbool:
            w = self.ste(w, tf.zeros_like(w))

        w = tf.reshape(w, [batch_size, 1, 1, self.features_count])
        
        return self.flatten(w * features)