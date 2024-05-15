"""Module implementing SimpleCNN-based models creation"""

import tensorflow as tf
import extractors

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from utils.stop_layer import StopLayer
from utils.module_builders import get_classifier, get_aggregator, get_causality, pass_to_classifier
from extractors.simple_cnn import simple_cnn_conv
from causal_aggregation.causality import CausalityMatrix

def SimpleCnn(
        num_classes: int, 
        num_features: int, 
        input_shape: list[int], 
        aggregation: str = "avg", 
        dropout: float = 0., 
        classifier: str = "classic"
    ) -> tf.keras.models.Model:    
    
    """
    Builds default (Not causality-aware) SimpleCNN model.

    Arguments:
    num_classes (int): number of classes to predict, 
    num_features (int): number of features for classifier, 
    input_shape (list[int]): shape of the input image, 
    aggregation (str): aggregation method used, 
    dropout (float): dropout rate, 
    classifier (str): classifier used
    """

    aggregation_layer = get_aggregator(aggregation)
    classifier = get_classifier(classifier, num_classes, num_features, dropout)
    conv = simple_cnn_conv()
    
    model_simple = Sequential([
        layers.InputLayer(input_shape=input_shape),
        conv,
        aggregation_layer,
        classifier,
    ])
    
    return model_simple

def SimpleCnnCA(
        num_classes: int, 
        num_features: int, 
        input_shape: list[int], 
        numerator: str | float = 0., 
        denominator: str | float = 0., 
        trainable_lehmer: bool = True, 
        causal_aggregator: str = "concat", 
        aggregation: str = "avg",
        dropout: float = 0., 
        classifier: str = "classic",  
        mulcat_direction: str = "cause",
        mulcatbool: bool = False,
        mulcat_init_weight: float = 0., 
        mulcat_weights_shape: str = "global", 
        mulcat_trainable_threshold: bool = False, 
        mulcat_weights_mode: str = "threshold",
        causality_application: str = "concatenated",
    ) -> tf.keras.models.Model:
    
    """
    Builds causality-aware SimpleCNN model.

    Arguments:
    num_classes (int): number of classes to predict, 
    num_features (int): number of features for classifier, 
    input_shape (list[int]): shape of the input image, 
    numerator (str | float): numerator to be used in causality estimation (float or special case keyword), 
    denominator: (str | float): denominator to be used in causality estimation (float or special case keyword), 
    trainable_lehmer (bool): whether to update parameters used for lehmer calculation or not (works only with parameters initialized with floats), 
    causal_aggregator (str): causal aggregation method used, 
    aggregation (str): aggregation method used, 
    dropout (float): dropout rate,
    classifier (str): classifier used
    mulcat_direction (str): mulcat computation direction, 
    mulcatbool (bool): should mulcat be used as a binary function,
    mulcat_init_weight (float): initial value for threshold computation,
    mulcat_weights_shape (str): should mulcat threshold be "global" or "local" (per-matrix or per-value), 
    mulcat_trainable_threshold (bool): should threshold be trained, 
    mulcat_weights_mode (str): which threshold should be applied: "variable" for independent variable or "function" for linear function from the input 
    causality_application (str): way to apply causality to features,
    """

    conv = Sequential([
        layers.InputLayer(input_shape=input_shape),
        simple_cnn_conv(),
        layers.ReLU(max_value = 1),
    ])
    
    ca = CausalityMatrix(
        numerator=numerator, 
        denominator=denominator, 
        trainable_lehmer=trainable_lehmer
    )
    causality_matrix = Sequential([
        layers.Permute((3, 1, 2)),
        ca
    ])
    
    aggregation_layer = get_aggregator(aggregation)
    classifier = get_classifier(classifier, num_classes, num_features, dropout)
    
    inputs = layers.Input(shape=input_shape)
    
    features = conv(inputs)
    features_processed = aggregation_layer(features)

    matrix = causality_matrix(features)
    
    matrix_processed = get_causality(
        causal_aggregator, matrix, features, mulcat_direction, mulcatbool, mulcat_init_weight, 
        mulcat_weights_shape, mulcat_trainable_threshold, mulcat_weights_mode
    )
    
    outputs = pass_to_classifier(causality_application, features_processed, matrix_processed, classifier)
    
    model_simple = tf.keras.models.Model(inputs = inputs, outputs = outputs, name = "simple_cnn_ca")
    
    return model_simple

def SimpleCnnCA_transfer(
        pretrained_simple_cnn: tf.keras.models.Model, 
        num_classes: int, 
        num_features: int, 
        input_shape: list[int], 
        numerator: str | float = 0., 
        denominator: str | float = 0., 
        trainable_lehmer: bool = True, 
        causal_aggregator: str = "concat", 
        aggregation: str = "avg", 
        dropout: float = 0., 
        classifier: str = "classic",  
        mulcat_direction: str = "cause", 
        mulcatbool: bool = False,
        mulcat_init_weight: float = 0., 
        causality_application: str = "concatenated",
        mulcat_weights_shape: str = "global", 
        mulcat_trainable_threshold: bool = False, 
        mulcat_weights_mode: str = "threshold"
    ) -> tf.keras.models.Model:
    
    """
    Builds causality-aware SimpleCNN model based on pre-trained feature extractor.

    Arguments:
    pretrained_simple_cnn (tf.keras.models.Model): pre-trained simple cnn,
    num_classes (int): number of classes to predict, 
    num_features (int): number of features for classifier, 
    input_shape (list[int]): shape of the input image, 
    numerator (str | float): numerator to be used in causality estimation (float or special case keyword), 
    denominator: (str | float): denominator to be used in causality estimation (float or special case keyword), 
    trainable_lehmer (bool): whether to update parameters used for lehmer calculation or not (works only with parameters initialized with floats), 
    causal_aggregator (str): causal aggregation method used, 
    aggregation (str): aggregation method used, 
    dropout (float): dropout rate,
    classifier (str): classifier used
    mulcat_direction (str): mulcat computation direction, 
    mulcatbool (bool): should mulcat be used as a binary function,
    mulcat_init_weight (float): initial value for threshold computation,
    mulcat_weights_shape (str): should mulcat threshold be "global" or "local" (per-matrix or per-value), 
    mulcat_trainable_threshold (bool): should threshold be trained, 
    mulcat_weights_mode (str): which threshold should be applied: "variable" for independent variable or "function" for linear function from the input 
    causality_application (str): way to apply causality to features,
    """

    conv = Sequential([
        layers.InputLayer(input_shape=input_shape),
        pretrained_simple_cnn,
        layers.ReLU(max_value = 1),
        StopLayer()
    ])
    
    ca = CausalityMatrix(
        numerator=numerator, 
        denominator=denominator, 
        trainable_lehmer=trainable_lehmer
    )
    causality_matrix = Sequential([
        layers.Permute((3, 1, 2)),
        ca
    ])
    
    aggregation_layer = get_aggregator(aggregation)
    classifier = get_classifier(classifier, num_classes, num_features, dropout)
    
    inputs = layers.Input(shape=input_shape)
    
    features = conv(inputs)
    flat_features = aggregation_layer(features)

    matrix = causality_matrix(features)
    
    flat_matrix = get_causality(
        causal_aggregator, matrix, features, mulcat_direction, mulcatbool, mulcat_init_weight, 
        mulcat_weights_shape, mulcat_trainable_threshold, mulcat_weights_mode
    )
    
    outputs = pass_to_classifier(causality_application, flat_features, flat_matrix, classifier)
    
    model_simple = tf.keras.models.Model(inputs = inputs, outputs = outputs, name = "simple_cnn_ca_transfer")
    
    return model_simple