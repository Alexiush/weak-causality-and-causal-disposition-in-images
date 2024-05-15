"""Module implementing ResNet-based models creation"""

import tensorflow as tf
import extractors

from collections.abc import Callable
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from utils.stop_layer import StopLayer
from utils.module_builders import get_classifier, get_aggregator, get_causality, pass_to_classifier
from extractors.resnet import resnet_blocks
from causal_aggregation.causality import CausalityMatrix


def ResNet18(
        num_classes: int, 
        num_features: int, 
        input_shape: list[int],
        stack_fn: Callable[[None], tf.keras.models.Model] = extractors.resnet.stack_fn, 
        aggregation: str = "avg", 
        dropout: float = 0., 
        classifier: str = "classic"
    ) -> tf.keras.models.Model:
    """
    Builds default (Not causality-aware) ResNet18 model.

    Arguments:
    num_classes (int): number of classes to predict, 
    num_features (int): number of features for classifier, 
    input_shape (list[int]): shape of the input image,
    stack_fn (Callable[[None], tf.keras.models.Model]): residual blocks configuration for a model,
    aggregation (str): aggregation method used, 
    dropout (float): dropout rate, 
    classifier (str): classifier used
    """

    aggregation_layer = get_aggregator(aggregation)
    classifier = get_classifier(classifier, num_classes, num_features, dropout)
    
    inputs = layers.Input(shape=input_shape)
    
    features = resnet_blocks(input_shape, num_classes, stack_fn)(inputs)
    flat_features = aggregation_layer(features)
    outputs = classifier(flat_features)
    
    model_resnet_18 = tf.keras.models.Model(inputs = inputs, outputs = outputs, name = "resnet18_ca")

    return model_resnet_18

def ResNet18CA(
        num_classes: int, 
        num_features: int, 
        input_shape: list[int], 
        stack_fn: Callable[[None], tf.keras.models.Model] = extractors.resnet.stack_fn, 
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
    Builds causality-aware ResNet18 model.

    Arguments:
    num_classes (int): number of classes to predict, 
    num_features (int): number of features for classifier, 
    input_shape (list[int]): shape of the input image,
    stack_fn (Callable[[None], tf.keras.models.Model]): residual blocks configuration for a model,
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
        resnet_blocks(input_shape, num_classes, stack_fn),
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
    fc = get_classifier(classifier, num_classes, num_features, dropout)
    
    inputs = layers.Input(shape=input_shape)
    
    features = conv(inputs)
    flat_features = aggregation_layer(features)

    matrix = causality_matrix(features)
    
    flat_matrix = get_causality(
        causal_aggregator, matrix, features, mulcat_direction, mulcatbool, mulcat_init_weight, 
        mulcat_weights_shape, mulcat_trainable_threshold, mulcat_weights_mode
    )
    
    outputs = pass_to_classifier(causality_application, flat_features, flat_matrix, fc)
    
    model_resnet_18 = tf.keras.models.Model(inputs = inputs, outputs = outputs, name = "resnet18_ca")
    
    return model_resnet_18

def ResNet18CA_transfer(
        pretrained_resnet: tf.keras.models.Model, 
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
    Builds causality-aware ResNet18 model based on pre-trained feature extractor.

    Arguments:
    pretrained_resnet (tf.keras.models.Model): pre-trained ResNet18,
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
        pretrained_resnet,
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
    fc = get_classifier(classifier, num_classes, num_features, dropout)
    
    inputs = layers.Input(shape=input_shape)
    
    features = conv(inputs)
    flat_features = aggregation_layer(features)

    matrix = causality_matrix(features)
    
    flat_matrix = get_causality(
        causal_aggregator, matrix, features, mulcat_direction, mulcatbool, mulcat_init_weight, 
        mulcat_weights_shape, mulcat_trainable_threshold, mulcat_weights_mode
    )
    
    outputs = pass_to_classifier(causality_application, flat_features, flat_matrix, fc)
    
    model_resnet_18 = tf.keras.models.Model(inputs = inputs, outputs = outputs, name = "resnet18_ca_transfer")
    return model_resnet_18

def ResNet18CA_ensemble(
        resnet_no_ca: tf.keras.models.Model, 
        resnet_causal: tf.keras.models.Model, 
        num_classes: int, 
        input_shape: list[int]
    ) -> tf.keras.models.Model:
    """Creates an ensemble from causal_only and classic resnet models"""

    def freeze_layers(model):
        for layer in model.layers:
            layer.trainable = False

    freeze_layers(resnet_no_ca)
    freeze_layers(resnet_causal)

    inputs = layers.Input(shape=input_shape)

    cnn_outputs = resnet_no_ca(inputs)
    causal_outputs = resnet_causal(inputs)

    features_concatenated = layers.Concatenate(axis = 1)([cnn_outputs, causal_outputs])

    fc = Sequential([
        layers.Dense(4, activation="tanh"),
        layers.Dense(num_classes, activation="tanh")
    ])

    outputs = fc(features_concatenated)

    model_ensemble = tf.keras.models.Model(inputs = inputs, outputs = outputs, name="resnet18_ensemble")

    return model_ensemble