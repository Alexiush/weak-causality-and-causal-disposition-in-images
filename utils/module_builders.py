"""Module containing helper functions to construct model parts"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from causal_aggregation.causality_weigher import CausalityWeigher
from causal_aggregation.causality_statistics_extractor import CausalityStatisticsExtractor

def get_classifier(classifier: str, num_classes: int, num_features: int, dropout: float = 0.) -> Sequential:
    """
    Gets classifier specified by keyword.

    Arguments:
    classifier (str): keyword
    num_classes (int): number of classes to predict
    num_features (int): number of features to be passed
    dropout (float): dropout rate

    Returns: final classifier as keras.Sequential model

    Raises: ValueError if wrong keyword was passed

    """

    if classifier == "classic":
        fc = Sequential([
            layers.Dropout(dropout),
            layers.Dense(num_classes, name="outputs", activation='sigmoid'),
        ])
    elif classifier == "mlp":
        fc = Sequential([
            layers.Dropout(dropout),
            layers.Dense(num_features),
            layers.Dense(num_features // 2),
            layers.Dense(num_classes, name="outputs", activation='sigmoid')
        ])
    else:
        raise ValueError("Unknown final classifier")
        
    return fc

def get_aggregator(aggregation: str) -> tf.keras.layers.Layer:
    """
    Gets aggregation function specified by keyword.

    Arguments:
    aggregation (str): keyword

    Returns: Aggregation layer

    Raises: ValueError if wrong keyword was passed

    """

    if aggregation == "avg":
        aggregation_layer = layers.GlobalAveragePooling2D()
    elif aggregation == "max":
        aggregation_layer = layers.GlobalMaxPooling2D()
    elif aggregation == "flatten":
        aggregation_layer = layers.Flatten()
    else:
        raise ValueError("Unknown aggregation operator")
    
    return aggregation_layer

def get_causality(
        causality: str, 
        causality_matrix: tf.Tensor, 
        features: tf.Tensor, 
        mulcat_direction: str, 
        mulcatbool: bool, 
        mulcat_init_weight: float,
        mulcat_weights_shape: str, 
        mulcat_trainable_threshold: bool, 
        mulcat_weights_mode: str
    ) -> tf.Tensor:
    
    """
    Gets causality processing function specified by keyword.

    Arguments:
    causality (str): keyword, 
    causality_matrix (tf.Tensor): inferred causality matrix, 
    features (tf.Tesor): features extracted from the input, 
    mulcat_direction (str): mulcat computation direction, 
    mulcatbool (bool): should mulcat be used as a binary function,
    mulcat_init_weight (float): initial value for threshold computation,
    mulcat_weights_shape (str): should mulcat threshold be "global" or "local" (per-matrix or per-value), 
    mulcat_trainable_threshold (bool): should threshold be trained, 
    mulcat_weights_mode (str): which threshold should be applied: "variable" for independent variable or "function" for linear function from the input

    Returns: Processed causality matrix

    Raises: ValueError if wrong keyword was passed

    """

    if causality == "concat":
        causality_matrix = layers.Flatten()(causality_matrix[0])
    elif causality == "mulcat":
        causality_matrix = CausalityWeigher(
            mulcatbool = mulcatbool, direction = mulcat_direction, 
            init_weight = mulcat_init_weight, 
            weights_shape=mulcat_weights_shape, trainable_threshold=mulcat_trainable_threshold, 
            weights_mode=mulcat_weights_mode
        )([causality_matrix, features])
    elif causality == "mulcatbool":
        causality_matrix = CausalityWeigher(
            mulcatbool = True, direction = mulcat_direction, 
            init_weight = mulcatbool, 
            weights_shape=mulcat_weights_shape, trainable_threshold=mulcat_trainable_threshold, 
            weights_mode=mulcat_weights_mode
        )([causality_matrix, features])
    elif causality == "statcat":
        causality_matrix = CausalityStatisticsExtractor()(causality_matrix[0])
    else:
        raise ValueError("Unknown causality application method")
    
    return causality_matrix

def pass_to_classifier(
        causality_application: str, 
        features: tf.Tensor, 
        causality_matrix: tf.Tensor, 
        classifier: tf.keras.models.Model
    ) -> tf.Tensor:
    """
    Combines spatial and causal data in a way specified by the keyword and passes it to classifier

    Arguments:
    causality_application (str): keyword, 
    flat_features (tf.Tensor): aggregated extracted features, 
    flat_matrix (tf.Tensor): processed causality matrixs, 
    classifier (tf.keras.models.Model): classifier

    Returns: classifier's predictions

    Raises: ValueError if wrong keyword was passed

    """

    if causality_application == "causal_only":
        outputs = classifier(causality_matrix)
    elif causality_application == "concatenated":
        features_concatenated = layers.Concatenate(axis = 1)([features, causality_matrix])
        outputs = classifier(features_concatenated)
    else:
        raise ValueError("Unknown causality application method")
    
    return outputs