"""Module implementing utils for training configurations storing"""
from dataclasses import dataclass
import tensorflow as tf

@dataclass
class DatasetConfig():
    """Class storing the processed dataset configuration data"""
    input_shape: list[int]
    num_classes: int
    datasets: list[tf.data.Dataset]
    steps_per_epoch: int
    validation_steps: int
        