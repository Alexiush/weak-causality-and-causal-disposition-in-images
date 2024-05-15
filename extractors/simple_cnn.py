"""Module containing functions to build simple cnn feature extractor"""

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def simple_cnn_conv() -> Sequential:
    return Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
    ])