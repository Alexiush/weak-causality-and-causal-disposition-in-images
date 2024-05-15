"""Module containing helper functions for plotting"""
import matplotlib.pyplot as plt
import tensorflow as tf

def filter_zeros(values: list[float], epsilon: float = 1e-9) -> list[float]:
    return list(filter(lambda v: v - 0 > epsilon, values))

def plot_history(history: tf.keras.callbacks.History):
    with_parameter = "lehmer_numerator" in history.history.keys()
    metrics_count = 3 if with_parameter else 2
    
    plt.figure(figsize=(8, 8))
    
    acc = filter_zeros(history.history['accuracy'])
    val_acc = filter_zeros(history.history['val_accuracy'])

    plt.subplot(1, metrics_count, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    loss = filter_zeros(history.history['loss'])
    val_loss = filter_zeros(history.history['val_loss'])
    
    plt.subplot(1, metrics_count, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    if with_parameter:
        l_num = filter_zeros(history.history['lehmer_numerator'])
        l_den = filter_zeros(history.history['lehmer_denominator'])

        plt.figure(figsize=(8, 8))
        plt.subplot(1, metrics_count, 3)
        plt.plot(l_num, label='Lehmer numerator')
        plt.plot(l_den, label='Lehmer denominator')
        plt.legend(loc='lower right')
        plt.title('P values')
        
    plt.show()