import tensorflow as tf


def load_mnist_data(normalize: bool = True):
    """Load MNIST dataset with optional normalization."""
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if normalize:
        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)
    return (x_train, y_train), (x_test, y_test)
