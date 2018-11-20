import numpy as np
import tensorflow as tf


def sample_bernoulli(probs, straight_through=False):
    """ Samples from the bernoulli distribution.

    Args:
        probs: probability distribution to sample from.
    Returns:
        sample: sample from probability distribution
    """
    sample = tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))
    if straight_through:
        sample = tf.stop_gradient(sample - probs) + probs
    return sample
