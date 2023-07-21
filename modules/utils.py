import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np

def generate_fake(n_samples, latent_dim):
    noise = tf.random.normal([n_samples, latent_dim])
    return noise

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

