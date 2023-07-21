import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

# clip model weights to a given hypercube
class ClipConstraint(keras.constraints.Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

# define the standalone critic model
def define_critic(in_shape=(28,28,1)):
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=0.02)
    # weight constraint
    const = ClipConstraint(0.01)
    # define model
    model = keras.Sequential()
    # downsample to 14x14
    model.add(layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
    model.add(layers.BatchNormalization()) #kernel_constraint=const
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample to 7x7
    model.add(layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
    model.add(layers.BatchNormalization()) #kernel_constraint=const
    model.add(layers.LeakyReLU(alpha=0.2))
    # scoring, linear activation
    model.add(layers.Flatten())
    model.add(layers.Dense(1)) #kernel_constraint=const
    return model

model = define_critic()