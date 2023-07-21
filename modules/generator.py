from tensorflow import keras
from tensorflow.keras import layers


# define the standalone generator model
def define_generator(latent_dim):
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=0.02)
    # define model
    model = keras.Sequential()
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    model.add(layers.Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7, 7, 128)))
    # upsample to 14x14
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 28x28
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # output 28x28x1
    model.add(layers.Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
    return model