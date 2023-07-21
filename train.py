import tensorflow as tf
from tensorflow import keras
from modules.critic import define_critic
from modules.generator import define_generator
from model import GAN
import matplotlib.pyplot as plt
from modules.utils import wasserstein_loss

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
train_images = X_train.reshape(-1, 28, 28, 1)
train_images = (train_images - 127.5) / 127.5

# limiting training set to user input
number = int(input('What number do you want to generate(from 0-9): '))
selected = y_train == number
train_ds = train_images[selected]
train_ds = tf.data.Dataset.from_tensor_slices(train_ds).batch(64)

latent_dim = 100
n_samples = 16

generator = define_generator(latent_dim)
critic = define_critic()

epochs = 50
seed = tf.random.normal([16, 100])

# callback function for monitoring
class Generate(keras.callbacks.Callback):
    def __init__(self):
        super(Generate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        prediction = generator(seed)

        plt.figure(figsize=(4, 4))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(prediction[i, :, :, 0], cmap='gray')
            plt.axis('off')

        plt.savefig(f'image at epoch {epoch + 1}.png')
        plt.show


WGAN = GAN(critic=critic, generator=generator)

WGAN.compile(c_optimizer=keras.optimizers.RMSprop(learning_rate=0.00005),
             g_optimizer=keras.optimizers.RMSprop(learning_rate=0.00005),
             loss_fn=wasserstein_loss)

WGAN.fit(train_ds, epochs=50, callbacks=[Generate()])

