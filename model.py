import tensorflow as tf
from tensorflow import keras
from modules.utils import generate_fake


class GAN(keras.Model):
    def __init__(self, critic, generator):
        super(GAN, self).__init__()
        self.critic = critic
        self.generator = generator

    def compile(self, c_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def critic_loss(self, real, fake):
        real_loss = self.loss_fn(-(tf.ones_like(real)), real)
        fake_loss = self.loss_fn(tf.ones_like(fake), fake)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake):
        fake_loss = self.loss_fn(-(tf.ones_like(fake)), fake)
        return fake_loss

    def train_step(self, batch, n_critic=5):
        # tr_image, tr_label = batch
        noise = generate_fake(64, 100)
        with tf.GradientTape() as gen_tape, tf.GradientTape(persistent=True) as cr_tape:
            # update the critic more.
            for i in range(n_critic):
                generated = self.generator(noise, training=True)
                loss1 = self.critic(batch[0:32, :, :, :], training=True)
                loss2 = self.critic(generated[0:32, :, :, :], training=True)
                full_loss = self.critic_loss(loss1, loss2)

                gr = cr_tape.gradient(full_loss, self.critic.trainable_variables)
                self.c_optimizer.apply_gradients(zip(gr, self.critic.trainable_variables))

            generated_image = self.generator(noise, training=True)

            real = self.critic(batch, training=True)
            fake = self.critic(generated_image, training=True)

            gen_loss = self.generator_loss(fake)
            cr_loss = self.critic_loss(real, fake)

        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        cr_grad = cr_tape.gradient(cr_loss, self.critic.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
        self.c_optimizer.apply_gradients(zip(cr_grad, self.critic.trainable_variables))
        return {"cr_loss": cr_loss, "g_loss": gen_loss}