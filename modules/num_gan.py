import tensorflow as tf 
import time
import os
import matplotlib.pyplot as plt
import imageio
import glob
import numpy as np


class NumDiscriminator(tf.keras.models.Model):
    """
    A class for defining a generic discriminator for numeric data
    """

    def __init__(self, num_nodes, num_layers):
        super().__init__(name='NumDiscriminator')
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.layers_ = [tf.keras.layers.Dense(units=num_nodes, activation=tf.nn.leaky_relu) \
                        for _ in range(num_layers-1)]
        self.layers_.append(tf.keras.layers.Dense(units=1))

    def call(self, x):
        for layer in self.layers_:
            x = layer(x)
        return x

    def save(self, path):
        super().save_weights(path)

    def load(self, path):
        super().load_weights(path)


class NumGenereator(tf.keras.models.Model):
    """
    A class for defining a generic genereator for numeric data
    """

    def __init__(self, num_nodes, num_layers, shape):
        super().__init__(name='NumGenerator')
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.layers_ = [tf.keras.layers.Dense(units=num_nodes, activation=tf.nn.leaky_relu) \
                        for _ in range(num_layers-1)]
        self.layers_.append(tf.keras.layers.Dense(units=shape))

    def call(self, x):
        for layer in self.layers_:
            x = layer(x)
        return x

    def save(self, path):
        super().save_weights(path)

    def load(self, path):
        super().load_weights(path)


class NumGAN:
    """
    A twin-model class for GAN
    """
    def __init__(self, num_nodes, num_layers, dim, noise_dim, name='NumGAN', savedir='.'):
        self.dim = dim
        self.noise_dim = noise_dim
        self.name = name
        self.discriminator = NumDiscriminator(num_nodes, num_layers)
        self.generator = NumGenereator(num_nodes, num_layers, dim)
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits
        if savedir != '':
            savedir += '/'
        self.savedir = savedir + '/' + self.name
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)
    
    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, samples, batch_size):
        noise = tf.random.normal([batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_samples = self.generator(noise, training=True)

            real_output = self.discriminator(samples, training=True)
            fake_output = self.discriminator(generated_samples, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(logits=real_output,labels=tf.ones_like(real_output))
        fake_loss = self.cross_entropy(logits=fake_output,labels=tf.zeros_like(fake_output))
        total_loss = real_loss + fake_loss
        return tf.reduce_mean(total_loss) 
    

    def generator_loss(self, fake_output):
        return tf.reduce_mean(self.cross_entropy(logits=fake_output,labels=tf.ones_like(fake_output)))

    
    
    def train(self, dataset, epochs, gap=100, learning_rate=1e-4):
        self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate)

        for epoch in range(epochs):
            start = time.time()

            for batch in dataset:
                gen_loss, disc_loss = self.train_step(batch, batch.shape[0])

            # check the model every few epochs
            if (epoch + 1) % gap == 0:
                print ('\rTime for epoch {} is {:.3f} sec, gen loss = {}, disc loss = {}'\
                    .format(epoch + 1, time.time()-start, gen_loss.numpy(), disc_loss.numpy()), end="")
                self.generate_and_save_images(batch, epoch=epoch + 1)

        self.make_gif()
        self.save()
        

    def save(self):
        self.generator.save(self.savedir + '/generator')
        self.discriminator.save(self.savedir + '/discriminator')

    
    def load(self, path=None):
        if path is None:
            path = self.savedir
        try:
            self.generator.load(path + '/generator')
            self.discriminator.load(path + '/discriminator')
        except OSError as error:
            print(error.errno)


    def summary(self):
        self.generator.summary()
        self.discriminator.summary()


    def generate_and_save_images(self, real_samples, epoch=None):
        fig = plt.figure(figsize=(8, 8))
        if self.dim >= 3:
            ax =  fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        num_samples = real_samples.shape[0]
        noise = tf.random.normal([num_samples, self.noise_dim])
        samples = tf.reshape(self.generator(noise, training=False), shape=(num_samples, self.dim))
        #print(samples)
        if self.dim >=3:
            ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], label='generated')
            ax.scatter(real_samples[:, 0], real_samples[:, 1], real_samples[:, 2], label='true')
        else:
            ax.scatter(samples[:, 0], samples[:, 1], label='generated')
            ax.scatter(real_samples[:, 0], real_samples[:, 1], label='true')
        if epoch is not None:
            ax.set_title('generated samples at epoch #{}'.format(epoch))
        else:
            ax.set_title('generated samples')
        plt.legend()
        
        file_dir = '{}/images'.format(self.savedir)
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        plt.savefig(file_dir + '/samples_{}.png'.format(epoch))
        plt.close()


    def make_gif(self):
        file_dir = '{}/images'.format(self.savedir)
        anim_file = '{}/train.gif'.format(file_dir)

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob('{}/samples*.png'.format(file_dir))
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)
        for filename in filenames:
            os.remove(filename)



class NumCompleter:
    """
    A class for completing numeric data
    """
    def __init__(self, gan, mask, lam=0.1, name='Numcompleter', savedir='.'):
        self.gan = gan
        self.mask = mask
        self.lam = lam
        self.name = name
        if savedir != '':
            savedir += '/'
        self.savedir = savedir + '/' + self.name
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)


    def contextual_loss(self, samples, generated_samples):
        masked_samples = tf.math.multiply(self.mask, samples)
        masked_output = tf.math.multiply(self.mask, generated_samples)
        return tf.keras.losses.mean_absolute_error(masked_samples, masked_output)


    def perceptual_loss(self, fake_output):
        return self.gan.generator_loss(fake_output)
    
    def complete_loss(self, samples, noise):
        generated_samples = self.gan.generator(self.noise, training=True)
        fake_output = self.gan.discriminator(generated_samples, training=True)

        perc_loss = self.perceptual_loss(fake_output)
        cont_loss = self.contextual_loss(samples, generated_samples)
        return tf.reduce_mean(cont_loss + self.lam * perc_loss)

    @tf.function
    def train_step(self, samples):
        with tf.GradientTape() as tape:
            tape.watch(self.noise)
            total_loss = self.complete_loss(samples, self.noise)
            grads = tape.gradient(total_loss, [self.noise])
        

        self.optimizer.apply_gradients(zip(grads, [self.noise]))
        
        
        return total_loss


    def train(self, data, epochs, gap=100, learning_rate=1e-4):
        #momentum, lr = 0.9, 1e-4
        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,\
        #                        decay_steps=10000, decay_rate=0.9)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        if not hasattr(self, 'noise'):
            self.noise = tf.Variable(tf.random.normal([data.shape[0], self.gan.noise_dim]), trainable=True)
        #"""
        v = 0
        for epoch in range(epochs):
            start = time.time()
            
            total_loss = self.train_step(data)
            #v_prev = np.copy(v)
            #v = momentum*v - lr*grads[0]
            #self.noise += -momentum * v_prev + (1 + momentum)*v
            #self.noise = tf.clip_by_value(self.noise, -1, 1)
            # Save the model every few epochs
            if (epoch + 1) % gap == 0:
                print ('Time for epoch {} is {:.3f} sec, total loss = {:.3f}'\
                      .format(epoch + 1, time.time()-start, total_loss.numpy()))
                #if np.isnan(total_loss.numpy()):
                #    print(self.noise, '***********\n', self.complete_loss(data, self.noise))
                self.save()
        #"""
        loss = lambda: self.complete_loss(data, self.noise)
        #self.optimizer.minimize(loss, var_list=[self.noise])


    def complete(self, incomplete_data):
        given = tf.math.multiply(self.mask, incomplete_data)
        rest = tf.math.multiply(1. - self.mask, self.gan.generator(self.noise, training=False))
        return  given + rest 


    def save(self):
        np.save(self.savedir + '/optimal_noise.npy', self.noise.numpy())
            

    def load(self):
        self.noise = tf.Variable(np.load(self.savedir + '/optimal_noise.npy'), trainable=True)
    


