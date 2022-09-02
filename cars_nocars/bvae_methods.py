import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

 
class Sampling(layers.Layer):  
    """Sampler. Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def encoder(input_shape=[28,28], filters=[32,64], kernels=[3,3], strides=[2,2], hidden_dim=16, latent_dim=2):
    """Encoder with default parameters"""
    input = keras.Input(shape=(input_shape[0], input_shape[1], 1))
    x = input
    for i in range(len(filters)):
        x = layers.Conv2D(filters=filters[i], kernel_size=kernels[i], strides=strides[i], padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(hidden_dim)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)  #encoder encodes to mean and var=std^2. We sample random vector from these values
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    enc = keras.Model(input, [z_mean, z_log_var, z], name="encoder")
    return enc

def decoder(latent_dim=2, hidden=[7,7], filters=[64, 32], kernels=[3,3], strides=[2,2]):
    """Decoder with default parameters"""
    input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(hidden[0]*hidden[1]*filters[0])(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Reshape((hidden[0], hidden[1], filters[0]))(x)

    for i in range(len(filters)):
        x = layers.Conv2DTranspose(filters=filters[i], kernel_size=kernels[i], strides=strides[i], padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(1, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    output = layers.Activation('sigmoid')(x)

    dec = keras.Model(input, output, name="decoder")
    return dec

class bVAE(keras.Model):
    """Beta-Vae model """      
    def __init__(self, beta, encoder, decoder, **kwargs):
        super(bVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -self.beta*0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def tf_preprocessing(tensor_row,input_shape):
    """Image preprocessing. This has to be implemented by using tf functions only"""
    image = tensor_row['image']
    height, width = tf.unstack(tf.shape(image)[:2])
    box = tensor_row['bbox']
    scaled_box = box * [height, width, height, width]
    ymin, xmin, ymax, xmax = tf.unstack(tf.cast(scaled_box, tf.int32))
    box_width = xmax - xmin
    box_height = ymax - ymin
    image = tf.image.crop_to_bounding_box(image, ymin, xmin, box_height, box_width)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_with_pad(image, input_shape[0], input_shape[1])
    image = image/255.0
    return image

def tf_preprocessing_augmentation(tensor_row,input_shape):
    """Same as tf_preprocessing, but image flip is applied"""
    image = tensor_row['image']
    height, width = tf.unstack(tf.shape(image)[:2])
    box = tensor_row['bbox']
    scaled_box = box * [height, width, height, width]
    ymin, xmin, ymax, xmax = tf.unstack(tf.cast(scaled_box, tf.int32))
    box_width = xmax - xmin
    box_height = ymax - ymin
    image = tf.image.crop_to_bounding_box(image, ymin, xmin, box_height, box_width)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_with_pad(image, input_shape[0], input_shape[1])
    image = tf.image.flip_left_right(image)  #augmentation is here
    image = image/255.0
    return image

def tf_nocars(tensor_row,input_shape):
    return tf.random.uniform(shape=input_shape+[1],minval=0,maxval=1,dtype=tf.dtypes.float32,seed=11)