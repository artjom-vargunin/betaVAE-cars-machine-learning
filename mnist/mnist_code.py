#input parameters
input_shape=[28,28]
batch_size = 128
beta = 1.5
filters=[64,64]
kernels=[3,5]
strides=[2,2]
hidden_dim=300
latent_dim=16
hidden = [7,7]
epochs = 30


import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import bvae_methods as bm

#download dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
digits = np.expand_dims(x_train, -1).astype("float32") / 255

#creating model
enc = bm.encoder(input_shape=input_shape, filters=filters, kernels=kernels, strides=strides, hidden_dim=hidden_dim, latent_dim=latent_dim)
dec = bm.decoder(latent_dim=latent_dim, hidden=hidden, filters=filters[-1::-1], kernels=kernels[-1::-1], strides=strides[-1::-1])

vae = bm.bVAE(beta, enc, dec)
vae.compile(optimizer=keras.optimizers.Adam())  

checkpoint_path = "bvae_latest_checkpoint/latest_checkpoint"
callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)  #Creates callback that saves the model's weights
history = vae.fit(digits, epochs=epochs, batch_size=batch_size, callbacks=[callback]) 

#save output
output = []
output += [f'input_shape = {input_shape}']
output += [f'batch_size = {batch_size:.0f}']
output += [f'beta = {beta:.3f}']
output += [f'filters = {filters}']
output += [f'kernels = {kernels}']
output += [f'strides = {strides}']
output += [f'hidden_dim = {hidden_dim:.0f}']
output += [f'latent_dim = {latent_dim:.0f}']
output += [f'hidden = {hidden}']
output += [f'epochs = {epochs:.0f}']
output += ['\nloss  reconstruction_loss  kl_loss']
for i in range(len(history.history['loss'])):
    output += [f"{history.history['loss'][i]:.3f} {history.history['reconstruction_loss'][i]:.3f} {history.history['kl_loss'][i]:.3f}"]

file = open("bvae_output.output", 'w')
for o in output: file.write(o+'\n')  #we remove comma after last json object in collection
file.close()