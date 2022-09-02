#input parameters
input_shape = [64,96]
batch_size = 128
beta = 1.5
filters = [64,64,64,64]
kernels=[3,5,7,9] 
strides=[2,2,2,2]
hidden_dim = 512
latent_dim = 48
hidden = [4,6]
epochs = 150



import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import bvae_methods as bm

#download dataset
cars0_train, cars0_test = tfds.load('cars196', split=['train+test[:90%]','test[90%:]'], as_supervised=False, shuffle_files=False)

#preprocessing and augmentation
cars = cars0_train.map(lambda x: bm.tf_preprocessing(x,input_shape))
cars_au = cars0_train.map(lambda x: bm.tf_preprocessing_augmentation(x,input_shape))
cars = cars.concatenate(cars_au)
#add images without cars
nocars = cars.take(3000).map(lambda x: bm.tf_nocars(x,input_shape))
cars = cars.concatenate(nocars)

length = len(tfds.as_numpy(cars))
cars = cars.cache().shuffle(length+1).batch(batch_size).prefetch(buffer_size=3)  #model.fit wont work without this step

#creating model
enc = bm.encoder(input_shape=input_shape, filters=filters, kernels=kernels, strides=strides, hidden_dim=hidden_dim, latent_dim=latent_dim)
dec = bm.decoder(latent_dim=latent_dim, hidden=hidden, filters=filters[-1::-1], kernels=kernels[-1::-1], strides=strides[-1::-1])

vae = bm.bVAE(beta, enc, dec)
vae.compile(optimizer=keras.optimizers.Adam())  

checkpoint_path = "bvae_latest_checkpoint/latest_checkpoint"
callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)  #Creates callback that saves the model's weights
history = vae.fit(cars, epochs=epochs, batch_size=batch_size, callbacks=[callback]) 

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