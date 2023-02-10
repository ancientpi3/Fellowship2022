import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2
import os
from PIL import Image
from keras.initializers import RandomNormal

def folder_to_numpy(folder_path,take=400):
    images = []

	
    #for filename in os.listdir(folder_path):
    for i in range(take):
	    filename = random.choice(os.listdir(folder_path))
        if filename.endswith(".jpg"):
            image = Image.open(os.path.join(folder_path, filename))
            image=image.resize((64,64))
            images.append(np.array(image))

    images = np.array(images)
    images = 2*(images/255)-1
    return images

def discriminator(in_shape=(64,64,3),dim=64):
  init = RandomNormal(stddev=0.02)
  discriminator_input = keras.Input(shape=(64,64,3),name="d_input")
  
  x = keras.layers.Conv2D(64,(5,5),strides = (2,2),padding = "same",kernel_initializer=init, activation="leaky_relu")(discriminator_input)
  x = keras.layers.BatchNormalization(momentum=.9)(x,training=False)

  x = keras.layers.Conv2D(128,(5,5),strides = (2,2),padding = "same",kernel_initializer=init, activation="leaky_relu")(x)
  x = keras.layers.BatchNormalization(momentum=.9)(x,training=False)

  x = keras.layers.Conv2D(256,(5,5),strides = (2,2),padding = "same",kernel_initializer=init, activation="leaky_relu")(x)
  x = keras.layers.BatchNormalization(momentum=.9)(x,training=False)

  x = keras.layers.Conv2D(512,(5,5),strides = (2,2),padding = "same",kernel_initializer=init, activation="leaky_relu")(x)
  x = keras.layers.BatchNormalization(momentum=.9)(x,training=False)
  x = keras.layers.Flatten()(x)
  
  discriminator_output = keras.layers.Dense(1, activation='sigmoid')(x)


  model = keras.Model(discriminator_input, discriminator_output,name="discriminator")
  opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model


def generator(latent_dim=128,dim=64):

  generator_input = keras.Input(shape=(latent_dim),name="g_input")
  n_nodes = 4*4*8*64

  x = keras.layers.Dense(n_nodes)(generator_input)
  x = keras.layers.Reshape((4,4,8*64))(x)
  x = keras.layers.BatchNormalization(momentum=.9)(x,training=True)

  x = keras.layers.Conv2DTranspose(256,(5,5),strides=(2,2),padding='same',activation="relu")(x)
  x = keras.layers.BatchNormalization(momentum=.9)(x,training=True)

  x = keras.layers.Conv2DTranspose(128,(5,5),strides=(2,2),padding='same',activation="relu")(x)
  x = keras.layers.BatchNormalization(momentum=.9)(x,training=True)

  x = keras.layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',activation="relu")(x)
  x = keras.layers.BatchNormalization(momentum=.9)(x,training=True)

  generator_output = keras.layers.Conv2DTranspose(3,(5,5),strides=(2,2),padding='same',activation="tanh")(x)


  return keras.Model(generator_input, generator_output,name="generator")

def GAN(d_model,g_model):
	d_model.trainable = False
	model = keras.models.Sequential()
	model.add(g_model)
	model.add(d_model)
 
	opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def grab_real_batch(x_train, batch_size):
	ix = np.random.randint(0, x_train.shape[0], batch_size)
	X = x_train[ix]
	#y = np.ones((batch_size, 1))
	return X

def grab_fake_batch(g_model,batch_size, latent_dim=128):
  latent_points = np.random.randn(latent_dim*batch_size).reshape(batch_size,latent_dim)
  predictions = g_model.predict(latent_points,verbose=0)
  return predictions

def save_plot(examples, epoch, n=5):
	for i in range(25):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow((examples[i, :, :]+1)/2)
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	plt.show()
	#plt.savefig(filename)
	plt.close()
 
def train(d_model,g_model,gan_model, x_train, epochs=400, batches_per_epoch=64, verbose=1, print_every=20):
  latent_dim = g_model.input.shape[1]
  batch_size = 24
  half_batch = int(batch_size/2)
  batches_per_epoch = 64
  save_plot(grab_fake_batch(g_model,50).reshape(50,64,64,3),epoch=0)
  for epoch in range(epochs):
    print("Epoch: ",epoch)
    for _ in range(batches_per_epoch):
      d_model.trainable = True

      #print("training discriminator on real batch")
      real_batch = grab_real_batch(x_train, half_batch)
      d_model.train_on_batch(x=real_batch.reshape(half_batch,64,64,3), y = np.ones(half_batch))
      #print("training discriminator on fake batch")
      fake_batch = grab_fake_batch(g_model, half_batch)
      d_model.train_on_batch(x=fake_batch.reshape(half_batch,64,64,3), y = np.zeros(half_batch))

      d_model.trainable = False
      #print("training GAN on fake batch with real label to measure how well G fools D")
      latent_points = np.random.randn(latent_dim*batch_size).reshape(batch_size,latent_dim)
      gan_model.train_on_batch(x=latent_points, y=np.ones(batch_size))



    if(epoch%print_every==print_every-1):
      save_plot(grab_fake_batch(g_model,50).reshape(50,64,64,3),epoch=epoch)
