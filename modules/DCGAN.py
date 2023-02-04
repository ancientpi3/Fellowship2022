import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2
import os


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import ReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.optimizers import RMSprop


def discriminator(in_shape=(64,64,3),dim=64):

  init = RandomNormal(stddev=0.02)
  model = Sequential()

  model.add(Conv2D(dim, (5,5), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
  model.add(BatchNormalization())
  model.add(LeakyReLU())

  model.add(Conv2D(2*dim, (5,5), strides=(2,2), padding='same', kernel_initializer=init))
  model.add(BatchNormalization())
  model.add(LeakyReLU())

  model.add(Conv2D(4*dim, (5,5), strides=(2,2), padding='same', kernel_initializer=init))
  model.add(BatchNormalization())
  model.add(LeakyReLU())

  model.add(Conv2D(8*dim, (5,5), strides=(2,2), padding='same', kernel_initializer=init))
  model.add(BatchNormalization())
  model.add(LeakyReLU())

  model.add(Flatten())
  model.add(Dense(1))

  opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss="binary_crossentropy", optimizer=opt)
  return model


def generator(latent_dim=128,dim=64):

  init = RandomNormal(stddev=0.02)
  model = Sequential()

  n_nodes = 4*4*8*dim
  model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
  model.add(Reshape((4, 4, 8*dim)))
  model.add(BatchNormalization())
  model.add(ReLU())

  model.add(Conv2DTranspose(4*dim, (5,5), strides=(2,2), padding='same', kernel_initializer=init))
  model.add(BatchNormalization())
  model.add(ReLU())

  model.add(Conv2DTranspose(2*dim, (5,5), strides=(2,2), padding='same', kernel_initializer=init))
  model.add(BatchNormalization())
  model.add(ReLU())

  model.add(Conv2DTranspose(dim, (5,5), strides=(2,2), padding='same', kernel_initializer=init))
  model.add(BatchNormalization())
  model.add(ReLU())

  model.add(Conv2DTranspose(3, (5,5), strides=(2,2),activation='tanh', padding='same', kernel_initializer=init))
  return model

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
		plt.imshow(examples[i, :, :, 0], cmap='gray')
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	plt.show()
	#plt.savefig(filename)
	plt.close()
 

def train(d_model,g_model,gan_model, x_train, verbose=1, print_every=2500,ITERS = 25000):
  latent_dim = 128
  batch_size = 64
  half_batch = 32

  for i in range(ITERS):
    d_model.trainable = True

    #print("training discriminator on real batch")
    real_batch = grab_real_batch(x_train, half_batch)
    d_model.train_on_batch(x=real_batch.reshape(half_batch,64,64,1), y = np.ones(half_batch))
    #print("training discriminator on fake batch")
    fake_batch = grab_fake_batch(g_model, half_batch)
    d_model.train_on_batch(x=fake_batch.reshape(half_batch,64,64,1), y = np.zeros(half_batch))

    d_model.trainable = False
    #print("training GAN on fake batch with real label to measure how well G fools D")
    latent_points = np.random.randn(latent_dim*batch_size).reshape(batch_size,latent_dim)
    gan_model.train_on_batch(x=latent_points, y=np.ones(batch_size))

    if((i)%print_every==print_every-1):
      save_plot(grab_fake_batch(g_model,50).reshape(50,64,64,3),epoch=i)
