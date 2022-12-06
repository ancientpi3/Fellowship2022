import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
import cv2

#probabilistic generator parameters
mean = .0
scale = .2

#model meta parameters
lr = .0005

def show(image):
  cv2_imshow(zoom(image*255,8))
  
def zoom(img, zoom_factor=1.5):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor,interpolation=0)

def generated_image(mean, stddev, minval, maxval):
  return np.clip(np.random.normal(loc=mean, scale=stddev,size=(28,28)), minval, maxval)
  
def generate_mixed_data(mean=0, scale=.2, data_size = 100):
  data = []
  labels = []
  for _ in range(data_size):
    data.append(generated_image(mean,scale,0,1))
    labels.append(0)
    data.append(np.random.rand(28,28))
    labels.append(1)
  return (np.array(labels),np.array(data))

def generate_real_data(mean=0, scale=.2, data_size = 100):
  data = []
  labels = []
  for _ in range(data_size):
    data.append(generated_image(mean,scale,0,1))
    labels.append(0)
  return (np.array(labels),np.array(data))

def define_discriminator(in_shape=(28,28,1)):
  model = Sequential()
  model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape, activation='relu'))
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))

  opt = Adam(learning_rate=lr, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
 
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (2,2), strides=(4,4), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model
