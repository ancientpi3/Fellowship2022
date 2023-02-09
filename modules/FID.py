import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

def scale_images(images, new_shape):
  images_list = list()
  for image in images:
    new_image = resize(image, new_shape, 0)
    images_list.append(new_image)
  return asarray(images_list)
 
def calculate_fid(model, images1, images2):
  images1 = images1.astype('float32')
  images2 = images2.astype('float32')

  images1 = scale_images(images1, (299,299,3))
  images2 = scale_images(images2, (299,299,3))

  images1 = preprocess_input(images1)
  images2 = preprocess_input(images2)
  
  act1 = model.predict(images1)
  act2 = model.predict(images2)

  mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
  mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

  ssdiff = numpy.sum((mu1 - mu2)**2.0)

  covmean = sqrtm(sigma1.dot(sigma2))

  if iscomplexobj(covmean):
    covmean = covmean.real

  fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid

def Inception():
  return InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
