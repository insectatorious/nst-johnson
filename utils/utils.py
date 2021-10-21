import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras, Tensor
from tensorflow.keras.applications import vgg19
from tqdm import tqdm


def get_img_dimensions(img_path: str) -> Tuple[int, int]:
  width, height = tf.keras.preprocessing.image.load_img(img_path).size

  return width, height


def preprocess_image(image_path, img_nrows, img_ncols):
  # Util function to open, resize and format pictures into appropriate tensors
  img = keras.preprocessing.image.load_img(image_path,
                                           target_size=(img_nrows, img_ncols))
  img = keras.preprocessing.image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = vgg19.preprocess_input(img)

  return tf.convert_to_tensor(img)


def preprocess_img_tensor(img_batch, img_nrows, img_ncols) -> Tensor:
  # tf.expand_dims(img_batch, a)
  # No need to expand dims as img_batch is a list of images
  img_batch = tf.image.resize(img_batch, (img_nrows, img_ncols))

  return vgg19.preprocess_input(img_batch)


def deprocess_image(x, img_nrows, img_ncols, add_offset: bool = True):
  # Util function to convert a tensor into a valid image
  if isinstance(x, tf.Tensor):
    x = tf.reshape(x, (-1, img_nrows, img_ncols, 3))

    if add_offset:
      x = x + tf.constant([103.939, 116.779, 123.68])

    # 'BGR'->'RGB'
    x = x[:, :, :, ::-1]

    x = tf.clip_by_value(x, 0, 255)
    x = tf.cast(x, dtype=tf.uint8)

  else:
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')

  return x


def get_pbar(total: int = 100):
  return tqdm(total=total, leave=False, ascii=True)

"""## Compute the style transfer loss

First, we need to define 4 utility functions:

- `gram_matrix` (used to compute the style loss)
- The `style_loss` function, which keeps the generated image close to the local textures of the style reference image
- The `content_loss` function, which keeps the high-level representation of the
generated image close to that of the base image
- The `total_variation_loss` function, a regularization loss which keeps the generated image locally-coherent
"""
# The gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
  x = tf.transpose(x, (2, 0, 1))
  features = tf.reshape(x, (tf.shape(x)[0], -1))
  gram = tf.matmul(features, tf.transpose(features))

  return gram


def batch_gram_matrix(x, normalise: bool = True):
  x = tf.transpose(x, (0, 3, 1, 2))
  (b, h, w, ch) = x.shape.as_list()
  features = tf.reshape(x, (b, ch, h * w))
  # gram = tf.matmul(features, tf.transpose(features))
  gram = tf.linalg.matmul(a=features, b=features, transpose_b=True)
  gram = tf.reduce_mean(gram, axis=0)
  gram = tf.expand_dims(gram, axis=0)
  gram = gram / (ch * h * w) if normalise else gram

  return gram


# The "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image

def style_loss(style, combination, img_ncols, img_nrows):
  S = gram_matrix(style)
  C = gram_matrix(combination)
  channels = 3
  size = img_nrows * img_ncols

  return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


# An auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image

def content_loss(base, combination):
  return tf.reduce_sum(tf.square(combination - base))

# The 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x, img_ncols, img_nrows, l2: bool = True):
  l_loss = tf.math.square if l2 else tf.math.abs
  a = l_loss(
    x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
  b = l_loss(
    x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])

  return tf.reduce_sum(tf.pow(a + b, 1.25))


def get_training_strategy(config):
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
  except ValueError:  # If TPU not found
    tpu = None
  if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    config["tpu"] = True
  else:
    strategy = tf.distribute.get_strategy()
    print("Not running on TPU...")
    config["tpu"] = False
  return strategy