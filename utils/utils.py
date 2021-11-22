import os
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow import keras, Tensor
from tensorflow.keras.applications import vgg19, vgg16
from tensorflow.python.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.losses import Reduction
from tqdm import tqdm

from models.perceptual_loss import PerceptualModelType


def get_img_dimensions(img_path: str) -> Tuple[int, int]:
  width, height = tf.keras.preprocessing.image.load_img(img_path).size

  return width, height


def preprocess_image(image_path: str,
                     img_nrows: int,
                     img_ncols: int,
                     model_type: PerceptualModelType = PerceptualModelType.VGG_19):
  """Util function to open, resize and format pictures into appropriate tensors"""

  if model_type == PerceptualModelType.VGG_19:
    preprocess_fn = vgg19.preprocess_input
  elif model_type == PerceptualModelType.VGG_16:
    preprocess_fn = vgg16.preprocess_input
  else:
    preprocess_fn = lambda x: x

  img = keras.preprocessing.image.load_img(image_path,
                                           target_size=(img_nrows, img_ncols))
  img = keras.preprocessing.image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = preprocess_fn(img)

  return tf.convert_to_tensor(img)


def preprocess_img_tensor(img_batch: Tensor,
                          img_nrows: int,
                          img_ncols: int,
                          model_type: PerceptualModelType = PerceptualModelType.VGG_19) -> Tensor:
  # tf.expand_dims(img_batch, a)
  # No need to expand dims as img_batch is a list of images
  img_batch = tf.image.resize(img_batch, (img_nrows, img_ncols))

  if model_type == PerceptualModelType.VGG_19:
    preprocess_fn = vgg19.preprocess_input
  elif model_type == PerceptualModelType.VGG_16:
    preprocess_fn = vgg16.preprocess_input
  else:
    preprocess_fn = lambda x: x

  return preprocess_fn(img_batch)


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


def content_loss(content_batch_feature_maps: List[Tensor],
                 stylised_batch_feature_maps: List[Tensor]) -> Tensor:
  loss = tf.zeros(shape=())
  content_weights = [1.0, 1.0, 1.0, 1.0]
  content_weights.reverse()
  for (target_content_representation,
       current_content_representation) in zip(content_batch_feature_maps,
                                              stylised_batch_feature_maps):
    loss += content_weights.pop() * tf.reduce_sum(
      tf.abs(target_content_representation - current_content_representation)
      # tf.square(target_content_representation - current_content_representation)
    )

  return loss


def style_loss(target_style_representation: List[Tensor],
               stylised_batch_feature_maps: List[Tensor],
               channels: int = 3) -> Tensor:
  loss = tf.zeros(shape=())
  current_style_representation = [batch_gram_matrix(x, normalise=True)
                                  for x in stylised_batch_feature_maps]
  # layer_weights = [0.1, 0.2, 0.2, 0.2, 0.4]
  layer_weights = [1., 1., 1.2, 1.3, 1.4]
  layer_weights.reverse()
  for gram_gt, gram_hat in zip(target_style_representation,
                               current_style_representation):
    # loss += layer_weights.pop() * MeanSquaredError()(gram_gt, gram_hat)
    loss += layer_weights.pop() * MeanAbsoluteError()(gram_gt, gram_hat)
    # S, C = gram_gt, gram_hat
    # size = gram_gt.shape[1] * gram_gt.shape[2]
    # loss += layer_weights.pop() * tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

  loss /= len(target_style_representation)

  return loss


def total_variation_loss(stylised_batch):
  # Small change 👇 to reduce_sum instead of mean
  return tf.reduce_sum(tf.image.total_variation(stylised_batch))


def compute_losses(stylised_batch,
                   target_style_representation,
                   content_feature_maps,
                   stylised_content_feature_maps,
                   stylised_style_feature_maps) -> Tuple[Tensor, Tensor, Tensor]:
  batch_content_loss = content_loss(content_batch_feature_maps=content_feature_maps,
                                    stylised_batch_feature_maps=stylised_content_feature_maps)
  batch_style_loss = style_loss(target_style_representation=target_style_representation,
                                stylised_batch_feature_maps=stylised_style_feature_maps)
  batch_total_variation_loss = total_variation_loss(stylised_batch=stylised_batch)

  return batch_content_loss, batch_style_loss, batch_total_variation_loss