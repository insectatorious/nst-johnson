from typing import List, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import (
  Conv2D,
  UpSampling2D,
  BatchNormalization,
  ZeroPadding2D,
  SpatialDropout2D,
  ReLU,
  LeakyReLU,
  Add
)
from tensorflow_addons.layers import InstanceNormalization


class ResidualBlock(tf.keras.layers.Layer):

  def __init__(self,
               num_of_filters: int = 16,
               norm_type: str = "batch",
               final_relu: bool = True,
               **kwargs):
    # TODO: Replace this ugly hack 👇 with an Enum
    if norm_type not in ["batch", "instance"]:
      raise AttributeError(f"Expected 'norm_type' to be one of 'batch' or "
                           f"'instance'. Got '{norm_type}")

    super(ResidualBlock, self).__init__(**kwargs)
    self.num_of_filters = num_of_filters
    self.norm_type = norm_type
    self.final_relu = final_relu
    self.conv_1 = None
    self.conv_2 = None
    self.relu_1 = None
    self.relu_2 = None
    self.norm_1 = None
    self.norm_2 = None
    if self.final_relu:
      self.add_1 = None

  def build(self, input_shape: List) -> None:
    self.conv_1 = Conv2D(filters=self.num_of_filters,
                         kernel_size=(1, 1),
                         input_shape=input_shape)

    if self.norm_type == "batch":
      self.norm_1 = BatchNormalization()
    else:
      self.norm_1 = InstanceNormalization()
    self.relu_1 = LeakyReLU()
    self.conv_2 = Conv2D(filters=self.num_of_filters,
                         kernel_size=3,
                         padding="same")
    if self.norm_type == "batch":
      self.norm_2 = BatchNormalization()
    else:
      self.norm_2 = InstanceNormalization()
    self.add_1 = Add()
    if self.final_relu:
      self.relu_2 = LeakyReLU()

  def call(self, inputs: Tensor) -> Tensor:
    layer = self.conv_1(inputs)
    layer = self.norm_1(layer)
    layer = self.relu_1(layer)
    layer = self.conv_2(layer)
    layer = self.norm_2(layer)
    layer = self.add_1([layer, inputs])
    if self.final_relu:
      layer = self.relu_2(layer)

    return layer

  def get_config(self) -> Dict:
    config = super(ResidualBlock, self).get_config()
    config.update({"num_of_filters": self.num_of_filters,
                   "norm_type": self.norm_type,
                   "final_relu": self.final_relu})

    return config


def get_transformer(num_of_channels=None,
                    kernel_sizes=None,
                    stride_sizes=None,
                    pad_inputs: bool = True,
                    dropout_rate: float = 0.1) -> tf.keras.Model:
  if num_of_channels is None:
    num_of_channels = [3, 32, 64, 128]
  if kernel_sizes is None:
    kernel_sizes = [9, 3, 3]
  if stride_sizes is None:
    stride_sizes = [1, 2, 2]
  relu = LeakyReLU()
  input_layer = tf.keras.Input(shape=(None, None, 3), name="input_img")
  if pad_inputs:
    x = tf.pad(input_layer, [[0, 0], [3, 3], [3, 3], [0, 0]], "SYMMETRIC")
  else:
    x = input_layer
  x = Conv2D(num_of_channels[1],
             kernel_size=kernel_sizes[0],
             padding="same",
             strides=stride_sizes[0])(x)
  x = SpatialDropout2D(dropout_rate)(x)
  x = InstanceNormalization()(x)
  # x = LeakyReLU()(x)
  x = relu(x)
  x = Conv2D(num_of_channels[2],
             kernel_size=kernel_sizes[1],
             padding="valid" if pad_inputs else "same",
             strides=stride_sizes[1])(x)
  x = SpatialDropout2D(dropout_rate)(x)
  x = InstanceNormalization()(x)
  # x = LeakyReLU()(x)
  x = relu(x)
  x = Conv2D(num_of_channels[3],
             kernel_size=kernel_sizes[2],
             padding="valid" if pad_inputs else "same",
             strides=stride_sizes[2])(x)
  x = SpatialDropout2D(dropout_rate)(x)
  x = InstanceNormalization()(x)
  # x = LeakyReLU()(x)
  x = relu(x)

  residual_block_filters = 128
  for _ in range(5):
    x = ResidualBlock(residual_block_filters, "instance", False)(x)
    x = SpatialDropout2D(dropout_rate)(x)

  x = UpSampling2D(size=stride_sizes[-1],
                   interpolation="nearest")(x)
  x = SpatialDropout2D(dropout_rate)(x)
  x = InstanceNormalization()(x)
  # x = LeakyReLU()(x)
  x = relu(x)
  x = Conv2D(num_of_channels[-2],
             kernel_size=kernel_sizes[-1],
             padding="same")(x)

  x = UpSampling2D(size=stride_sizes[-2],
                   interpolation="nearest")(x)
  x = SpatialDropout2D(dropout_rate)(x)
  x = InstanceNormalization()(x)
  # x = LeakyReLU()(x)
  x = relu(x)
  x = Conv2D(num_of_channels[-3],
             kernel_size=kernel_sizes[-2],
             padding="same")(x)
  x = SpatialDropout2D(dropout_rate)(x)
  x = Conv2D(num_of_channels[-4],
             kernel_size=kernel_sizes[-3],
             strides=stride_sizes[-3],
             padding="same")(x)
  # x = SpatialDropout2D(dropout_rate)(x)
  # x = tf.nn.tanh(x) * 150.

  return tf.keras.Model(input_layer, x, name="style_model")
