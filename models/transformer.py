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


class SimpleResidualBlock(tf.keras.layers.Layer):

  def __init__(self,
               num_of_filters: int = 16,
               norm_type: str = "batch",
               final_relu: bool = True,
               **kwargs):
    # TODO: Replace this ugly hack 👇 with an Enum
    if norm_type not in ["batch", "instance"]:
      raise AttributeError(f"Expected 'norm_type' to be one of 'batch' or "
                           f"'instance'. Got '{norm_type}")

    super(SimpleResidualBlock, self).__init__(**kwargs)
    self.num_of_filters = num_of_filters
    self.norm_type = norm_type
    self.final_relu = final_relu

    self.conv_1 = None
    self.norm_1 = None
    self.conv_2 = None
    self.norm_2 = None
    self.relu_1 = None
    if self.final_relu:
      self.relu_2 = None

  def build(self, input_shape: List) -> None:
    self.conv_1 = Conv2D(filters=self.num_of_filters,
                         kernel_size=3,
                         strides=1,
                         padding="same",
                         input_shape=input_shape)

    if self.norm_type == "batch":
      self.norm_1 = BatchNormalization()
    else:
      self.norm_1 = InstanceNormalization()
    self.relu_1 = LeakyReLU()
    self.conv_2 = Conv2D(filters=self.num_of_filters,
                         kernel_size=3,
                         strides=1,
                         padding="same")
    if self.norm_type == "batch":
      self.norm_2 = BatchNormalization()
    else:
      self.norm_2 = InstanceNormalization()
    if self.final_relu:
      self.relu_2 = LeakyReLU()

  def call(self, inputs: Tensor) -> Tensor:
    layer = self.conv_1(inputs)
    layer = self.norm_1(layer)
    layer = self.relu_1(layer)
    layer = self.conv_2(layer)
    layer = self.norm_2(layer)
    layer = layer + inputs
    if self.final_relu:
      layer = self.relu_2(layer)

    return layer

  def get_config(self) -> Dict:
    config = super(SimpleResidualBlock, self).get_config()
    config.update({"num_of_filters": self.num_of_filters,
                   "norm_type": self.norm_type,
                   "final_relu": self.final_relu})

    return config


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


class Transformer(tf.keras.Model):

  def __init__(self, **kwargs) -> tf.keras.Model:
    super(Transformer, self).__init__(**kwargs)
    self.num_of_channels = [3, 32, 64, 128]
    self.kernel_sizes = [9, 3, 3]
    self.stride_sizes = [1, 2, 2]
    self.pad_1 = ZeroPadding2D(2, input_shape=(None, None, None, 3))
    self.conv1 = Conv2D(self.num_of_channels[1],
                        kernel_size=self.kernel_sizes[0],
                        padding="same",
                        strides=self.stride_sizes[0])
    self.instance_norm_1 = InstanceNormalization()
    self.relu_1 = LeakyReLU()
    self.conv2 = Conv2D(self.num_of_channels[2],
                        kernel_size=self.kernel_sizes[1],
                        padding="same",
                        strides=self.stride_sizes[1])
    self.instance_norm_2 = InstanceNormalization()
    self.relu_2 = LeakyReLU()
    self.conv3 = Conv2D(self.num_of_channels[3],
                        padding="valid",
                        kernel_size=self.kernel_sizes[2],
                        strides=self.stride_sizes[2])
    self.instance_norm_3 = InstanceNormalization()
    self.relu_3 = LeakyReLU()

    residual_block_filters = 128
    self.res_block_1 = ResidualBlock(residual_block_filters, "instance", False)
    self.res_block_2 = ResidualBlock(residual_block_filters, "instance", False)
    self.res_block_3 = ResidualBlock(residual_block_filters, "instance", False)
    self.res_block_4 = ResidualBlock(residual_block_filters, "instance", False)
    self.res_block_5 = ResidualBlock(residual_block_filters, "instance", False)

    # For upsampling, we need to use a combination of the UpSampling2D followed
    # by a Conv2D layer to match his UpsampleConvLayer
    # See https://github.com/gordicaleksa/pytorch-neural-style-transfer-johnson/blob/00c96e8e3f1b0b7fb4c14254fd0c6f1281a29598/models/definitions/transformer_net.py#L106

    self.upsample_1 = UpSampling2D(size=self.stride_sizes[-1],
                                   interpolation="nearest")
    self.instance_norm_4 = InstanceNormalization()
    self.relu_4 = LeakyReLU()
    self.conv4 = Conv2D(self.num_of_channels[-2],
                        kernel_size=self.kernel_sizes[-1],
                        padding="same")

    self.upsample_2 = UpSampling2D(size=self.stride_sizes[-2],
                                   interpolation="nearest")
    self.instance_norm_5 = InstanceNormalization()
    self.relu_5 = LeakyReLU()
    self.conv5 = Conv2D(self.num_of_channels[-3],
                        kernel_size=self.kernel_sizes[-2],
                        padding="same")

    self.conv6 = Conv2D(self.num_of_channels[-4],
                        kernel_size=self.kernel_sizes[-3],
                        strides=self.stride_sizes[-3],
                        padding="same")

  def call(self, inputs: Tensor) -> Tensor:
    x = self.pad_1(inputs)
    x = self.relu_1(self.instance_norm_1(self.conv1(x)))
    x = self.relu_2(self.instance_norm_2(self.conv2(x)))
    x = self.relu_3(self.instance_norm_3(self.conv3(x)))

    x = self.res_block_1(x)
    x = self.res_block_2(x)
    x = self.res_block_3(x)
    x = self.res_block_4(x)
    x = self.res_block_5(x)

    x = self.conv4(self.relu_4(self.instance_norm_4(self.upsample_1(x))))
    x = self.conv5(self.relu_5(self.instance_norm_5(self.upsample_2(x))))

    return self.conv6(x)


def get_transformer(num_of_channels=None,
                    kernel_sizes=None,
                    stride_sizes=None,
                    dropout_rate: int = 0.1) -> tf.keras.Model:
  if num_of_channels is None:
    num_of_channels = [3, 32, 64, 128]
  if kernel_sizes is None:
    kernel_sizes = [9, 3, 3]
  if stride_sizes is None:
    stride_sizes = [1, 2, 2]
  relu = LeakyReLU()
  input_layer = tf.keras.Input(shape=(None, None, 3), name="input_img")
  x = tf.pad(input_layer, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
  # x = ZeroPadding2D(2)(input_layer)
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
             padding="same",
             strides=stride_sizes[1])(x)
  x = SpatialDropout2D(dropout_rate)(x)
  x = InstanceNormalization()(x)
  # x = LeakyReLU()(x)
  x = relu(x)
  x = Conv2D(num_of_channels[3],
             kernel_size=kernel_sizes[2],
             padding="valid",
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
