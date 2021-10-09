from typing import List, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import (
  Conv2D,
  UpSampling2D,
  BatchNormalization,
  LeakyReLU,
  Add
)
from tensorflow_addons.layers import InstanceNormalization


class ResidualBlock(tf.keras.layers.Layer):

  def __init__(self, num_of_filters: int = 16, **kwargs):
    super(ResidualBlock, self).__init__(**kwargs)
    self.num_of_filters = num_of_filters
    self.conv_1 = None
    self.conv_2 = None
    self.relu_1 = None
    self.relu_2 = None
    self.batch_norm_1 = None
    self.batch_norm_2 = None
    self.add_1 = None

  def build(self, input_shape: List) -> None:
    self.conv_1 = Conv2D(filters=self.num_of_filters,
                         kernel_size=(1, 1),
                         input_shape=input_shape)
    self.batch_norm_1 = BatchNormalization()
    self.relu_1 = LeakyReLU()
    self.conv_2 = Conv2D(filters=self.num_of_filters,
                         kernel_size=3,
                         padding="same")
    self.batch_norm_2 = BatchNormalization()
    self.add_1 = Add()
    self.relu_2 = LeakyReLU()

  def call(self, inputs: Tensor) -> Tensor:
    layer = self.conv_1(inputs)
    layer = self.batch_norm_1(layer)
    layer = self.relu_1(layer)
    layer = self.conv_2(layer)
    layer = self.batch_norm_2(layer)
    layer = self.add_1([layer, inputs])
    layer = self.relu_2(layer)

    return layer

  def get_config(self) -> Dict:
    config = super(ResidualBlock, self).get_config()
    config.update({"num_of_filters": self.num_of_filters})

    return config

#
# class UpsampleConv(tf.keras.layers.Layer):
#   """Nearest-neighbour up-sampling followed by a convolution.
#
#   Appears to give better results than learned up-sampling aka transposed conv.
#   Initially proposed on distill pub: http://distill.pub/2016/deconv-checkerboard/#
#   """
#
#   def __init__(self,
#                out_channels: int,
#                kernel_size: Optional[int, Tuple[int, int]],
#                stride: Optional[int, Tuple[int, int]],
#                **kwargs):
#     super(UpsampleConv, self).__init__(**kwargs)
#     self.upsampling_factor = stride
#     self.out_channels = out_channels
#     self.kernel_size = kernel_size
#     self.conv2d = None
#
#   def build(self, input_shape: List) -> None:
#     self.conv2d = Conv2D(filters=self.out_channels,
#                          kernel_size=self.kernel_size,
#                          strides=self.upsampling_factor)
#
#   def call(self, inputs: Tensor) -> Tensor:
#     if self.upsampling_factor > 1:
#       pass
#
#     return self.conv2d(inputs)


class Transformer(tf.keras.Model):

  def __init__(self, **kwargs) -> tf.keras.Model:
    super(Transformer, self).__init__(**kwargs)
    self.num_of_channels = [3, 32, 64, 128]
    self.kernel_sizes = [9, 3, 3]
    self.stride_sizes = [1, 2, 2]
    self.conv1 = Conv2D(self.num_of_channels[1],
                        kernel_size=self.kernel_sizes[0],
                        strides=self.stride_sizes[0])
    self.instance_norm_1 = InstanceNormalization()
    self.relu_1 = LeakyReLU()
    self.conv2 = Conv2D(self.num_of_channels[2],
                        kernel_size=self.kernel_sizes[1],
                        strides=self.stride_sizes[1])
    self.instance_norm_2 = InstanceNormalization()
    self.relu_2 = LeakyReLU()
    self.conv3 = Conv2D(self.num_of_channels[3],
                        kernel_size=self.kernel_sizes[2])
    self.instance_norm_3 = InstanceNormalization()
    self.relu_3 = LeakyReLU()

    residual_block_filters = 128
    self.res_block_1 = ResidualBlock(residual_block_filters)
    self.res_block_2 = ResidualBlock(residual_block_filters)
    self.res_block_3 = ResidualBlock(residual_block_filters)
    self.res_block_4 = ResidualBlock(residual_block_filters)
    self.res_block_5 = ResidualBlock(residual_block_filters)

    self.upsample_1 = UpSampling2D(size=self.stride_sizes[-1],
                                   interpolation="nearest")
    self.instance_norm_4 = InstanceNormalization()
    self.relu_4 = LeakyReLU()
    self.conv4 = Conv2D(self.num_of_channels[-1],
                        kernel_size=self.kernel_sizes[-1],
                        strides=self.stride_sizes[-1])

    self.upsample_2 = UpSampling2D(size=self.stride_sizes[-2],
                                   interpolation="nearest")
    self.instance_norm_5 = InstanceNormalization()
    self.relu_5 = LeakyReLU()
    self.upsample_3 = UpSampling2D(size=self.stride_sizes[-3],
                                   interpolation="nearest")
    self.conv5 = Conv2D(self.num_of_channels[0],
                        kernel_size=self.kernel_sizes[0],
                        strides=self.stride_sizes[0])

  def call(self, inputs: Tensor) -> Tensor:
    x = self.relu_1(self.instance_norm_1(self.conv1(inputs)))
    x = self.relu_2(self.instance_norm_2(self.conv2(x)))
    x = self.relu_3(self.instance_norm_3(self.conv3(x)))

    x = self.res_block_1(x)
    x = self.res_block_2(x)
    x = self.res_block_3(x)
    x = self.res_block_4(x)
    x = self.res_block_5(x)

    x = self.conv4(self.relu_4(self.instance_norm_4(self.upsample_1(x))))
    x = self.relu_5(self.instance_norm_5(self.upsample_2(x)))

    return self.conv5(self.upsample_3(x))
