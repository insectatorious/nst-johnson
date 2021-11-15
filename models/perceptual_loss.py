from enum import Enum
from typing import Union, Any, Dict, Optional

import tensorflow as tf
from tensorflow.keras.applications import vgg19, vgg16
from tensorflow.python.keras import Model


class PerceptualModelType(Enum):
  VGG_19 = 1
  VGG_16 = 2


def get_model(style_layer_names,
              content_layer_names,
              img_ncols: int = 224,
              img_nrows: int = 224,
              model_type: PerceptualModelType = PerceptualModelType.VGG_19) -> tf.keras.Model:
  if model_type == PerceptualModelType.VGG_19:
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model: Model = vgg19.VGG19(weights='imagenet',
                               include_top=False,
                               input_shape=(img_ncols, img_nrows, 3))
  else:
    # Build a VGG16 model loaded with pre-trained ImageNet weights
    model: Model = vgg16.VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(img_ncols, img_nrows, 3))

  model.trainable = False
  # Get the symbolic outputs of each "key" layer (we gave them unique names).
  # TODO: Type of layer.output
  outputs_dict: Dict[Optional[str], str] = dict([(layer.name, layer.output)
                                                 for layer in model.layers
                                                 if layer.name in style_layer_names or
                                                 layer.name in content_layer_names])

  # Set up a model that returns the activation values for every layer (as a dict)
  feature_extractor: Model = tf.keras.Model(inputs=model.inputs,
                                            outputs=outputs_dict)

  feature_extractor.trainable = False

  return feature_extractor
