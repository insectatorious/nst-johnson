from typing import Union, Any, Dict, Optional

import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.python.keras import Model


def get_model(style_layer_names, content_layer_names) -> tf.keras.Model:

  # Build a VGG19 model loaded with pre-trained ImageNet weights
  model: Model = vgg19.VGG19(weights='imagenet', include_top=False)

  model.trainable = False
  # Get the symbolic outputs of each "key" layer (we gave them unique names).
  # TODO: Type of layer.output
  outputs_dict: Dict[Optional[str], str] = dict([(layer.name, layer.output)
                       for layer in model.layers
                       if layer.name in style_layer_names or
                       layer.name in content_layer_names])

  # Set up a model that returns the activation values for every layer in
  # VGG19 (as a dict).
  feature_extractor: Model = tf.keras.Model(inputs=model.inputs,
                                            outputs=outputs_dict)

  feature_extractor.trainable = False

  return feature_extractor
