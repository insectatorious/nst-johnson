import os
import argparse
import logging
from os.path import dirname, join, basename

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import save_img

from utils.utils import get_img_dimensions, preprocess_image, deprocess_image

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                      level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument("model_name",
                      type=str,
                      help="Model name")
  parser.add_argument("input_img_path",
                      type=str,
                      help="Path to image that will be used styled")
  parser.add_argument("--data_path",
                      type=str,
                      help="Path to data directory",
                      default=join(dirname(__file__), "data"))
  parser.add_argument("--style_width",
                      type=int,
                      help="Width in pixels of output image",
                      default=800)
  args: argparse.Namespace = parser.parse_args()
  model_root_path: str = join(dirname(__file__),
                              args.data_path,
                              "models",
                              "binaries")
  output_path: str = join(dirname(__file__),
                          args.data_path,
                          "data",
                          "output")
  os.makedirs(output_path, exist_ok=True)

  with tf.device("/cpu:0"):
    transformer = load_model(join(model_root_path, args.model_name),
                             compile=False)

    width, height = get_img_dimensions(args.input_img_path)
    img_nrows = int(height * args.style_width / width)
    input_image: Tensor = preprocess_image(args.input_img_path,
                                           img_nrows=img_nrows,
                                           img_ncols=args.style_width)
    styled_image = transformer(input_image).numpy().squeeze()
    styled_image = deprocess_image(styled_image,
                                   img_nrows=styled_image.shape[0],
                                   img_ncols=styled_image.shape[1])

    output_filename: str = join(output_path,
                                f"{basename(args.input_img_path).split('.')[0]}-"
                                f"{args.model_name}.png")
    save_img(output_filename, styled_image)
    logging.info(f"Saved {output_filename}")
