import pprint
import argparse
import logging
from argparse import Namespace
from datetime import datetime
from os import makedirs
from os.path import join, dirname, exists, isfile, basename
from time import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from tensorflow import Tensor
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer, Adam, SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import image_dataset_from_directory

from models.perceptual_loss import get_model
from utils.utils import (
  get_img_dimensions,
  get_training_strategy,
  preprocess_image,
  preprocess_img_tensor,
  get_pbar,
  batch_gram_matrix,
  deprocess_image
)
from models.transformer import Transformer


def content_loss(content_batch_feature_maps,
                 stylised_batch_feature_maps,
                 content_layer_names) -> Tensor:
  loss = tf.zeros(shape=())
  for content_layer in content_layer_names:
    target_content_representation = content_batch_feature_maps[content_layer]
    current_content_representation = stylised_batch_feature_maps[content_layer]
    loss += MeanSquaredError()(target_content_representation,
                               current_content_representation)

  loss /= len(content_layer_names)

  return loss


def style_loss(target_style_representation,
               stylised_batch_feature_maps,
               style_layer_names,
               channels: int = 3) -> Tensor:
  loss = tf.zeros(shape=())
  current_style_representation = [batch_gram_matrix(stylised_batch_feature_maps[x],
                                                    normalise=True)
                                  for x in stylised_batch_feature_maps
                                  if x in style_layer_names]
  for gram_gt, gram_hat in zip(target_style_representation,
                               current_style_representation):
    # loss += MeanSquaredError()(gram_gt, gram_hat)
    S, C = gram_gt, gram_hat
    # print("@@@@@", gram_gt.shape)
    size = gram_gt.shape[1] * gram_gt.shape[2]
    loss += tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

  loss /= len(target_style_representation)

  return loss


def total_variation_loss(stylised_batch):
  # Small change 👇 to reduce_sum instead of mean
  return tf.reduce_mean(tf.image.total_variation(stylised_batch))


@tf.function
def train_batch(transformer: Model,
                perceptual_loss: Model,
                optimiser: Optimizer,
                content_batch,
                target_style_representation,
                content_weight,
                style_weight,
                total_variation_weight,
                content_layer_names,
                style_layer_names):
  with tf.GradientTape() as tape:
    # Step 1: Feed content batch through the transformer
    stylised_batch = transformer(content_batch)

    # Step 2: Feed content and stylised batch through perceptual net (VGG19)
    content_batch_feature_maps = perceptual_loss(content_batch)
    stylised_batch_feature_maps = perceptual_loss(stylised_batch)

    # Step 3: Calculate content representations and content loss
    batch_content_loss = content_weight * content_loss(content_batch_feature_maps,
                                                       stylised_batch_feature_maps,
                                                       content_layer_names)

    # Step 4: Calculate style representations and style loss
    batch_style_loss = style_weight * style_loss(target_style_representation,
                                                 stylised_batch_feature_maps,
                                                 style_layer_names)

    # Step 5: Calculate total variation loss
    batch_total_variation_loss = (total_variation_weight *
                                  total_variation_loss(stylised_batch))

    # Step 6: Combine losses
    loss = batch_content_loss + batch_style_loss + batch_total_variation_loss

  grads = tape.gradient(loss, transformer.trainable_weights)
  optimiser.apply_gradients(zip(grads, transformer.trainable_weights))

  return (batch_content_loss, batch_style_loss, batch_total_variation_loss), grads


# @tf.function
# def distributed_train_step(strategy,
#                            transformer: Model,
#                            perceptual_loss: Model,
#                            optimiser: Optimizer,
#                            content_batch,
#                            target_style_representation,
#                            content_weight,
#                            style_weight,
#                            total_variation_weight,
#                            content_layer_names,
#                            style_layer_names):
#   per_replica_losses, _ = strategy.run(train_batch,
#                                        args=(transformer,
#                                              perceptual_loss,
#                                              optimiser,
#                                              content_batch,
#                                              target_style_representation,
#                                              content_weight,
#                                              style_weight,
#                                              total_variation_weight,
#                                              content_layer_names,
#                                              style_layer_names))
#
#   return strategy.reduce(tf.distribute.ReduceOp.SUM,
#                          per_replica_losses,
#                          axis=0)


def train(config) -> Model:
  strategy = get_training_strategy(config)

  with strategy.scope(), config["tf_writer"].as_default():

    if config["optimiser"] == "adam":
      optimiser = Adam(learning_rate=config["initial_learning_rate"])
    else:
      optimiser = SGD(learning_rate=config["initial_learning_rate"],
                      # momentum=0.7,
                      # nesterov=True,
                      clipnorm=None)

    tf.summary.text("Config", tf.constant(pprint.pformat(config)),
                    step=0,
                    description="Hyperparameters for this run")

    transformer: Model = Transformer()
    perceptual_loss: Model = get_model(config["style_layer_names"],
                                       config["content_layer_names"])

    transformer.build(input_shape=(None, config["img_nrows"], config["img_ncols"], 3))
    transformer.call(tf.keras.layers.Input(shape=(config["img_nrows"], config["img_ncols"], 3)))
    transformer.summary()

    # perceptual_loss.summary()

    style_image = preprocess_image(image_path=config["style_img_path"],
                                   img_nrows=config["img_nrows"],
                                   img_ncols=config["img_ncols"])
    tf.summary.image("style_image",
                     deprocess_image(style_image,
                                     config["img_nrows"],
                                     config["img_ncols"]),
                     step=0,
                     max_outputs=1,
                     description="Style to be learned.")
    style_image_feature_maps = perceptual_loss(style_image)
    target_style_representation = [batch_gram_matrix(style_image_feature_maps[x])
                                   for x in style_image_feature_maps
                                   if x in config["style_layer_names"]]

    mean_loss = 0
    lowest_loss = np.Inf
    iterations_since_improvement = 0
    start_time = time()
    pbar: tqdm = get_pbar()

    for i, batch in enumerate(config["dataset"]):
      pbar.update(1)

      content_batch = preprocess_img_tensor(batch,
                                            config["img_nrows"],
                                            config["img_ncols"])
      if config["rotate_content"]:
        tfa.image.rotate(images=content_batch,
                         angles=np.random.uniform(-5, 5,
                                                  size=content_batch.shape[0]))
      # loss = distributed_train_step(strategy,
      #                               transformer,
      #                               perceptual_loss,
      #                               optimiser,
      #                               content_batch,
      #                               target_style_representation,
      #                               config["content_weight"],
      #                               config["style_weight"],
      #                               config["total_variation_weight"],
      #                               config["content_layer_names"],
      #                               config["style_layer_names"])
      (batch_content_loss,
       batch_style_loss,
       batch_total_variation_loss), grads = train_batch(
        transformer,
        perceptual_loss,
        optimiser,
        content_batch,
        target_style_representation,
        config["content_weight"],
        config["style_weight"],
        config["total_variation_weight"],
        config["content_layer_names"],
        config["style_layer_names"])

      if i % 10 == 0:
        # https://stackoverflow.com/a/56961915
        tf.summary.scalar(name="content_loss", data=batch_content_loss, step=i + 1)
        tf.summary.scalar(name="style_loss", data=batch_style_loss, step=i + 1)
        tf.summary.scalar(name="tv_loss", data=batch_total_variation_loss, step=i + 1)

      loss = batch_content_loss + batch_style_loss + batch_total_variation_loss
      mean_loss += loss.numpy()
      mean_loss /= i % 100 + 1

      if i % 100 == 0:
        duration = time() - start_time
        pbar.close()
        logging.info(f"Iteration {i:06d}: loss={mean_loss:,.2f} @ "
                     f"{duration / 100.:,.4f} secs per iter")

        pbar = get_pbar()
        mean_loss = 0
        tf.summary.image("content_image",
                         deprocess_image(content_batch,
                                         config["img_nrows"],
                                         config["img_ncols"]),
                         step=i + 1,
                         max_outputs=3)
        tf.summary.image("stylised_image",
                         deprocess_image(transformer.call(content_batch),
                                         config["img_nrows"],
                                         config["img_ncols"]),
                         step=i + 1, max_outputs=3)
        start_time = time()

      if lowest_loss - mean_loss if mean_loss > 0 else np.Inf >= config["min_improvement"]:
        lowest_loss = mean_loss
        iterations_since_improvement = 0
      else:
        iterations_since_improvement += 1

      if iterations_since_improvement > config["patience"]:
        logging.info(f"Patience of {config['patience']} steps exhausted. Terminating.")
        break

      if i > 50000:
        break

    return transformer


if __name__ == '__main__':
  # Set seeds for random generators
  np.random.seed(42)
  tf.random.set_seed(42)

  # Enable XLA
  tf.config.optimizer.set_jit(True)

  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                      level=logging.INFO)

  parser = argparse.ArgumentParser()

  parser.add_argument("style_img_name",
                      type=str,
                      help="Path to style image that will be used for training.")
  parser.add_argument("--content_weight",
                      type=float,
                      help="Weight factor for content loss. Higher values mean "
                           "more of the original image will be kept.",
                      default=1e0)
  parser.add_argument("--style_weight",
                      type=float,
                      help="Weight factor for style loss. Higher values mean "
                           "more of the style image will be kept.",
                      default=1e0)
  parser.add_argument("--tv_weight",
                      type=float,
                      help="Weight factor for total variation loss. Affects "
                           "sharpness vs smoothness of the resulting image.",
                      default=0e0)  # 1e9)
  parser.add_argument("--patience",
                      type=int,
                      help="Number of steps without any improvement.",
                      default=1050)
  parser.add_argument("--dataset_path",
                      type=str,
                      help="Path to MS COCO dataset",
                      default=join(dirname(__file__), "data"))
  args: Namespace = parser.parse_args()

  assert exists(args.style_img_name), \
    f"Style image not found at {args.style_img_name}"

  assert isfile(args.style_img_name), \
    f"Style image path is not a valid file: {args.style_img_name}"

  # Create directory for downloading MS COCO
  makedirs(args.dataset_path, exist_ok=True)

  checkpoints_root_path: str = join(dirname(__file__),
                                    args.dataset_path,
                                    "models",
                                    "checkpoints")
  model_root_path: str = join(dirname(__file__),
                              args.dataset_path,
                              "models",
                              "binaries")

  logging.debug(f"Creating {model_root_path}")
  makedirs(model_root_path, exist_ok=True)

  checkpoints_path: str = join(checkpoints_root_path,
                               basename(args.style_img_name).split(".")[0])
  makedirs(checkpoints_path, exist_ok=True)

  # Dimensions of the generated picture.
  width, height = get_img_dimensions(args.style_img_name)

  # TODO: 👇 Make this a script parameter
  img_nrows = 224

  dataset = image_dataset_from_directory(
    directory=args.dataset_path,
    labels=None,
    label_mode=None,
    color_mode="rgb",
    batch_size=2,
    image_size=(224, 224),
    shuffle=False,

  )
  # dataset = dataset.map(normalise).repeat()

  logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
  writer = tf.summary.create_file_writer(logdir)

  config = dict(
    style_img_path=args.style_img_name,
    # result_prefix="tabitha_viii_cc",
    # Weights of the different loss components
    total_variation_weight=args.tv_weight,
    style_weight=args.style_weight,
    content_weight=args.content_weight,

    # Dimensions of the generated picture.
    width=width,
    height=height,
    img_nrows=img_nrows,
    img_ncols=img_nrows,  # int(width * img_nrows / height),

    patience=args.patience,  # Steps without improvement
    min_improvement=0.1,
    initial_learning_rate=0.0001,
    optimiser="adam",
    lr_decay_steps=2500,

    # List of layers to use for the style loss.
    style_layer_names=[
      'block1_conv1', 'block2_conv1',
      'block3_conv1', 'block4_conv1',
      'block5_conv1'
    ],

    # List of layers to use for the content loss.
    content_layer_names=["block5_conv2", "block5_conv3"],
    dataset=dataset,
    tf_writer=writer,

    # Data augmentation
    rotate_content=True

  )

  transformer: Model = train(config)
