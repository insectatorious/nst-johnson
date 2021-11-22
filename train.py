import pprint
import argparse
import logging
from argparse import Namespace
from datetime import datetime, timedelta
from os import makedirs, getcwd, pardir
from os.path import join, dirname, exists, isfile, basename, abspath
from time import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer, Adam, SGD
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.perceptual_loss import get_model, PerceptualModelType
from utils.utils import (
  get_img_dimensions,
  get_training_strategy,
  preprocess_image,
  preprocess_img_tensor,
  get_pbar,
  batch_gram_matrix,
  deprocess_image,
  compute_losses
)
from models.transformer import get_transformer


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

    (batch_content_loss,
     batch_style_loss,
     batch_total_variation_loss) = compute_losses(
      stylised_batch=stylised_batch,
      target_style_representation=target_style_representation,
      content_feature_maps=[content_batch_feature_maps[x]
                            for x in content_batch_feature_maps
                            if x in content_layer_names],
      stylised_content_feature_maps=[stylised_batch_feature_maps[x]
                                     for x in stylised_batch_feature_maps
                                     if x in content_layer_names],
      stylised_style_feature_maps=[stylised_batch_feature_maps[x]
                                   for x in stylised_batch_feature_maps
                                   if x in style_layer_names]
    )

    loss = content_weight * batch_content_loss
    loss += style_weight * batch_style_loss
    loss += total_variation_weight * batch_total_variation_loss

  grads = tape.gradient(loss, transformer.trainable_weights)
  optimiser.apply_gradients(zip(grads, transformer.trainable_weights))

  return (batch_content_loss * content_weight,
          batch_style_loss * style_weight,
          batch_total_variation_loss * total_variation_weight), grads, stylised_batch


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
#   per_replica_losses, _, _ = strategy.run(train_batch,
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

  with strategy.scope():

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=config["initial_learning_rate"],
      decay_steps=1000,
      decay_rate=0.9,
      staircase=False
    )
    if config["optimiser"] == "adam":
      optimiser = Adam(learning_rate=lr_schedule,
                       beta_1=0.7,
                       beta_2=0.7,
                       clipnorm=5.0)
    else:
      optimiser = SGD(learning_rate=lr_schedule,
                      momentum=0.5,
                      nesterov=True,
                      clipnorm=5.0)

    with config["tf_writer"].as_default():
      interesting_hparams = ("augment_content",
                             "content_layer_names",
                             "content_weight",
                             "img_ncols",
                             "img_nrows",
                             "initial_learning_rate",
                             "min_improvement",
                             "optimiser",
                             "batch_size",
                             "epochs",
                             "patience",
                             "style_img_path",
                             "style_layer_names",
                             "style_weight",
                             "total_variation_weight",
                             "tpu")
      hparams_to_log = {k:config[k] for k in interesting_hparams}
      tf.summary.text("Config", tf.constant(pprint.pformat(hparams_to_log)),
                      step=0,
                      description="Hyperparameters for this run")

    transformer: Model = get_transformer(pad_inputs=True, dropout_rate=0.0)
    perceptual_loss: Model = get_model(config["style_layer_names"],
                                       config["content_layer_names"],
                                       img_ncols=config["img_ncols"],
                                       img_nrows=config["img_nrows"],
                                       model_type=PerceptualModelType.VGG_16)

    # transformer.build(input_shape=(None, None, None, 3))
    # transformer.compile(optimizer=optimiser)
    # transformer.build(input_shape=(None, config["img_nrows"], config["img_ncols"], 3))
    # transformer.call(tf.keras.layers.Input(shape=(config["img_nrows"], config["img_ncols"], 3)))
    # transformer.summary()

    style_image = preprocess_image(image_path=config["style_img_path"],
                                   img_nrows=config["img_nrows"],
                                   img_ncols=config["img_ncols"],
                                   model_type=PerceptualModelType.VGG_16)
    with config["img_writer"].as_default():
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
    last_mean_loss = mean_loss
    last_durations = []
    lowest_loss = np.Inf
    iterations_since_improvement = 0
    start_time = time()
    pbar: tqdm = get_pbar()

    for i, batch in enumerate(config["dataset"]):
      pbar.update(1)

      content_batch = preprocess_img_tensor(batch,
                                            config["img_nrows"],
                                            config["img_ncols"],
                                            PerceptualModelType.VGG_16)
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
       batch_total_variation_loss), grads, stylised_batch = train_batch(
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

        with config["metric_writers"]["min"].as_default():
          tf.summary.scalar(name="stylised_batch/metrics",
                            data=tf.reduce_min(stylised_batch),
                            step=i + 1)
        with config["metric_writers"]["max"].as_default():
          tf.summary.scalar(name="stylised_batch/metrics",
                            data=tf.reduce_max(stylised_batch),
                            step=i + 1)
        with config["metric_writers"]["mean"].as_default():
          tf.summary.scalar(name="stylised_batch/metrics",
                            data=tf.reduce_mean(stylised_batch),
                            step=i + 1)
        with config["metric_writers"]["median"].as_default():
          tf.summary.scalar(name="stylised_batch/metrics",
                          data=tfp.stats.percentile(stylised_batch,
                                                    50.0,
                                                    interpolation='midpoint'),
                            step=i + 1)
        with config["tf_writer"].as_default():
          tf.summary.scalar(name="loss/total",
                            data=batch_style_loss +
                                 batch_total_variation_loss +
                                 batch_content_loss,
                            step=i + 1)
          tf.summary.scalar(name="loss/content_loss", data=batch_content_loss, step=i + 1)
          tf.summary.scalar(name="loss/style_loss", data=batch_style_loss, step=i + 1)
          tf.summary.scalar(name="loss/tv_loss", data=batch_total_variation_loss, step=i + 1)
          tf.summary.scalar(name="learning_rate", data=lr_schedule(i + 1), step=i + 1)

      loss = batch_content_loss + batch_style_loss + batch_total_variation_loss
      mean_loss += loss.numpy()
      mean_loss /= i % 100 + 1

      if i % 100 == 0:
        duration = time() - start_time
        last_durations.append(duration)
        pbar.close()
        direction = "\u2197" if last_mean_loss <= mean_loss else "\u2198"
        logging.info(f"Step {i:05d}"
                     f"[{i / config['total_steps'] * 100.:05.2f}%]: "
                     f"loss={mean_loss:,.2f}"
                     f"[{direction} {np.abs(last_mean_loss - mean_loss):,.2f}]"
                     f" @ {duration / 100.:,.4f} secs/step")
        last_mean_loss = mean_loss
        if i % 1000 == 0 and i > 0:
          avg_duration_per_step = np.mean(last_durations[-2000:]) / 100.
          variation_per_step = np.var(np.asarray(last_durations) / 100.)
          steps_left = config['total_steps'] - i
          time_left_in_secs = steps_left * avg_duration_per_step
          time_left_in_secs_variation = steps_left * (avg_duration_per_step + variation_per_step)
          tolerance_in_secs = np.abs(time_left_in_secs_variation - time_left_in_secs)
          eta = datetime.now() + timedelta(seconds=time_left_in_secs)
          logging.info(f"ETA for the remaining {steps_left:,d} steps: "
                       f"{time_left_in_secs / 60.:.2f} minutes "
                       f"({eta.strftime('%Y-%m-%d %H:%M')} "
                       f"\u00B1 {tolerance_in_secs / 60.:.1f}mins)")
          # last_durations = []
          if i / config["total_steps"] > .5 and batch_style_loss - batch_content_loss > 1000.:
            logging.warning(f"'style_weight' ({config['style_weight']}) might be too high compared "
                            f"to content_weight({config['content_weight']}) as the "
                            f"loss between the two is still quite large "
                            f"(style_loss: {batch_style_loss:,.2f} "
                            f"content_loss: {batch_content_loss:,.2f}). "
                            f"Consider reducing the 'style_weight' or increasing "
                            f"the 'content_weight'. Alternatively, train for more "
                            f"epochs if the loss is decreasing.")

        pbar = get_pbar()
        mean_loss = 0
        with config["img_writer"].as_default():
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

      if i > config["total_steps"]:
        break

    return transformer


if __name__ == '__main__':
  # Set seeds for random generators
  # np.random.seed(42)
  # tf.random.set_seed(42)

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
                      default=1e-7)
  parser.add_argument("--style_weight",
                      type=float,
                      help="Weight factor for style loss. Higher values mean "
                           "more of the style image will be kept.",
                      default=2e0)
  parser.add_argument("--tv_weight",
                      type=float,
                      help="Weight factor for total variation loss. Affects "
                           "sharpness vs smoothness of the resulting image.",
                      default=0)  # 1e9)
  parser.add_argument("--batch_size",
                      type=int,
                      default=3,
                      help="Number of images to train in one go")
  parser.add_argument("--content_width",
                      type=int,
                      default=300,
                      help="Width to resize each image in MS COCO to before "
                           "passing through the network")
  parser.add_argument("--num_of_epochs",
                      type=int,
                      default=2,
                      help="Number of epochs (early stopping can result in "
                           "less than this)")
  parser.add_argument("--patience",
                      type=int,
                      help="Number of steps without any improvement.",
                      default=3050)
  parser.add_argument("--dataset_path",
                      type=str,
                      help="Path to MS COCO dataset",
                      default=join(dirname(__file__), "data"))
  parser.add_argument("--fixed_imgs",
                      action="store_true",
                      help="If True (default: False) then augment content images")
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
  model_name: str = basename(args.style_img_name).split(".")[0]

  checkpoints_path: str = join(checkpoints_root_path, model_name)
  makedirs(checkpoints_path, exist_ok=True)
  # Dimensions of the generated picture.
  width, height = get_img_dimensions(args.style_img_name)

  # TODO: 👇 Make this a script parameter
  img_nrows = args.content_width

  if args.fixed_imgs:
    dataset = image_dataset_from_directory(
      directory=join(args.dataset_path, "mscoco", "datasets"),
      labels=None,
      label_mode=None,
      color_mode="rgb",
      batch_size=args.batch_size,
      interpolation="nearest",
      image_size=(img_nrows, img_nrows),
      shuffle=False,
    ).repeat()
  else:
    # 0345 604 5629
    dataset = ImageDataGenerator(
      rotation_range=20,
      horizontal_flip=True,
      vertical_flip=False,
      fill_mode="reflect",
      width_shift_range=0.1,
      height_shift_range=100.1,
      # brightness_range=(-0.002, 0.002),
      zoom_range=0.15,
      shear_range=0.15
    ).flow_from_directory(
      directory=join(args.dataset_path, "mscoco", "datasets"),
      class_mode=None,
      target_size=(img_nrows, img_nrows),
      shuffle=False,
      batch_size=args.batch_size,
      color_mode="rgb"
    )

  logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  writer = tf.summary.create_file_writer(logdir + "/loss")
  img_writer = tf.summary.create_file_writer(logdir + "/images")
  metric_writers = {}
  for metric in ["max", "min", "median", "mean"]:
    metric_logdir = f"{logdir}/metrics/{metric}"
    metric_writers[metric] = tf.summary.create_file_writer(metric_logdir)

  config = dict(
    style_img_path=args.style_img_name,

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
    initial_learning_rate=.01,
    optimiser="sgd",
    epochs=args.num_of_epochs,
    batch_size=args.batch_size,
    total_steps=82783 // args.batch_size * args.num_of_epochs,

    # List of layers to use for the style loss.
    style_layer_names=[
      'block1_conv2',
      'block2_conv2',
      'block3_conv3',
      'block4_conv3'
    ],
    # style_layer_names=[
    #   'block1_conv1',
    #   'block2_conv1',
    #   'block3_conv1',
    #   'block4_conv1',
    #   'block5_conv1'
    # ],

    # List of layers to use for the content loss.
    content_layer_names=[
      # "block1_conv2",
      "block3_conv3"
    ],
    dataset=dataset,
    tf_writer=writer,
    metric_writers=metric_writers,
    img_writer=img_writer,

    # Data augmentation
    augment_content=args.fixed_imgs

  )

  logging.info(f"Training for {config['total_steps']} steps")
  transformer: Model = train(config)
  transformer.save(join(model_root_path, model_name))
  logging.info(f"Saved {model_name} at {model_root_path}")
