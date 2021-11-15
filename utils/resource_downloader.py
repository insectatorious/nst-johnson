import os
import logging
import argparse
import zipfile
from os.path import join, dirname, abspath

from tensorflow.keras.utils import get_file

# If the link is broken you can download the MS COCO 2014 dataset manually from http://cocodataset.org/#download
MS_COCO_2014_TRAIN_DATASET_PATH = r'http://images.cocodataset.org/zips/train2014.zip'  # ~13 GB after unzipping

PRETRAINED_MODELS_PATH = r'https://www.dropbox.com/s/fb39gscd1b42px1/pretrained_models.zip?dl=1'

DOWNLOAD_DICT = {
    'pretrained_models': PRETRAINED_MODELS_PATH,
    'mscoco_dataset': MS_COCO_2014_TRAIN_DATASET_PATH,
}
download_choices = list(DOWNLOAD_DICT.keys())


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                      level=logging.INFO)
  parser = argparse.ArgumentParser()

  parser.add_argument("--resource", "-r",
                      type=str,
                      choices=download_choices,
                      help="Specify which resource is required: MS COCO or "
                           "pretrained models",
                      default=download_choices[0])
  parser.add_argument("--dataset_path", "-p",
                      type=str,
                      help="Path to store MS COCO dataset (~40GB).",
                      default=join(abspath(join(os.getcwd(), os.pardir)), "data"))
  args = parser.parse_args()

  # Step 1: Download the resource to the local filesystem

  remote_resource_path: str = DOWNLOAD_DICT[args.resource]
  local_resource_path = args.dataset_path

  if args.resource == "pretrained_models":
    local_resource_path = join(local_resource_path, "models", 'binaries')
  else:
    local_resource_path = join(local_resource_path, "mscoco")

  local_resource_path = os.path.abspath(local_resource_path)
  os.makedirs(local_resource_path, exist_ok=True)
  logging.info(f"Saving to: {local_resource_path}")
  resource_tmp_path = get_file(fname=os.path.join(local_resource_path,
                                                  f"{args.resource}.zip"),
                               origin=remote_resource_path,
                               cache_dir=local_resource_path,
                               extract=True)

  # logging.info(f"Unzipping...")
  # with zipfile.ZipFile(resource_tmp_path) as zf:
  #   local_resource_path = os.path.join(os.path.dirname(__file__), os.pardir)
  #   if args.resource == "pretrained_models":
  #     local_resource_path = os.path.join(local_resource_path, 'models', 'binaries')
  #   else:
  #     local_resource_path = os.path.join(local_resource_path, 'data', 'mscoco')
  #
  #   os.makedirs(local_resource_path, exist_ok=True)
  #   zf.extractall(path=local_resource_path)

  logging.info(f"Unzipping to: {resource_tmp_path} completed.")

  # Step 3: Remove the temporary resource file
  # os.remove(resource_tmp_path)
  print(f"Removing tmp file {resource_tmp_path}")

