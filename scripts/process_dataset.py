import os
import random
import json
import PIL
import sys
import pandas as pd 
import json
from tqdm import tqdm

sys.path.append(os.path.join('..', 'config'))

from config import CONFIG

def compile_dataset_image_paths(path):
    """ 
    Compiles all the image paths and labels
    Args:
      path: Path of the dataset
    Returns:
      path_list: List of Image paths
      label_list: List of Corresponding labels
    """ 
    labels = os.listdir(path)
    label_list = []
    path_list = []
    for label in tqdm(labels):
        for file_name in os.listdir(f"{path}/{label}/"):
            label_list.append(label)
            path_list.append(f"{path}/{label}/{file_name}")

    return path_list,label_list


def main():
  """
  Reads the images from the datasets and creates Train and Test Dataframes which contains the paths 
  of images and labels
  Saves the dataframes and label id mapping in data/ folder
  """
  print("Processing the train dataset...")
  train_paths,train_labels = compile_dataset_image_paths(CONFIG['DATASET_TRAIN_PATH'])
  print("Processing the test dataset...")
  test_paths,test_labels = compile_dataset_image_paths(CONFIG['DATASET_TEST_PATH'])

  print("Creating train and test path dataframes...")
  labels_series = pd.Series(train_labels).unique()
  labels_series.sort()
  label_to_idx = { label:i for i, label in enumerate(labels_series.tolist())}

  train_df = pd.DataFrame({'path': train_paths, 'label': train_labels})
  train_df['label_id'] = train_df['label'].apply(lambda x: label_to_idx[x])

  test_df = pd.DataFrame({'path': test_paths, 'label': test_labels})
  test_df['label_id'] = test_df['label'].apply(lambda x: label_to_idx[x])

  train_df.to_csv("../data/train.csv",index=False)
  test_df.to_csv("../data/test.csv",index=False)

  with open("../data/label_to_idx.json","w") as f:
    f.write(json.dumps(label_to_idx))

  print("Saved the dataframes and label_id mapping at data/")


if __name__ == '__main__':
  main()



    





