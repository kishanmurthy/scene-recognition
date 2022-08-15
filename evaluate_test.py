import time
import os
import pickle
import random
import json
import PIL
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

import torchvision
from torchvision import transforms

from transformers import ViTForImageClassification,ViTFeatureExtractor

from config import CONFIG
from dataset import Places365
from utils import ModelSaver, fetch_label_mappings, eval_one_epoch,collate_fn
from utils import pretty_print_results
DISABLE_TQDM= CONFIG['DISABLE_TQDM']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'google/vit-base-patch16-224-in21k'


def main():
    """
    Evaluates model on the test set
    """
    
    test_df = pd.read_csv(os.path.join(CONFIG['DATASET_MAPPINGS_PATH'], "test.csv"))
    idx_to_label,label_to_idx = fetch_label_mappings(\
                            os.path.join(CONFIG['DATASET_MAPPINGS_PATH'], "label_to_idx.json"))
    num_labels = len(idx_to_label)

    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
    dataset = Places365(test_df.path.values,test_df.label_id.values,feature_extractor)
    
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8,collate_fn=collate_fn)

    model_saver = ModelSaver("vit_trained.pt", os.path.join(CONFIG['WANDB_PATH'],"wandb/latest-run/files"))
    
    model = model_saver.fetch()
    model = model.to(DEVICE)

    compute_metrics = {
        'Accuracy Top 1': Accuracy(average='macro',num_classes=num_labels).to(DEVICE),
        'Accuracy Top 5': Accuracy(average='macro',top_k=5,num_classes=num_labels).to(DEVICE),
        'Precision' : Precision(average='macro', num_classes=num_labels).to(DEVICE),
        'Recall' : Recall(average='macro', num_classes=num_labels).to(DEVICE),
        'F1-Score': F1Score(average='macro',num_classes=num_labels).to(DEVICE),
        'AUROC': AUROC(average='macro', num_classes=num_labels).to(DEVICE)
    }

    test_loss, metrics = eval_one_epoch(model,test_loader,compute_metrics)
    metrics['Test Loss'] = test_loss
    pretty_print_results(metrics)

    with open(os.path.join(CONFIG['WANDB_PATH'],"wandb/latest-run/files/test_results.json"),"w") as f:
        f.write(json.dumps(metrics))
    


if __name__ == '__main__':
    main()