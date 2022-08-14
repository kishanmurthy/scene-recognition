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
import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR,OneCycleLR
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

import torchvision
from torchvision import transforms
import cv2

from transformers import ViTForImageClassification,ViTFeatureExtractor

sys.path.append(os.path.join('..', 'config'))
sys.path.append(os.path.join('..', 'dataset'))
sys.path.append(os.path.join('..', 'utils'))


from config import CONFIG 
from dataset import Places365
from utils import ModelSaver, fetch_label_mappings, eval_one_epoch,collate_fn

DISABLE_TQDM= CONFIG['DISABLE_TQDM']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'google/vit-base-patch16-224-in21k'


def train_model(model,training_params):
    """ 
    Trains the model according to the training parameters
    Args:
        model: Deep Learning model 
        training_parameters: dictionary containing hyperparameters and optimizer
    """
    for epoch in range(1, training_params['epochs']+1):
        start = start_evaluate =  time.time()
        print("-"*100)
        print(f"\nEpoch {epoch}:")
        print("-"*100)
        print("Train")
        
        running_loss= 0
        
        for i, batch in enumerate(tqdm(training_params['train_loader'], disable=DISABLE_TQDM, position=0), start=1):
            
            images = batch['pixel_values'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            training_params['optim'].zero_grad()
            model.train()
            
            with torch.set_grad_enabled(True):
                out = model(images,labels=labels)
                out['loss'].backward()
                training_params['optim'].step()

                batch_loss =  out['loss'].item()
                running_loss += batch_loss
                wandb.log({'batch_loss': batch_loss, 'epoch_loss': running_loss / i, 'epoch': epoch, 'batch': i})
            
            if i%training_params['evaluate_every'] == 0:
                valid_loss, metrics = eval_one_epoch(model,training_params['valid_loader'], training_params['compute_metrics'])
                metrics['valid_loss'] = valid_loss
                training_params['model_saver'].save(model,valid_loss)

                wandb.log(metrics)
                wandb.log({'duration': time.time() - start})
                
                
        wandb.log({'epoch_duration': time.time() - start})

def main():
    """
    Finetunes the Vision Transformer model on the Places365 dataset and logs the metrics in wandb dashboard.
    """
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")
    idx_to_label,label_to_idx = fetch_label_mappings("../data/label_to_idx.json")
    num_labels = len(idx_to_label)

    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
    dataset = Places365(train_df.path.values,train_df.label_id.values,feature_extractor)
    
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, \
        [len(dataset)-len(test_df), len(test_df)], \
        generator=torch.Generator().manual_seed(42))

    wandb_config = dict(
        epochs = 1,
        lr = 2e-4,
        batch_size = 32,
        evaluate_every = 5000,
        optimizer = 'AdamW',
        architecture = MODEL_NAME,
        dataset_id = "places_365",
        infra = "CARC")


    wandb.init(
        dir= "/scratch2/knarasim/places365",
        project="Places365",
        notes="Finetune Vision Transformer",
        tags=["VIT","Places365"],
        config=wandb_config
    )

    train_loader = DataLoader(train_dataset, batch_size=wandb_config['batch_size'], shuffle=True, num_workers=8,collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=wandb_config['batch_size'], shuffle=True, num_workers=8,collate_fn=collate_fn)
    labels = list(label_to_idx.keys())


    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=idx_to_label,
        label2id=label_to_idx,
        output_attentions=False,
        output_hidden_states = False
    )

    model = model.to(DEVICE)

    compute_metrics = {
        'Accuracy Top 1': Accuracy().to(DEVICE),
        'Accuracy Top 5': Accuracy(top_k=5).to(DEVICE),
        'Precision' : Precision(average='macro', num_classes=num_labels).to(DEVICE),
        'Recall' : Recall(average='macro', num_classes=num_labels).to(DEVICE),
        'F1-Score': F1Score(num_classes=num_labels).to(DEVICE),
        'AUROC': AUROC(num_classes=num_labels).to(DEVICE)
    }

    training_params = dict(
        train_loader = train_loader,
        valid_loader = valid_loader,
        epochs = wandb_config['epochs'],
        optim = AdamW(model.parameters(), lr=wandb_config['lr']),
        lossf = nn.CrossEntropyLoss(),
        model_saver = ModelSaver("vit_trained.pt", wandb.run.dir), # Saves the model in wandb run directory
        evaluate_every = wandb_config['evaluate_every'],
        compute_metrics = compute_metrics
    )

    train_model(model, training_params)

if __name__ == '__main__':
    main()