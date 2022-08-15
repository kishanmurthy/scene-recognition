import sys
import os
import json
import torch
import torchmetrics
from tqdm import tqdm

from config import CONFIG 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelSaver:
    """
    Class to save the best model
    Check the validation loss to determine the saving of the model
    """
    def __init__(self, model_name, path=''):
        self.best_valid_loss = float('inf')
        self.path = path
        self.model_name = model_name
        
    def save(self, model, current_valid_loss):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            torch.save(model, os.path.join(self.path,self.model_name))
            print(f"Saved Model. Best validation loss: {self.best_valid_loss}")

    def fetch(self):
        return torch.load(os.path.join(self.path,self.model_name))


def fetch_label_mappings(file_path):
    """
    Reads the label mapping form json file and created label_to_idx and idx_to_label mapping
    Args:
        file_path: path of the file
    Returns:
        idx_to_label: mapping from idx to label
        label_to_idx: mapping form label to idx
    """
    with open(file_path,"r") as f:
       label_to_idx = json.loads(f.read())
       idx_to_label = { idx:label for label, idx in label_to_idx.items()}
       return idx_to_label, label_to_idx


def eval_one_epoch(model,loader,compute_metrics={}):
    """
    Evaluates the model on one epoch of the loader and computes loss and metrics
    Args:
        model: ML model
        loader: Dataloader containing the evaluation dataset
        compute_metrics: Metrics to compute based on the perdiction and labels.
    Returns:
        epoch_loss: Loss 
        metric_result: dictionary of computed_metrics
    """
    running_loss= 0
    running_hits = 0
    total_predictions = 0
    predictions, trues = [], []
    
    for batch in tqdm(loader, disable = CONFIG['DISABLE_TQDM'],position=1):
        images = batch['pixel_values'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        model.eval()
        with torch.set_grad_enabled(False):
            out = model(images,labels=labels)

            predictions.append(out['logits'])
            trues.append(labels)
            running_loss += out['loss'].item()
            
            
    epoch_loss = running_loss / len(loader)

    trues = torch.cat(trues)
    predictions = torch.cat(predictions)

    metrics = {}
    for metric_name, metric in compute_metrics.items():
        metrics[metric_name] = metric(predictions, trues).item()

    return epoch_loss, metrics

def collate_fn(batch):
    """
    Collates the data for the dataloader
    Args:
        batch: list of data from the dataset
    Returns:
        dictionary of concatenated pixel_valus and labels
    """
    
    return {
        'pixel_values': torch.cat([x['pixel_values'] for x in batch], 0),
        'label': torch.tensor([x['label'] for x in batch])
    }

def pretty_print_results(metrics):
    print("Test Results")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name} :  {metric_value}")
