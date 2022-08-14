import PIL
import numpy as np
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor


class Places365(Dataset):
    """ 
    Pytorch places 365 dataset

    """
    def __init__(self, paths, labels, feature_extractor):
        """
        Initialize the dataset
        Args:
            paths: image paths of the places
            labels: correspoding labels of the places
            feature_extractor:  extracts feature from the image for the transformer model
        """

        self.img_paths = paths
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get individual input for the transformer model based on idx.
        Args:
            idx: id of the image
        return:
            input: dictionary containing pixel_vales (pytorch tensor) and labels
        """
        image = PIL.Image.open(self.img_paths[idx])
        image_np = np.array(image)
        if len(image_np.shape)==2:
            image_np = np.repeat(image_np[:, :, np.newaxis], 3, axis=2)
        inputs = self.feature_extractor(image_np, return_tensors='pt')
        inputs['label'] = self.labels[idx]
        return inputs