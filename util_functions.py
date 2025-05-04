import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2

class PlantPathologyDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32)
        image = self.transform(image=image_np)['image']

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        else:
            return image


def create_tr_val_loader(train_df, val_df=None, tr_transform=None, val_transform=None):
    train_dataset = PlantPathologyDataset(train_df['image_path'].tolist(), 
                                          train_df['target'].tolist(), tr_transform)
    tr_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    
    if val_df is not None:  
        val_dataset = PlantPathologyDataset(val_df['image_path'].tolist(), 
                                            val_df['target'].tolist(), val_transform)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        return tr_loader, val_loader
    else:
        val_dataset = None
        return tr_loader

    







