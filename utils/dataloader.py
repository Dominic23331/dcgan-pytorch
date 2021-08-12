import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np


class DCGAN_dataloader(Dataset):
    def __init__(self, csv_path, transforms=None):
        super(DCGAN_dataloader, self).__init__()
        self.transforms = transforms
        self.data = pd.read_csv(csv_path)

    def __getitem__(self, item):
        img_path = self.data["img"][item]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.data["img"])


def transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
