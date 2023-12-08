import os
import numpy as np

from PIL import Image

from torch.utils.data import Dataset


class PathologyDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_files[index])
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img[np.where(img == 255)] = 0
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img
