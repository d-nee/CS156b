import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# ToTensor automatically detects ndarray as image, normalizes to 0, 1
class CheXpertDataset(Dataset):
    def __init__(self, img_dir, label_file_path):
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.img_dir = img_dir
        self.Y = torch.tensor(np.load(label_file_path), dtype=torch.long)
        self.num_samples = len(self.Y)

    def __getitem__(self, index):
        np_img = np.load(f"{self.img_dir}/img_{index}.npy")
        return self.transform(np_img), self.Y[index]

    def __len__(self):
        return self.num_samples
