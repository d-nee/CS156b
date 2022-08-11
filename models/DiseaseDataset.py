import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

IMG_SIZE = 128

# ToTensor automatically detects ndarray as image, normalizes to 0, 1
class DiseaseDataset(Dataset):
    def __init__(self, img_dir, label_file_path, is_float):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.img_dir = img_dir
        data = np.load(label_file_path)
        self.index_map = data[:,0].astype(int)
        if(is_float):
            self.Y = torch.tensor(data[:,1], dtype=torch.float)
        else:
            self.Y = torch.tensor(data[:,1], dtype=torch.long)

    def __getitem__(self, index):
        np_img = np.load(f"{self.img_dir}/img_{self.index_map[index]}.npy")
        return self.transform(np_img), self.Y[index]

    def __len__(self):
        return len(self.Y)