import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from collections import OrderedDict


FMODEL_FILE_PATH = "C:\\Users\\danie\\source\\repos\\CS156b\\model_weights\\frontal\\densenet121_class"
LMODEL_FILE_PATH = "C:\\Users\\danie\\source\\repos\\CS156b\\model_weights\\lateral\\densenet121_class"


TEST_IDS = "D:\\CS156b\\solution_ids.csv"

TEST_DIR = "D:\\CS156b\\solution_img_npy"

BATCH_SIZE = 256
NUM_CORES = 10
IMG_SIZE = 128

class CheXpertDataset(Dataset):
    def __init__(self, img_dir, num_images):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.img_dir = img_dir
        self.num_samples = num_images

    def __getitem__(self, index):
        np_img = np.load(f"{self.img_dir}/img{index}.npy")
        return self.transform(np_img)

    def __len__(self):
        return self.num_samples

if __name__ == "__main__":
    fmodel_paths = []
    lmodel_paths = []
    for x in range(14):
        fmodel_path = FMODEL_FILE_PATH + str(x) + "_best.pt"
        lmodel_path = LMODEL_FILE_PATH + str(x) + "_best.pt"
        fmodel_paths.append(fmodel_path)
        lmodel_paths.append(lmodel_path)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    with open(TEST_IDS, "r") as file:
        test_lines = file.read().split('\n')
    lines_split = [x.split(",") for x in test_lines[1:]]

    print(lines_split[0][0])
    print(lines_split[-1][0])
    n_test = len(lines_split)

    dataset = CheXpertDataset(TEST_DIR, n_test)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_CORES)

    denseNetArgs = {
        "bn_size": 4,
        "drop_rate": .05,
        "num_classes": 1,
        "memory_efficient": False
    }

    fmodels = []
    lmodels = []
    labels = np.zeros((n_test, 14))
    for trained_model in fmodel_paths:
        model = torchvision.models.densenet121(**denseNetArgs)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("lin", model.classifier),
                    ("tanh", torch.nn.Tanh()),
                ]
            )
        )
        model.load_state_dict(torch.load(trained_model))
        model.to(device)
        model.eval()
        fmodels.append(model)

    for trained_model in lmodel_paths:
        model = torchvision.models.densenet121(**denseNetArgs)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("lin", model.classifier),
                    ("tanh", torch.nn.Tanh()),
                ]
            )
        )
        model.load_state_dict(torch.load(trained_model))
        model.to(device)
        model.eval()
        lmodels.append(model)

    print("Models Loaded")
    # labels = np.load("D:\\CS156b\\final_eval_submission.npy")
    img_idx = 0
    with torch.no_grad():
        for j, data in enumerate(tqdm(testloader)):
            inputs = data.to(device)
            for i in range(14):
                foutputs = fmodels[i](inputs)
                loutputs = lmodels[i](inputs)
                for k in range(len(inputs)):
                    if "frontal" in lines_split[img_idx + k][1]:
                        labels[img_idx + k][i] = foutputs[k].cpu().numpy()[0]
                    else:
                        labels[img_idx + k][i] = loutputs[k].cpu().numpy()[0]
            img_idx += len(inputs)
    print(img_idx)
    np.save("D:\\CS156b\\final_submission.npy", labels)
