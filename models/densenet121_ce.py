import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from DiseaseDataset import DiseaseDataset

# Runs 33% faster than a Jupyter notebook

if __name__ == "__main__":
    # Classwise Densenet121 with Cross Entropy.
    CLASS_NUM = 6

    IMG_DIR = "D:\\CS156b\\train_img_npy"
    LABEL_FILE_PATH = f"C:\\Users\\danie\\source\\repos\\CS156b\\processing\\disease_splits\\class{CLASS_NUM}.npy"
    STATE_DICT_PATH = f"C:\\Users\\danie\\source\\repos\\CS156b\\model_weights\\densenet121_class{CLASS_NUM}_epoch"

    BATCH_SIZE = 4
    NUM_CORES = 8
    NUM_EPOCHS = 15

    # Optimizer paramters, MOMENTUM is only for SGD
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9

    CLASSES = [
        "Negative",
        "Unsure",
        "Positive"
    ]

    denseNetArgs = {
        "bn_size": 4,
        "drop_rate": .2,
        "num_classes": len(CLASSES),
        "memory_efficient": False
    }

    DISEASES = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices"
    ]

    print("Setup Done")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = DiseaseDataset(IMG_DIR, LABEL_FILE_PATH)
    N = len(dataset)
    print(f"{N} Images")
    trainset, testset = torch.utils.data.random_split(dataset, [N - (N//5), N//5])

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_CORES)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_CORES)

    net = torchvision.models.densenet121(**denseNetArgs)
    net.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    print("Objects Done")

    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        epoch_loss = 0.0
        for i, data in enumerate(tqdm(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data[0].to(device), (data[1] + 1).to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch} Loss: {epoch_loss}")
        epoch_loss = 0.0
        torch.save(net.state_dict(), STATE_DICT_PATH + f"{epoch}.pt", _use_new_zipfile_serialization=False)

        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(tqdm(testloader)):
                images, labels = data[0].to(device), data[1]
                outputs = net(images)
                for j in range(len(images)):
                    test_loss += ((outputs[j][2] - outputs[j][0]) - labels[j]) ** 2
        test_loss /= N//5
        print(f"Test MSE: {test_loss}")
        net.train()

    print('Finished Training')