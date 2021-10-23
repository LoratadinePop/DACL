import os, sys
import time

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import MNIST

from torch.utils.data import DataLoader

from model.sgd_gmm import SGDGMM

MNIST_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
])

resnet_18 = resnet18(pretrained=False, progress=True)
in_features = resnet_18.fc.in_features
resnet_18.fc = nn.Linear(in_features, 2)

torch.cuda.set_device(5)
torch.cuda.empty_cache()
device = torch.device("cuda")

train_set = MNIST(root="dataset",
                  train=False,
                  transform=MNIST_transform,
                  download=True)

loader = DataLoader(dataset=train_set,
                    batch_size=64,
                    shuffle=True,
                    drop_last=True)

gmm = SGDGMM(10,
             2,
             lr=0.01,
             batch_size=512,
             epochs=1,
             backbone_model=resnet_18,
             device=device,
             restarts=1,
             k_means_iter=1)
gmm.fit(train_set)

resnet_18.to(device)
for batch_data, _ in loader:
    batch_data = batch_data.to(device)
    start = time.time()
    batch_data = resnet_18(batch_data)
    samples = gmm.sample(batch_data)
    print(samples.shape)
    print("Duration: {}".format(time.time() - start))
