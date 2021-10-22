import os, sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import MNIST

from torch.utils.data import DataLoader


from models.sgd_gmm import SGDGMM

MNIST_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
])


model = resnet18(pretrained=False, progress=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)

torch.cuda.set_device(5)
torch.cuda.empty_cache()
device = torch.device("cuda")

train_set = MNIST(root="datasets", train=False,
                  transform=MNIST_transform, download=True)

loader = DataLoader(dataset=train_set, batch_size=64,
                    shuffle=True, drop_last=True)

gmm = SGDGMM(10, 2, lr=0.01, batch_size=768, model=model, device=device)
gmm.fit(train_set)
