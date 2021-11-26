from numpy import mod
from model.resnet_simclr import ResNetSimCLR
import torchvision
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets


train_dataset = datasets.CIFAR10('./dataset', train=True, download=True,
                                  transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=3,
                        num_workers=8, drop_last=True, shuffle=True)


# print(torchvision.models.resnet18(pretrained=False, num_classes=10))

model = ResNetSimCLR(base_model='resnet50', out_dim=10)



for x,_ in train_loader:
    out1 = model.backbone(x)
    out2 = model.mlp(torch.flatten(out1, start_dim=1))
    print(out2)
    print("####")
    print(model(x))
    break