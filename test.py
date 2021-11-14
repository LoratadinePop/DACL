import torch

checkpoint = torch.load('result/checkpoint/simclr/old/checkpoint_0010.pth.tar')
state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):
    print(k)


print(state_dict)