import torch

x = torch.Tensor([1,2,3])

y = torch.stack([x,x])

print(torch.stack([y,y,y]))
