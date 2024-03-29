from math import frexp
import torch
from torch.optim import optimizer, lr_scheduler
import torchvision
from torchvision.models import resnet18

def linear_warmup(optimizer, lr_min, lr_max, epoch_warmup_totol, epoch_cur):
    div = (lr_max - lr_min) / epoch_warmup_totol
    lr = lr_min + div * epoch_cur
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# lr = 5e-3
# model = resnet18(pretrained=False)
# optimizer = torch.optim.SGD(model.parameters(), lr)
# # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=30, T_mult=1, eta_min=0, last_epoch=-1, verbose=True)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=90, eta_min=0, last_epoch=-1, verbose=True)
# for epoch in range(100):
#     # print(optimizer.state_dict()['param_groups'][0]['lr'])
#     if epoch < 10:
#         linear_warmup(optimizer, 0, 1e-3, 10, epoch+1)
#     else:
#         scheduler.step()
#     print(optimizer.state_dict()['param_groups'][0]['lr'])
#     for i in range(50):
#         optimizer.step()

min_lr = 1e-10
max_lr = 1
model = resnet18(pretrained=False)
optimizer = torch.optim.SGD(model.parameters(), min_lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=190, eta_min=min_lr, last_epoch=-1, verbose=True)
list = []
for epoch in range(1, 201):
    if epoch > 10:
        scheduler.step()
    else:
        linear_warmup(optimizer, min_lr, max_lr, 10, epoch)
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    list.append(scheduler.get_lr())
    for i in range(50):
        optimizer.step()
print(len(list))
print(list)