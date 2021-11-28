import torch
device = torch.device("cuda:7")
checkpoint = torch.load('./result/checkpoint/simclr/checkpoint_0100.pth.tar', map_location=device)
print(checkpoint)
state_dict = checkpoint['state_dict']
for k in list(state_dict.keys()):
  if k.startswith('backbone.'):
    if k.startswith('backbone') and not k.startswith('backbone.fc'):
      # remove prefix
      state_dict[k[len("backbone."):]] = state_dict[k]
  del state_dict[k]