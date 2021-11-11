import torch
from model.sgd_gmm import SGDGMMModule

model = SGDGMMModule(10,2,0.001)
print(model)
model.load_state_dict(torch.load("Finish 0.ckpt"))
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor])


model = SGDGMMModule(10,2,0.001)
print(model)
model.load_state_dict(torch.load("Finish 1.ckpt"))
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor])