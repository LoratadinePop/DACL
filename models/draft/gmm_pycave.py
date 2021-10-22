import pycave
import torch
from pycave.bayes.gmm import GMM

data = torch.randn(10000, 16)
print(data.size())

gmm = GMM(num_components=100, num_features=16, covariance='spherical')
gmm.reset_parameters(data)
# history = gmm.fit(data)
# print(history)

samples, components = gmm.sample(100, return_components=True)
print(samples)
print(components)