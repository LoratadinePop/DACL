import torch

means = torch.randn([10, 2])
covar = torch.tensor([[[1., 0.],
                       [0., 1.]], [[1., 0.],
                                   [0., 1.]], [[1., 0.],
                                               [0., 1.]], [[1., 0.],
                                                           [0., 1.]], [[1., 0.],
                                                                       [0., 1.]], [[1., 0.],
                                                                                   [0., 1.]], [[1., 0.],
                                                                                               [0., 1.]], [[1., 0.],
                                                                                                           [0., 1.]], [[1., 0.],
                                                                                                                       [0., 1.]], [[1., 0.],
                                                                                                                                   [0., 1.]]])

dis_tmp = torch.distributions.multivariate_normal.MultivariateNormal(
    loc=means, covariance_matrix=covar)
data = torch.tensor([1, 2])
data1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(data)
print(dis_tmp.log_prob(data))
print(data1)
print(dis_tmp.log_prob(data1[:, None, :]))
