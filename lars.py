import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Iterable, Optional, Callable, Tuple
from torch import nn

"""
    We recommend using create_optimizer_lars and setting bn_bias_separately=True 
    instead of using class Lars directly, which helps LARS skip parameters
    in BatchNormalization and bias, and has better performance in general.
    Polynomial Warmup learning rate decay is also helpful for better performance in general.
"""


def create_optimizer_lars(model, lr, momentum=0.9, weight_decay=0.0, bn_bias_separately=True, epsilon=0.0):
    if bn_bias_separately:
        optimizer = Lars([
            dict(params=get_common_parameters(model, exclude_func=get_norm_bias_parameters)),
            dict(params=get_norm_bias_parameters(model), weight_decay=0, lars=False)],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon)
    else:
        optimizer = Lars(model.parameters(),
                         lr=lr,
                         momentum=momentum,
                         weight_decay=weight_decay,
                         epsilon=epsilon)
    return optimizer


class Lars(Optimizer):
    r"""Implements the LARS optimizer from `"Large batch training of convolutional networks"
    <https://arxiv.org/pdf/1708.03888.pdf>`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eeta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr=1e-3,
            momentum=0,
            eeta=1e-3,
            weight_decay=0,
            epsilon=0.0
    ) -> None:
        if not isinstance(lr, float) or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eeta <= 0 or eeta > 1:
            raise ValueError("Invalid eeta value: {}".format(eeta))
        if epsilon < 0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, eeta=eeta, epsilon=epsilon, lars=True)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eeta = group['eeta']
            lr = group['lr']
            lars = group['lars']
            eps = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                decayed_grad = p.grad
                scaled_lr = lr
                if lars:
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(p.grad)
                    trust_ratio = torch.where(
                        w_norm > 0 and g_norm > 0,
                        eeta * w_norm / (g_norm + weight_decay * w_norm + eps),
                        torch.ones_like(w_norm)
                    )
                    trust_ratio.clamp_(0.0, 50)
                    scaled_lr *= trust_ratio.item()
                    if weight_decay != 0:
                        decayed_grad = decayed_grad.add(p, alpha=weight_decay)
                decayed_grad = torch.clamp(decayed_grad, -10.0, 10.0)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            decayed_grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(decayed_grad)
                    decayed_grad = buf

                p.add_(decayed_grad, alpha=-scaled_lr)

        return loss


"""
    Functions which help to skip bias and BatchNorm
"""
BN_CLS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def get_parameters_from_cls(module, cls_):
    def get_members_fn(m):
        if isinstance(m, cls_):
            return m._parameters.items()
        else:
            return dict()

    named_parameters = module._named_members(get_members_fn=get_members_fn)
    for name, param in named_parameters:
        yield param


def get_norm_parameters(module):
    return get_parameters_from_cls(module, (nn.LayerNorm, *BN_CLS))


def get_bias_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters and 'bias' in name:
            yield param


def get_norm_bias_parameters(module):
    for param in get_norm_parameters(module):
        yield param
    for param in get_bias_parameters(module, exclude_func=get_norm_parameters):
        yield param


def get_common_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters:
            yield param



# """
# LARS: Layer-wise Adaptive Rate Scaling

# Converted from TensorFlow to PyTorch
# https://github.com/google-research/simclr/blob/master/lars_optimizer.py
# """

# import torch
# from torch.optim.optimizer import Optimizer, required
# import re

# EETA_DEFAULT = 0.001


# class LARS(Optimizer):
#     """
#     Layer-wise Adaptive Rate Scaling for large batch training.
#     Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
#     I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
#     """

#     def __init__(
#         self,
#         params,
#         lr=required,
#         momentum=0.9,
#         use_nesterov=False,
#         weight_decay=0.0,
#         exclude_from_weight_decay=None,
#         exclude_from_layer_adaptation=None,
#         classic_momentum=True,
#         eeta=EETA_DEFAULT,
#     ):
#         """Constructs a LARSOptimizer.
#         Args:
#         lr: A `float` for learning rate.
#         momentum: A `float` for momentum.
#         use_nesterov: A 'Boolean' for whether to use nesterov momentum.
#         weight_decay: A `float` for weight decay.
#         exclude_from_weight_decay: A list of `string` for variable screening, if
#             any of the string appears in a variable's name, the variable will be
#             excluded for computing weight decay. For example, one could specify
#             the list like ['batch_normalization', 'bias'] to exclude BN and bias
#             from weight decay.
#         exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
#             for layer adaptation. If it is None, it will be defaulted the same as
#             exclude_from_weight_decay.
#         classic_momentum: A `boolean` for whether to use classic (or popular)
#             momentum. The learning rate is applied during momeuntum update in
#             classic momentum, but after momentum for popular momentum.
#         eeta: A `float` for scaling of learning rate when computing trust ratio.
#         name: The name for the scope.
#         """

#         self.epoch = 0
#         defaults = dict(
#             lr=lr,
#             momentum=momentum,
#             use_nesterov=use_nesterov,
#             weight_decay=weight_decay,
#             exclude_from_weight_decay=exclude_from_weight_decay,
#             exclude_from_layer_adaptation=exclude_from_layer_adaptation,
#             classic_momentum=classic_momentum,
#             eeta=eeta,
#         )

#         super(LARS, self).__init__(params, defaults)
#         self.lr = lr
#         self.momentum = momentum
#         self.weight_decay = weight_decay
#         self.use_nesterov = use_nesterov
#         self.classic_momentum = classic_momentum
#         self.eeta = eeta
#         self.exclude_from_weight_decay = exclude_from_weight_decay
#         # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
#         # arg is None.
#         if exclude_from_layer_adaptation:
#             self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
#         else:
#             self.exclude_from_layer_adaptation = exclude_from_weight_decay

#     def step(self, epoch=None, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure()

#         if epoch is None:
#             epoch = self.epoch
#             self.epoch += 1

#         for group in self.param_groups:
#             weight_decay = group["weight_decay"]
#             momentum = group["momentum"]
#             eeta = group["eeta"]
#             lr = group["lr"]

#             for p in group["params"]:
#                 if p.grad is None:
#                     continue

#                 param = p.data
#                 grad = p.grad.data

#                 param_state = self.state[p]

#                 # TODO: get param names
#                 # if self._use_weight_decay(param_name):
#                 grad += self.weight_decay * param

#                 if self.classic_momentum:
#                     trust_ratio = 1.0

#                     # TODO: get param names
#                     # if self._do_layer_adaptation(param_name):
#                     w_norm = torch.norm(param)
#                     g_norm = torch.norm(grad)

#                     device = g_norm.get_device()
#                     trust_ratio = torch.where(
#                         w_norm.gt(0),
#                         torch.where(
#                             g_norm.gt(0),
#                             (self.eeta * w_norm / g_norm),
#                             torch.Tensor([1.0]).to(device),
#                         ),
#                         torch.Tensor([1.0]).to(device),
#                     ).item()

#                     scaled_lr = lr * trust_ratio
#                     if "momentum_buffer" not in param_state:
#                         next_v = param_state["momentum_buffer"] = torch.zeros_like(
#                             p.data
#                         )
#                     else:
#                         next_v = param_state["momentum_buffer"]

#                     next_v.mul_(momentum).add_(scaled_lr, grad)
#                     if self.use_nesterov:
#                         update = (self.momentum * next_v) + (scaled_lr * grad)
#                     else:
#                         update = next_v

#                     p.data.add_(-update)
#                 else:
#                     raise NotImplementedError

#         return loss

#     def _use_weight_decay(self, param_name):
#         """Whether to use L2 weight decay for `param_name`."""
#         if not self.weight_decay:
#             return False
#         if self.exclude_from_weight_decay:
#             for r in self.exclude_from_weight_decay:
#                 if re.search(r, param_name) is not None:
#                     return False
#         return True

#     def _do_layer_adaptation(self, param_name):
#         """Whether to do layer-wise learning rate adaptation for `param_name`."""
#         if self.exclude_from_layer_adaptation:
#             for r in self.exclude_from_layer_adaptation:
#                 if re.search(r, param_name) is not None:
#                     return False
#         return True
