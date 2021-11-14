from math import log
import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
import datetime
import matplotlib.pyplot as plt
import torch.distributed as dist

'''
Return time string with YYYY-mm-dd HH:MM:SS format
'''
def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# def save_config_file(model_checkpoints_folder, args):
#     if not os.path.exists(model_checkpoints_folder):
#         os.makedirs(model_checkpoints_folder)
#         with open(os.path.join(model_checkpoints_folder, 'config.yml'),
#                   'w') as outfile:
#             yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

'''
Get a TensorBoard SummaryWriter to record training procedure.
'''
def get_writer(log_dir):
    log_time = get_time()
    log_dir = log_dir + log_time
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    return writer

'''
Save matploit image
'''
def save_plot(xdata, ydata, xlabel, ylabel, title, location):
    fig, ax = plt.subplots()
    ax.plot(xdata, ydata)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()
    plt.savefig(f"{location}_{get_time()}")


'''
Get mean value of experiment data among all processes in DDP training
'''
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / dist.get_world_size()
    return rt