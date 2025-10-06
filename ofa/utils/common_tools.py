# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import numpy as np
import os
import sys
import torch

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

__all__ = [
    "sort_dict",
    "get_same_padding",
    "get_split_list",
    "list_sum",
    "list_mean",
    "list_join",
    "subset_mean",
    "sub_filter_start_end",
    "min_divisible_value",
    "val2list",
    "download_url",
    "write_log",
    "accuracy",
    "AverageMeter",
    "DistributedMetric",
    "DistributedTensor",
]

def sort_dict(src_dict, reverse=False, return_dict=True):
    output = sorted(src_dict.items(), key=lambda x: x[1], reverse=reverse)
    if return_dict:
        return dict(output)
    else:
        return output

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2

def get_split_list(in_dim, child_num, accumulate=False):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    if accumulate:
        for i in range(1, child_num):
            in_dim_list[i] += in_dim_list[i - 1]
    return in_dim_list

def list_sum(x):
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])

def list_mean(x):
    return list_sum(x) / len(x)

def list_join(val_list, sep="\t"):
    return sep.join([str(val) for val in val_list])

def subset_mean(val_list, sub_indexes):
    sub_indexes = val2list(sub_indexes, 1)
    return list_mean([val_list[idx] for idx in sub_indexes])

def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end

def min_divisible_value(n1, v1):
    """make sure v1 is divisible by n1, otherwise decrease v1"""
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1

def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]

def download_url(url, model_dir="/home/jetson/ofa_models/", overwrite=False):
    target_dir = url.split("/")[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        sys.stderr.write("Failed to download from url %s: %s\n" % (url, str(e)))
        return None

def write_log(logs_path="/home/jetson/ofa_logs/", log_str, prefix="valid", should_print=True, mode="a"):
    if not os.path.exists(logs_path):
        os.makedirs(logs_path, exist_ok=True)
    if prefix in ["valid", "test"]:
        with open(os.path.join(logs_path, "valid_console.txt"), mode) as fout:
            fout.write(log_str + "\n")
            fout.flush()
    if prefix in ["valid", "test", "train"]:
        with open(os.path.join(logs_path, "train_console.txt"), mode) as fout:
            if prefix in ["valid", "test"]:
                fout.write("=" * 10)
            fout.write(log_str + "\n")
            fout.flush()
    else:
        with open(os.path.join(logs_path, "%s.txt" % prefix), mode) as fout:
            fout.write(log_str + "\n")
            fout.flush()
    if should_print:
        print(log_str)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@1 for 8-class FER"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
    return [correct_k.mul_(100.0 / batch_size)]

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DistributedMetric(object):
    """Average metrics for single-GPU training"""
    def __init__(self, name):
        self.name = name
        self.sum = 0.0
        self.count = 0

    def update(self, val, delta_n=1):
        self.sum += val * delta_n
        self.count += delta_n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0

class DistributedTensor(object):
    """Store tensors for single-GPU training"""
    def __init__(self, name):
        self.name = name
        self.sum = None
        self.count = 0

    def update(self, val, delta_n=1):
        val = val.detach() * delta_n
        if self.sum is None:
            self.sum = val
        else:
            self.sum += val
        self.count += delta_n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else torch.tensor(0.0)
