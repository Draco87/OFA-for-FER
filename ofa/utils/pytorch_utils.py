# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import math
import copy
import time
import torch
import torch.nn as nn

__all__ = [
    "label_smooth",
    "cross_entropy_loss_with_soft_target",
    "cross_entropy_with_label_smoothing",
    "clean_num_batch_tracked",
    "rm_bn_from_net",
    "get_net_device",
    "count_parameters",
    "count_net_flops",
    "measure_net_latency",
    "get_net_info",
    "build_optimizer",
    "calc_learning_rate",
]

def label_smooth(target, n_classes: int, label_smoothing=0.1):
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target

def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))

def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    soft_target = label_smooth(target, pred.size(1), label_smoothing)
    return cross_entropy_loss_with_soft_target(pred, soft_target)

def clean_num_batch_tracked(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.num_batches_tracked is not None:
                m.num_batches_tracked.zero_()

def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x

def get_net_device(net):
    return net.parameters().__next__().device

def count_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params

def count_net_flops(net, data_shape=(1, 3, 64, 64)):
    from .flops_counter import profile
    if isinstance(net, nn.DataParallel):
        net = net.module
    flop, _ = profile(copy.deepcopy(net), data_shape)
    return flop

def measure_net_latency(
    net, l_type="gpu1", fast=True, input_shape=(3, 64, 64), clean=True
):
    if isinstance(net, nn.DataParallel):
        net = net.module
    rm_bn_from_net(net)
    if "gpu" in l_type:
        l_type, batch_size = l_type[:3], int(l_type[3:])
    else:
        batch_size = 1
    data_shape = [batch_size] + list(input_shape)
    if l_type == "cpu":
        n_warmup, n_sample = 5, 10
        if get_net_device(net) != torch.device("cpu"):
            net = copy.deepcopy(net).cpu()
    elif l_type == "gpu":
        n_warmup, n_sample = 5, 10
    else:
        raise NotImplementedError
    images = torch.zeros(data_shape, device=get_net_device(net))
    measured_latency = {"warmup": [], "sample": []}
    net.eval()
    with torch.no_grad():
        for i in range(n_warmup):
            inner_start_time = time.time()
            net(images)
            used_time = (time.time() - inner_start_time) * 1e3
            measured_latency["warmup"].append(used_time)
        outer_start_time = time.time()
        for i in range(n_sample):
            net(images)
        total_time = (time.time() - outer_start_time) * 1e3
        measured_latency["sample"].append((total_time, n_sample))
    return total_time / n_sample, measured_latency

def get_net_info(net, input_shape=(3, 64, 64), measure_latency="gpu1", print_info=False):
    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module
    net_info["params"] = count_parameters(net) / 1e6
    net_info["flops"] = count_net_flops(net, [1] + list(input_shape)) / 1e6
    latency_types = [] if measure_latency is None else measure_latency.split("#")
    for l_type in latency_types:
        latency, measured_latency = measure_net_latency(
            net, l_type, fast=True, input_shape=input_shape, clean=True
        )
        net_info["%s latency" % l_type] = {"val": latency, "hist": measured_latency}
    if print_info:
        print(net)
        print("Total training params: %.2fM" % (net_info["params"]))
        print("Total FLOPs: %.2fM" % (net_info["flops"]))
        for l_type in latency_types:
            print(
                "Estimated %s latency: %.3fms"
                % (l_type, net_info["%s latency" % l_type]["val"])
            )
    return net_info

def build_optimizer(
    net_params, opt_type="sgd", opt_param=None, init_lr=0.001, weight_decay=5e-4, no_decay_keys=None
):
    if no_decay_keys is not None:
        assert isinstance(net_params, list) and len(net_params) == 2
        net_params = [
            {"params": net_params[0], "weight_decay": weight_decay},
            {"params": net_params[1], "weight_decay": 0},
        ]
    else:
        net_params = [{"params": net_params, "weight_decay": weight_decay}]
    if opt_type == "sgd":
        opt_param = {} if opt_param is None else opt_param
        momentum, nesterov = opt_param.get("momentum", 0.9), opt_param.get("nesterov", True)
        optimizer = torch.optim.SGD(
            net_params, init_lr, momentum=momentum, nesterov=nesterov
        )
    else:
        raise NotImplementedError
    return optimizer

def calc_learning_rate(
    epoch, init_lr=0.001, n_epochs=300, batch=0, nBatch=None, lr_schedule_type="cosine"
):
    if lr_schedule_type == "cosine":
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError("do not support: %s" % lr_schedule_type)
    return lr
