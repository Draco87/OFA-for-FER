# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch
import torch.nn as nn

from .my_modules import MyConv2d
from .common_tools import val2list

__all__ = ["profile"]

def count_convNd(m, _, y):
    cin = m.in_channels
    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    ops_per_element = kernel_ops
    output_elements = y.nelement()
    total_ops = cin * output_elements * ops_per_element // m.groups
    m.total_ops = torch.zeros(1).fill_(total_ops)

def count_linear(m, _, __):
    total_ops = m.in_features * m.out_features
    m.total_ops = torch.zeros(1).fill_(total_ops)

register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    MyConv2d: count_convNd,
    nn.Linear: count_linear,
    nn.Dropout: None,
    nn.Dropout2d: None,
    nn.Dropout3d: None,
    nn.BatchNorm2d: None,
    nn.AdaptiveAvgPool2d: None,
}

def profile(model, input_size, custom_ops=None):
    handler_collection = []
    custom_ops = {} if custom_ops is None else custom_ops

    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return
        m_.register_buffer("total_ops", torch.zeros(1))
        m_.register_buffer("total_params", torch.zeros(1))
        for p in m_.parameters():
            m_.total_params += torch.zeros(1).fill_(p.numel())
        m_type = type(m_)
        fn = None
        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        if fn is not None:
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    original_device = torch.device("cuda:0")
    training = model.training
    model.eval()
    model.apply(add_hooks)

    results = []
    input_sizes = val2list(input_size, 1)
    for size in input_sizes:
        x = torch.zeros(size).to(original_device)
        with torch.no_grad():
            model(x)
        total_ops = 0
        total_params = 0
        for m in model.modules():
            if len(list(m.children())) > 0:
                continue
            total_ops += m.total_ops
            total_params += m.total_params
        results.append((total_ops.item(), total_params.item()))

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()
    return results[0] if len(results) == 1 else results
