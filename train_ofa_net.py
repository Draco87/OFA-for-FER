# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
from datetime import datetime

from ofa.fer.networks import OFAMobileNetV3
from ofa.fer.run_config import DistributedFERConfig
from ofa.utils import MyRandomResizedCrop
from ofa.utils.pytorch_utils import (
    build_optimizer,
    calc_learning_rate,
    get_net_info,
    cross_entropy_with_label_smoothing,
)
from ofa.utils.my_modules import init_models, set_bn_param
from ofa.fer.training.progressive_shrinking import train_elastic_depth, validate

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="exp/fer", help="Experiment directory")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="Training phase")
parser.add_argument("--manual_seed", type=int, default=0, help="Random seed")

# Training hyperparameters
parser.add_argument("--n_epochs", type=int, default=300, help="Number of epochs")
parser.add_argument("--base_lr", type=float, default=0.001, help="Base learning rate")
parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
parser.add_argument("--warmup_lr", type=float, default=-1, help="Warmup learning rate")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
parser.add_argument("--no_decay_keys", type=str, default="bn#bias", help="No decay parameters")

# Network configuration
parser.add_argument("--ks_list", type=str, default="3,5,7", help="Kernel size list")
parser.add_argument("--expand_list", type=str, default="3,4,6", help="Expand ratio list")
parser.add_argument("--depth_list", type=str, default="2,3,4", help="Depth list")
parser.add_argument("--width_mult_list", type=str, default="1.0", help="Width multiplier list")
parser.add_argument("--image_size", type=str, default="64,80,96,128", help="Input image sizes")
parser.add_argument("--base_batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--n_worker", type=int, default=4, help="Number of workers")
parser.add_argument("--resize_scale", type=float, default=0.08, help="Resize scale")
parser.add_argument("--distort_color", type=str, default="tf", help="Color distortion")
parser.add_argument("--bn_momentum", type=float, default=0.1, help="Batch norm momentum")
parser.add_argument("--bn_eps", type=float, default=1e-5, help="Batch norm epsilon")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--validation_frequency", type=int, default=1, help="Validation frequency")
parser.add_argument("--print_frequency", type=int, default=10, help="Print frequency")

args = parser.parse_args()

# Set experiment path for phase
args.path = os.path.join(args.path, f"phase{args.phase}")
args.dynamic_batch_size = 2  # For progressive shrinking
args.lr_schedule_type = "cosine"
args.opt_type = "sgd"
args.opt_param = {"momentum": 0.9, "nesterov": True}
args.model_init = "he_fout"
args.base_stage_width = "proxyless"
args.dy_conv_scaling_mode = 1

if __name__ == "__main__":
    os.makedirs(args.path, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # Image size handling
    args.image_size = [int(img_size) for img_size in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = True
    MyRandomResizedCrop.SYNC_DISTRIBUTED = False

    # Build run config
    args.init_lr = args.base_lr
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 2
    run_config = DistributedFERConfig(
        **args.__dict__, num_replicas=1, rank=0
    )

    # Print run config
    print("Run config:")
    for k, v in run_config.config.items():
        print(f"\t{k}: {v}")

    # Build network
    args.width_mult_list = [float(w) for w in args.width_mult_list.split(",")]
    args.ks_list = [int(ks) for ks in args.ks_list.split(",")]
    args.expand_list = [int(e) for e in args.expand_list.split(",")]
    args.depth_list = [int(d) for d in args.depth_list.split(",")]
    args.width_mult_list = args.width_mult_list[0] if len(args.width_mult_list) == 1 else args.width_mult_list

    net = OFAMobileNetV3(
        n_classes=8,
        bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout,
        base_stage_width=args.base_stage_width,
        width_mult=args.width_mult_list,
        ks_list=args.ks_list,
        expand_ratio_list=args.expand_list,
        depth_list=args.depth_list,
    ).cuda()
    init_models(net, model_init=args.model_init)
    set_bn_param(net, momentum=args.bn_momentum, eps=args.bn_eps, ws_eps=1e-5)

    # Optimizer
    optimizer = build_optimizer(
        net.parameters(),
        opt_type=args.opt_type,
        opt_param=args.opt_param,
        init_lr=args.init_lr,
        weight_decay=args.weight_decay,
        no_decay_keys=args.no_decay_keys,
    )

    # Run manager
    class RunManager:
        def __init__(self, path, net, run_config):
            self.path = path
            self.net = net
            self.run_config = run_config
            self.optimizer = optimizer
            self.best_acc = 0
            self.start_epoch = 0
            self.log_file = os.path.join(path, "training_log.txt")

        def save_model(self, epoch, acc):
            state_dict = {
                "state_dict": self.net.state_dict(),
                "epoch": epoch,
                "best_acc": acc,
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(state_dict, os.path.join(self.path, "checkpoint.pth.tar"))

        def load_model(self, checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cuda:0")
            self.net.load_state_dict(state_dict["state_dict"])
            self.start_epoch = state_dict["epoch"]
            self.best_acc = state_dict.get("best_acc", 0)
            if "optimizer" in state_dict:
                self.optimizer.load_state_dict(state_dict["optimizer"])

        def write_log(self, log_str, prefix):
            log_str = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {prefix}: {log_str}"
            print(log_str)
            with open(self.log_file, "a") as f:
                f.write(log_str + "\n")

        def train_epoch(self, epoch, data_loader, is_test=False):
            self.net.train(not is_test)
            total_loss, total_correct, total_samples = 0, 0, 0
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                outputs = self.net(images)
                loss = cross_entropy_with_label_smoothing(outputs, labels, self.run_config.label_smoothing)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                if i % self.run_config.print_frequency == 0:
                    self.write_log(f"Batch {i}, Loss: {loss.item():.4f}, Acc: {total_correct/total_samples:.4f}", "train")
            return total_loss / total_samples, total_correct / total_samples

    run_manager = RunManager(args.path, net, run_config)

    # Profile network
    net_info = get_net_info(net, input_shape=(3, 64, 64), measure_latency="gpu1", print_info=False)
    run_manager.write_log(
        f"Params: {net_info['params']:.2f}M, FLOPs: {net_info['flops']:.2f}M, "
        f"Latency: {net_info['gpu1 latency']['val']:.2f}ms",
        "profile"
    )

    # Load checkpoint if resuming
    checkpoint_path = os.path.join(args.path, "checkpoint.pth.tar")
    if args.resume and os.path.exists(checkpoint_path):
        run_manager.load_model(checkpoint_path)
        run_manager.write_log(f"Resumed from checkpoint at epoch {run_manager.start_epoch}", "info")

    # Validation settings
    validate_func_dict = {
        "image_size_list": {64},
        "ks_list": sorted(args.ks_list),
        "expand_ratio_list": sorted(args.expand_list),
        "depth_list": sorted(args.depth_list),
    }

    # Training
    if args.phase == 1:
        checkpoint_path = "pretrained/ofa_phase1.pth.tar"
    else:
        checkpoint_path = "pretrained/ofa_phase2.pth.tar"

    if run_manager.start_epoch == 0:
        if os.path.exists(checkpoint_path):
            run_manager.load_model(checkpoint_path)
            acc = validate(run_manager, is_test=True, **validate_func_dict)
            run_manager.write_log(f"Initial validation: {acc:.4f}", "valid")
        else:
            run_manager.write_log("No pretrained checkpoint found, training from scratch", "info")

    train_elastic_depth(
        lambda _run_manager, epoch, is_test: validate(
            _run_manager, epoch, is_test, **validate_func_dict
        ),
        run_manager,
        args,
        validate_func_dict,
    )
