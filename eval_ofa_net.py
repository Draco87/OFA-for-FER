# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import torch
import argparse
from datetime import datetime
from sklearn.metrics import f1_score

from ofa.fer.networks import OFAMobileNetV3
from ofa.fer.run_config import DistributedFERConfig
from ofa.utils.pytorch_utils import get_net_info, cross_entropy_with_label_smoothing
from ofa.utils.my_modules import set_bn_param

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", type=str, default="/dataset/affectnet", help="Path to AffectNet dataset"
)
parser.add_argument(
    "--checkpoint", type=str, default="exp/fer/phase1/checkpoint.pth.tar",
    help="Path to checkpoint (phase1 or phase2)"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for validation"
)
parser.add_argument(
    "--workers", type=int, default=4, help="Number of data loading workers"
)
parser.add_argument(
    "--image_size", type=int, default=64, choices=[64, 80, 96, 128], help="Input image size"
)
parser.add_argument(
    "--subnet", type=str, default="random", help="Subnet config (random or ks,e,d)"
)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    # Run config
    run_config = DistributedFERConfig(
        test_batch_size=args.batch_size,
        n_worker=args.workers,
        image_size=args.image_size,
        n_classes=8,
        num_replicas=1,
        rank=0,
        label_smoothing=0.1
    )

    # Build network
    net = OFAMobileNetV3(
        n_classes=8,
        bn_param=(0.1, 1e-5),
        dropout_rate=0.1,
        base_stage_width="proxyless",
        width_mult=1.0,
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[2, 3, 4],
    ).cuda()
    set_bn_param(net, momentum=0.1, eps=1e-5, ws_eps=1e-5)

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location="cuda:0")
        net.load_state_dict(state_dict["state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint {args.checkpoint} not found")

    # Set subnet
    if args.subnet == "random":
        net.sample_active_subnet()
    else:
        ks, e, d = map(int, args.subnet.split(","))
        net.set_active_subnet(ks=ks, e=e, d=d)
    subnet = net.get_active_subnet(preserve_weight=True)

    # Run manager
    class RunManager:
        def __init__(self, path, net, run_config):
            self.path = path
            self.net = net
            self.run_config = run_config
            self.log_file = os.path.join(path, "eval_log.txt")
            os.makedirs(self.path, exist_ok=True)

        def write_log(self, log_str, prefix):
            log_str = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {prefix}: {log_str}"
            print(log_str)
            with open(self.log_file, "a") as f:
                f.write(log_str + "\n")

        def validate(self, net=None):
            net = self.net if net is None else net
            net.eval()
            data_loader = self.run_config.data_provider.test
            total_loss, total_correct, total_samples = 0, 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in data_loader:
                    images, labels = images.cuda(), labels.cuda()
                    outputs = net(images)
                    loss = cross_entropy_with_label_smoothing(outputs, labels, self.run_config.label_smoothing)
                    total_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            avg_loss = total_loss / total_samples
            acc = total_correct / total_samples
            f1 = f1_score(all_labels, all_preds, average="weighted")
            return avg_loss, acc, f1

    # Initialize run manager
    run_manager = RunManager(os.path.dirname(args.checkpoint), subnet, run_config)

    # Profile subnet
    net_info = get_net_info(subnet, input_shape=(3, args.image_size, args.image_size), measure_latency="gpu1", print_info=False)
    run_manager.write_log(
        f"Subnet: {subnet.module_str}\n"
        f"Params: {net_info['params']:.2f}M, FLOPs: {net_info['flops']:.2f}M, "
        f"Latency: {net_info['gpu1 latency']['val']:.2f}ms",
        "profile"
    )

    # Evaluate subnet
    run_manager.write_log("Evaluating subnet...", "info")
    loss, acc, f1 = run_manager.validate(net=subnet)
    run_manager.write_log(
        f"Results: loss={loss:.5f}, accuracy={acc:.4f}, f1_score={f1:.4f}",
        "eval"
    )
