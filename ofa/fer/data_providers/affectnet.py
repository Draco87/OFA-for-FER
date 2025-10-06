# Once for All: Train One Network and Specialize it for Efficient Deployment
# Adapted for AffectNet Facial Expression Recognition
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import warnings
import os
import math
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

from .base_provider import DataProvider
from ofa.utils.my_dataloader import MyRandomResizedCrop, WeightedDistributedSampler

__all__ = ["AffectNetDataProvider"]

class AffectNetDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        # List all image files
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.image_ids = [os.path.splitext(f)[0] for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load expression label from .npy
        img_id = self.image_ids[idx]
        exp_path = os.path.join(self.annotation_dir, f"{img_id}_exp.npy")
        label = int(np.load(exp_path))  # Integer label (0-7)

        if self.transform:
            image = self.transform(image)
        
        return image, label

class AffectNetDataProvider(DataProvider):
    DEFAULT_PATH = "/dataset/affectnet"

    def __init__(
        self,
        save_path=None,
        train_batch_size=64,  # Reduced for 2.91K dataset and Jetson
        test_batch_size=128,
        valid_size=None,
        n_worker=2,  # Reduced for Jetson
        resize_scale=0.5,
        distort_color="torch",
        image_size=[64, 80, 96, 128],  # FER-specific resolutions
        num_replicas=None,
        rank=None,
    ):
        warnings.filterwarnings("ignore")
        self._save_path = save_path
        self.image_size = image_size
        self.distort_color = distort_color
        self.resize_scale = resize_scale

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            from ofa.utils.my_dataloader.my_data_loader import MyDataLoader
            assert isinstance(self.image_size, list)
            self.image_size.sort()
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)
            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
            self.active_img_size = max(self.image_size)
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader

        train_dataset = self.train_dataset(self.build_train_transform())

        # Compute class weights for imbalanced sampling
        class_weights = self.compute_class_weights()

        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset) * valid_size)

            valid_dataset = self.train_dataset(valid_transforms)
            train_indexes, valid_indexes = self.random_sample_valid_set(
                len(train_dataset), valid_size
            )

            if num_replicas is not None:
                train_sampler = WeightedDistributedSampler(
                    train_dataset, num_replicas, rank, True, weights=class_weights
                )
                valid_sampler = WeightedDistributedSampler(
                    valid_dataset, num_replicas, rank, True
                )
            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                sampler=train_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                sampler=valid_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
        else:
            if num_replicas is not None:
                train_sampler = WeightedDistributedSampler(
                    train_dataset, num_replicas, rank, True, weights=class_weights
                )
                self.train = train_loader_class(
                    train_dataset,
                    batch_size=train_batch_size,
                    sampler=train_sampler,
                    num_workers=n_worker,
                    pin_memory=True,
                )
            else:
                self.train = train_loader_class(
                    train_dataset,
                    batch_size=train_batch_size,
                    shuffle=True,
                    num_workers=n_worker,
                    pin_memory=True,
                )
            self.valid = None

        test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas, rank
            )
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                sampler=test_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=True,
            )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return "affectnet"

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 8  # Happy, Sad, Angry, Surprise, Fear, Disgust, Contempt, Neutral

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser("~/dataset/affectnet")
        return self._save_path

    @property
    def data_url(self):
        raise ValueError("unable to download %s" % self.name())

    def train_dataset(self, _transforms):
        return AffectNetDataset(
            image_dir=os.path.join(self.save_path, "train/images"),
            annotation_dir=os.path.join(self.save_path, "train/annotations"),
            transform=_transforms
        )

    def test_dataset(self, _transforms):
        return AffectNetDataset(
            image_dir=os.path.join(self.save_path, "val/images"),
            annotation_dir=os.path.join(self.save_path, "val/annotations"),
            transform=_transforms
        )

    @property
    def train_path(self):
        return os.path.join(self.save_path, "train/images")

    @property
    def valid_path(self):
        return os.path.join(self.save_path, "val/images")

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def compute_class_weights(self):
        # Compute class weights from expression annotations
        annotation_dir = os.path.join(self.save_path, "train/annotations")
        image_files = [f for f in os.listdir(self.train_path) if f.endswith('.jpg')]
        labels = []
        for img_file in image_files:
            img_id = os.path.splitext(img_file)[0]
            exp_path = os.path.join(annotation_dir, f"{img_id}_exp.npy")
            if os.path.exists(exp_path):
                label = int(np.load(exp_path))
                labels.append(label)
        labels = np.array(labels)
        class_counts = np.bincount(labels, minlength=self.n_classes)
        total_samples = len(labels)
        weights = total_samples / (self.n_classes * class_counts)
        weights = weights / weights.sum()
        return weights

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print(
                "Color jitter: %s, resize_scale: %s, img_size: %s"
                % (self.distort_color, self.resize_scale, image_size)
            )

        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop
            print(
                "Use MyRandomResizedCrop: %s, \t %s"
                % MyRandomResizedCrop.get_candidate_image_size(),
                "sync=%s, continuous=%s"
                % (
                    MyRandomResizedCrop.SYNC_DISTRIBUTED,
                    MyRandomResizedCrop.CONTINUOUS,
                ),
            )
        else:
            resize_transform_class = transforms.RandomResizedCrop

        train_transforms = [
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0), ratio=(0.9, 1.1), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            self.normalize,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose(
            [
                transforms.Resize(int(math.ceil(image_size / 0.875))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[
                self.active_img_size
            ] = self.build_valid_transform()
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_sub_train_loader(
        self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        if self.__dict__.get("sub_train_%d" % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = self.train_dataset(
                self.build_train_transform(
                    image_size=self.active_img_size, print_log=False
                )
            )
            class_weights = self.compute_class_weights()
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = WeightedDistributedSampler(
                    new_train_dataset,
                    num_replicas,
                    rank,
                    True,
                    weights=class_weights,
                    np.array(chosen_indexes),
                )
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    chosen_indexes
                )
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset,
                batch_size=batch_size,
                sampler=sub_sampler,
                num_workers=num_worker,
                pin_memory=True,
            )
            self.__dict__["sub_train_%d" % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__["sub_train_%d" % self.active_img_size].append(
                    (images, labels)
                )
        return self.__dict__["sub_train_%d" % self.active_img_size]
