# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from ofa.utils import calc_learning_rate, build_optimizer
from ofa.fer.data_providers import AffectNetDataProvider

__all__ = ["RunConfig", "AffectNetRunConfig", "DistributedAffectNetRunConfig"]

class RunConfig:
    def __init__(
        self,
        n_epochs=25,  # Reduced for small dataset
        init_lr=0.01,  # Lower for stability
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        dataset="affectnet",  # Changed for AffectNet
        train_batch_size=64,  # Reduced for Jetson
        test_batch_size=128,  # Reduced for Jetson
        valid_size=0.3,  # 30% for validation
        opt_type="sgd",
        opt_param=None,
        weight_decay=1e-4,  # Increased for regularization
        label_smoothing=0.1,
        no_decay_keys=None,
        mixup_alpha=0.2,  # Added for data augmentation
        model_init="he_fout",
        validation_frequency=1,
        print_frequency=10,
    ):
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.mixup_alpha = mixup_alpha

        self.model_init = model_init
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith("_"):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """adjust learning of a given optimizer and return the new learning rate"""
        new_lr = calc_learning_rate(
            epoch, self.init_lr, self.n_epochs, batch, nBatch, self.lr_schedule_type
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def warmup_adjust_learning_rate(
        self, optimizer, T_total, nBatch, epoch, batch=0, warmup_lr=0
    ):
        T_cur = epoch * nBatch + batch + 1
        warmup_fraction = 0.05  # Warmup for 5% of iterations
        if T_cur <= T_total * warmup_fraction:
            new_lr = T_cur / (T_total * warmup_fraction) * (self.init_lr - warmup_lr) + warmup_lr
        else:
            new_lr = calc_learning_rate(
                epoch - (T_total * warmup_fraction / nBatch),
                self.init_lr,
                self.n_epochs,
                batch,
                nBatch,
                self.lr_schedule_type
            )
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    @property
    def data_provider(self):
        raise NotImplementedError

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    def random_sub_train_loader(
        self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        return self.data_provider.build_sub_train_loader(
            n_images, batch_size, num_worker, num_replicas, rank
        )

    def build_optimizer(self, net_params):
        return build_optimizer(
            net_params,
            self.opt_type,
            self.opt_param,
            self.init_lr,
            self.weight_decay,
            self.no_decay_keys,
        )

class AffectNetRunConfig(RunConfig):
    def __init__(
        self,
        n_epochs=25,  # Reduced for small dataset
        init_lr=0.01,  # Lower for stability
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        dataset="affectnet",  # Changed for AffectNet
        train_batch_size=64,  # Reduced for Jetson
        test_batch_size=128,  # Reduced for Jetson
        valid_size=0.3,  # 30% for validation
        opt_type="sgd",
        opt_param=None,
        weight_decay=1e-4,  # Increased for regularization
        label_smoothing=0.1,
        no_decay_keys=None,
        mixup_alpha=0.2,  # Added for data augmentation
        model_init="he_fout",
        validation_frequency=1,
        print_frequency=10,
        n_worker=2,  # Reduced for Jetson
        resize_scale=0.5,  # Match affectnet.py
        distort_color="torch",  # Match affectnet.py
        image_size=[64, 80, 96, 128],  # For Progressive Shrinking
        **kwargs
    ):
        super(AffectNetRunConfig, self).__init__(
            n_epochs,
            init_lr,
            lr_schedule_type,
            lr_schedule_param,
            dataset,
            train_batch_size,
            test_batch_size,
            valid_size,
            opt_type,
            opt_param,
            weight_decay,
            label_smoothing,
            no_decay_keys,
            mixup_alpha,
            model_init,
            validation_frequency,
            print_frequency,
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size

    @property
    def data_provider(self):
        if self.__dict__.get("_data_provider", None) is None:
            if self.dataset == AffectNetDataProvider.name():
                DataProviderClass = AffectNetDataProvider
            else:
                raise NotImplementedError
            self.__dict__["_data_provider"] = DataProviderClass(
                train_batch_size=self.train_batch_size,
                test_batch_size=self.test_batch_size,
                valid_size=self.valid_size,
                n_worker=self.n_worker,
                resize_scale=self.resize_scale,
                distort_color=self.distort_color,
                image_size=self.image_size,
            )
        return self.__dict__["_data_provider"]

class DistributedAffectNetRunConfig(AffectNetRunConfig):
    def __init__(
        self,
        n_epochs=25,  # Reduced for small dataset
        init_lr=0.01,  # Lower for stability
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        dataset="affectnet",  # Changed for AffectNet
        train_batch_size=64,  # Suitable for Jetson
        test_batch_size=128,  # Suitable for Jetson
        valid_size=0.3,  # 30% for validation
        opt_type="sgd",
        opt_param=None,
        weight_decay=1e-4,  # Increased for regularization
        label_smoothing=0.1,
        no_decay_keys=None,
        mixup_alpha=0.2,  # Added for data augmentation
        model_init="he_fout",
        validation_frequency=1,
        print_frequency=10,
        n_worker=2,  # Reduced for Jetson
        resize_scale=0.5,  # Match affectnet.py
        distort_color="torch",  # Match affectnet.py
        image_size=[64, 80, 96, 128],  # For Progressive Shrinking
        **kwargs
    ):
        super(DistributedAffectNetRunConfig, self).__init__(
            n_epochs,
            init_lr,
            lr_schedule_type,
            lr_schedule_param,
            dataset,
            train_batch_size,
            test_batch_size,
            valid_size,
            opt_type,
            opt_param,
            weight_decay,
            label_smoothing,
            no_decay_keys,
            mixup_alpha,
            model_init,
            validation_frequency,
            print_frequency,
            n_worker,
            resize_scale,
            distort_color,
            image_size,
            **kwargs
        )

        self._num_replicas = kwargs["num_replicas"]
        self._rank = kwargs["rank"]

    @property
    def data_provider(self):
        if self.__dict__.get("_data_provider", None) is None:
            if self.dataset == AffectNetDataProvider.name():
                DataProviderClass = AffectNetDataProvider
            else:
                raise NotImplementedError
            self.__dict__["_data_provider"] = DataProviderClass(
                train_batch_size=self.train_batch_size,
                test_batch_size=self.test_batch_size,
                valid_size=self.valid_size,
                n_worker=self.n_worker,
                resize_scale=self.resize_scale,
                distort_color=self.distort_color,
                image_size=self.image_size,
                num_replicas=self._num_replicas,
                rank=self._rank,
            )
        return self.__dict__["_data_provider"]
