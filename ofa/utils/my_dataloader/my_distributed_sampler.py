import math
import torch
from torch.utils.data.distributed import DistributedSampler

__all__ = ["MyDistributedSampler", "WeightedDistributedSampler"]

class MyDistributedSampler(DistributedSampler):
    """Allow Subset Sampler in Single or Distributed Training"""

    def __init__(
        self, dataset, num_replicas=1, rank=0, shuffle=True, sub_index_list=None
    ):
        super(MyDistributedSampler, self).__init__(dataset, num_replicas, rank, shuffle)
        self.sub_index_list = (
            sub_index_list if sub_index_list is not None else list(range(len(dataset)))
        )

        self.num_samples = int(
            math.ceil(len(self.sub_index_list) * 1.0 / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.sub_index_list), generator=g).tolist()

        indices += indices[: (self.total_size - len(indices))]
        indices = [self.sub_index_list[i] for i in indices]
        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

class WeightedDistributedSampler(DistributedSampler):
    """Allow Weighted Random Sampling in Single or Distributed Training"""

    def __init__(
        self,
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        weights=None,
        replacement=False,
    ):
        super(WeightedDistributedSampler, self).__init__(
            dataset, num_replicas, rank, shuffle
        )
        if weights is None and hasattr(dataset, "labels"):
            # Compute weights based on class distribution
            labels = dataset.labels
            class_counts = torch.bincount(torch.tensor(labels, dtype=torch.int64))
            class_weights = 1.0 / (class_counts.float() + 1e-6)
            self.weights = class_weights[labels].double()
        else:
            self.weights = (
                torch.as_tensor(weights, dtype=torch.double)
                if weights is not None
                else torch.ones(len(dataset), dtype=torch.double)
            )
        self.replacement = replacement

    def __iter__(self):
        if self.weights is None or torch.all(self.weights == 1.0):
            return super(WeightedDistributedSampler, self).__iter__()
        else:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            if self.shuffle:
                indices = torch.multinomial(
                    self.weights, len(self.dataset), self.replacement, generator=g
                ).tolist()
            else:
                indices = list(range(len(self.dataset)))

            indices += indices[: (self.total_size - len(indices))]
            assert len(indices) == self.total_size

            indices = indices[self.rank : self.total_size : self.num_replicas]
            assert len(indices) == self.num_samples

            return iter(indices)
