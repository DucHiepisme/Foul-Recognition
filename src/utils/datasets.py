import bisect
import warnings
import math
import torch

from torch.utils.data import (
    Dataset,
    IterableDataset
)
from torch.utils.data.dataset import T_co

from typing import Iterable, Sequence


class ConcatDataset(torch.utils.data.dataset.ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__(datasets)
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def get_origin_item(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_origin_item(sample_idx)
    
class Subset(torch.utils.data.dataset.Subset):
    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        super(Subset, self).__init__(dataset, indices)
    
    def get_origin_item(self, idx):
        if isinstance(idx, list):
            return [self.dataset.get_origin_item(self.indices[i]) for i in idx]
        return self.dataset.get_origin_item(self.indices[idx])