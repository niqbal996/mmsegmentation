import copy
import math
import random
from typing import List, Optional, Sequence, Union

from mmengine.dataset import force_full_init
from mmengine.registry import init_default_scope

from mmseg.registry import DATASETS


@DATASETS.register_module()
class CombinedDataset:
    """Combine one or more source datasets using a fixed per-batch ratio.

    This wrapper is designed for training with strict per-batch source mixing.
    For example, two datasets with ``data_ratio=[0.2, 0.8]`` and
    ``batch_size=10`` produce 2 samples from dataset-0 and 8 samples from
    dataset-1 per batch.

    It can also be used with a single dataset (``data_ratio=[1.0]``) as a
    deterministic subset wrapper controlled by ``subset_ratios``.

    Note:
        To keep strict per-batch composition, use a non-shuffled sampler
        (e.g. ``InfiniteSampler(shuffle=False)``) in the dataloader.

    Args:
        datasets (Sequence[Union[dict, object]]): Source dataset configs or
            built datasets.
        data_ratio (Sequence[float]): Source mixing ratios.
        batch_size (int): Train batch size used by dataloader.
        subset_ratios (Sequence[float], optional): Optional subset ratio per
            source dataset. If provided, each value is clipped to ``(0, 1]``.
        subset_seed (int): Random seed used for subset sampling.
        lazy_init (bool): Whether to skip immediate ``full_init``.
    """

    def __init__(self,
                 datasets: Sequence[Union[dict, object]],
                 data_ratio: Sequence[float],
                 batch_size: int,
                 subset_ratios: Optional[Sequence[float]] = None,
                 subset_seed: int = 42,
                 ensure_mmseg_scope: bool = True,
                 lazy_init: bool = False) -> None:
        if not isinstance(datasets, Sequence) or len(datasets) < 1:
            raise ValueError('datasets must be a non-empty sequence.')
        if not isinstance(data_ratio, Sequence) or len(data_ratio) != len(datasets):
            raise ValueError('data_ratio length must match datasets length.')
        if batch_size <= 0:
            raise ValueError('batch_size must be a positive integer.')

        if ensure_mmseg_scope:
            init_default_scope('mmseg')

        self.datasets = []
        for dataset in datasets:
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            else:
                self.datasets.append(dataset)

        self.batch_size = int(batch_size)
        self._data_ratio = [float(r) for r in data_ratio]
        self._subset_seed = int(subset_seed)

        if subset_ratios is None:
            subset_ratios = [1.0] * len(self.datasets)
        if len(subset_ratios) != len(self.datasets):
            raise ValueError('subset_ratios length must match datasets length.')
        self.subset_ratios = [max(0.0, min(1.0, float(r))) for r in subset_ratios]

        self._fully_initialized = False
        self._slot_to_source: List[int] = self._build_slot_mapping()
        self._source_indices: List[List[int]] = []
        self._source_counts_per_batch: List[int] = [
            self._slot_to_source.count(i) for i in range(len(self.datasets))
        ]
        self._dataset_len = 0
        self._metainfo = {}

        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def _build_slot_mapping(self) -> List[int]:
        ratio_sum = sum(self._data_ratio)
        if ratio_sum <= 0:
            raise ValueError('sum(data_ratio) must be > 0.')

        normalized = [r / ratio_sum for r in self._data_ratio]
        raw_counts = [r * self.batch_size for r in normalized]
        counts = [int(math.floor(v)) for v in raw_counts]

        remain = self.batch_size - sum(counts)
        if remain > 0:
            fracs = [
                (raw_counts[i] - counts[i], i) for i in range(len(raw_counts))
            ]
            fracs.sort(key=lambda x: x[0], reverse=True)
            for _, idx in fracs[:remain]:
                counts[idx] += 1

        positive_ratio_indices = [i for i, r in enumerate(normalized) if r > 0]
        zero_count_positive = [i for i in positive_ratio_indices if counts[i] == 0]
        if zero_count_positive and len(positive_ratio_indices) <= self.batch_size:
            for idx in zero_count_positive:
                donor_candidates = [i for i in positive_ratio_indices if counts[i] > 1]
                if not donor_candidates:
                    break
                donor = max(donor_candidates, key=lambda i: counts[i])
                counts[donor] -= 1
                counts[idx] += 1

        if sum(counts) != self.batch_size:
            raise RuntimeError('Failed to convert data_ratio into per-batch counts.')

        slots = []
        for source_idx, count in enumerate(counts):
            slots.extend([source_idx] * count)
        return slots

    def full_init(self):
        if self._fully_initialized:
            return

        for dataset in self.datasets:
            dataset.full_init()

        self._metainfo = self.datasets[0].metainfo

        rng = random.Random(self._subset_seed)
        self._source_indices = []
        source_sizes = []

        for source_idx, dataset in enumerate(self.datasets):
            size = len(dataset)
            ratio = self.subset_ratios[source_idx]
            if size <= 0:
                raise ValueError(f'source dataset {source_idx} is empty.')

            if ratio <= 0:
                raise ValueError(
                    f'subset_ratios[{source_idx}] must be > 0, got {ratio}.')

            subset_size = max(1, int(size * ratio))
            subset_size = min(size, subset_size)
            all_indices = list(range(size))
            if subset_size < size:
                selected = rng.sample(all_indices, subset_size)
                selected.sort()
            else:
                selected = all_indices
            self._source_indices.append(selected)
            source_sizes.append(len(selected))

        num_batches = 1
        for source_idx, per_batch in enumerate(self._source_counts_per_batch):
            if per_batch > 0:
                num_batches = max(
                    num_batches,
                    math.ceil(source_sizes[source_idx] / per_batch))
        self._dataset_len = num_batches * self.batch_size
        self._fully_initialized = True

    def _source_from_slot(self, slot: int) -> int:
        return self._slot_to_source[slot]

    def _source_occurrence_before(self, global_idx: int, source_idx: int) -> int:
        full_batches = global_idx // self.batch_size
        slot = global_idx % self.batch_size
        base = full_batches * self._source_counts_per_batch[source_idx]

        in_partial = 0
        for pos in range(slot):
            if self._slot_to_source[pos] == source_idx:
                in_partial += 1
        return base + in_partial

    def _map_global_to_local(self, idx: int):
        idx = idx % len(self)
        slot = idx % self.batch_size
        source_idx = self._source_from_slot(slot)

        source_occurrence = self._source_occurrence_before(idx, source_idx)
        source_pool = self._source_indices[source_idx]
        local_pos = source_occurrence % len(source_pool)
        local_idx = source_pool[local_pos]
        return source_idx, local_idx

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        source_idx, local_idx = self._map_global_to_local(idx)
        data_info = self.datasets[source_idx].get_data_info(local_idx)
        data_info = copy.deepcopy(data_info)
        data_info['source_dataset_idx'] = source_idx
        return data_info

    @force_full_init
    def __len__(self):
        return self._dataset_len

    def __getitem__(self, idx: int):
        source_idx, local_idx = self._map_global_to_local(idx)
        sample = self.datasets[source_idx][local_idx]
        if isinstance(sample, dict):
            sample = copy.deepcopy(sample)
            sample['source_dataset_idx'] = source_idx
        return sample
