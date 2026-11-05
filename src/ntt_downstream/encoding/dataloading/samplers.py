"""Session-aware batch samplers for multi-session neural datasets.

Both samplers require the dataset to expose a ``session_to_global_indices``
attribute: ``dict[str, list[int]]`` mapping session IDs to global item indices.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler


class SessionBatchSampler(Sampler):
    """Yields batches where all items come from the same session.

    Each batch contains indices from exactly one session (neuron counts differ
    across sessions, so cross-session batches can't be collated). Within an
    epoch, batches are globally shuffled so sessions are interleaved across
    training steps rather than appearing in contiguous blocks — this prevents
    the block-structured loss fluctuations that arise when the model sees all
    of session A, then all of session B, etc.
    """

    def __init__(self, dataset: Any, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _build_batches(self, rng: np.random.RandomState) -> list[list[int]]:
        batches: list[list[int]] = []
        for raw_indices in self.dataset.session_to_global_indices.values():
            indices = np.asarray(raw_indices)
            if self.shuffle:
                indices = indices[rng.permutation(len(indices))]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size].tolist()
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        return batches

    def __iter__(self):
        rng = np.random.RandomState(self._epoch)
        batches = self._build_batches(rng)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        total = 0
        for indices in self.dataset.session_to_global_indices.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


class DistributedSessionBatchSampler(Sampler):
    """DDP-aware session batch sampler: splits sessions round-robin across ranks.

    Yields the same number of batches on every rank so DDP all-reduce calls
    stay in lockstep. The common length is the min over ranks of each rank's
    local batch count; any rank that would otherwise produce more batches
    stops early at that cap.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._epoch = 0
        all_sids = sorted(dataset.session_to_global_indices.keys())
        self.rank_sessions = all_sids[rank::num_replicas]

        per_rank_lens = []
        for r in range(num_replicas):
            r_sids = all_sids[r::num_replicas]
            total = 0
            for sid in r_sids:
                n = len(dataset.session_to_global_indices[sid])
                total += n // batch_size if drop_last else (n + batch_size - 1) // batch_size
            per_rank_lens.append(total)
        self._common_len = min(per_rank_lens) if per_rank_lens else 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self._epoch + self.rank * 1000)
        batches: list[list[int]] = []
        for sid in self.rank_sessions:
            indices = np.asarray(self.dataset.session_to_global_indices[sid])
            if self.shuffle:
                indices = indices[rng.permutation(len(indices))]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size].tolist()
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        # Cap emission so all DDP ranks yield the same count.
        for batch in batches[: self._common_len]:
            yield batch

    def __len__(self):
        return self._common_len
