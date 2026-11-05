"""Lightning DataModule for Wingert auditory cortex encoding."""

from __future__ import annotations

import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ntt_downstream.encoding.dataloading.samplers import (
    DistributedSessionBatchSampler,
    SessionBatchSampler,
)
from ntt_downstream.encoding.dataloading.wingert_auditory import (
    WingertEvalCollate,
    WingertMultiSessionDataset,
    WingertRepeatAveragedDataset,
    WingertTrainCollate,
)


class WingertAuditoryDataModule(L.LightningDataModule):
    """DataModule for Wingert auditory cortex multi-session encoding.

    Args:
        cfg: Hydra config with data and dataloader groups.
        num_devices: Number of DDP devices (for distributed sampler).
    """

    def __init__(self, cfg: DictConfig, num_devices: int = 1):
        super().__init__()
        self.cfg = cfg
        self.num_devices = num_devices

        self.train_dataset: WingertMultiSessionDataset | None = None
        self.val_dataset: WingertMultiSessionDataset | None = None
        self.val_avg_dataset: WingertRepeatAveragedDataset | None = None
        self.test_dataset: WingertMultiSessionDataset | None = None
        self._train_sampler = None

    def setup(self, stage: str | None = None):
        dcfg = self.cfg.data
        data_dir = str(dcfg.dataset_path)
        session_ids = list(dcfg.session_ids) if dcfg.get("session_ids") else None
        fs = float(dcfg.get("spectrogram_fs", 100.0))

        # Compute response_fs from response_bin_size if provided
        response_bin_size = dcfg.get("response_bin_size", None)
        response_fs = None
        if response_bin_size is not None:
            response_fs = 1.0 / float(response_bin_size)

        response_normalization = str(dcfg.get("response_normalization", "none"))
        token_target_cache_dir = dcfg.get("token_target_cache_dir", None)
        common_kwargs = {
            "data_dir": data_dir,
            "session_ids": session_ids,
            "fs": fs,
            "response_fs": response_fs,
            "response_normalization": response_normalization,
            "token_target_cache_dir": token_target_cache_dir,
        }

        self.train_dataset = WingertMultiSessionDataset(split="train", **common_kwargs)
        self.val_dataset = WingertMultiSessionDataset(split="valid", **common_kwargs)
        self.val_avg_dataset = WingertRepeatAveragedDataset(self.val_dataset)
        self.test_dataset = WingertMultiSessionDataset(split="test", **common_kwargs)

    def _make_sampler(self, dataset, shuffle=True):
        bs = int(self.cfg.dataloader.batch_size)
        if self.num_devices > 1:
            return DistributedSessionBatchSampler(dataset, batch_size=bs, shuffle=shuffle)
        return SessionBatchSampler(dataset, batch_size=bs, shuffle=shuffle)

    def train_dataloader(self) -> DataLoader:
        crop_length = int(self.cfg.data.get("crop_length", 1000))
        model_window_size = self.cfg.model.get("window_size", None)
        model_window_size = int(model_window_size) if model_window_size is not None else None
        assert self.train_dataset is not None, "call setup() first"
        self._train_sampler = self._make_sampler(self.train_dataset, shuffle=True)
        nw = int(self.cfg.dataloader.get("num_workers", 4))
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._train_sampler,
            collate_fn=WingertTrainCollate(
                crop_length=crop_length,
                model_window_size=model_window_size,
                spike_augmentations=self.cfg.data.get("spike_augmentations", None),
            ),
            num_workers=nw,
            prefetch_factor=int(self.cfg.dataloader.get("prefetch_factor", 2)) if nw > 0 else None,
            persistent_workers=bool(self.cfg.dataloader.get("persistent_workers", True)) and nw > 0,
            pin_memory=True,
        )

    @property
    def train_sampler(self) -> SessionBatchSampler | DistributedSessionBatchSampler | None:
        """The training batch sampler (set after `train_dataloader()` is called)."""
        return self._train_sampler

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "call setup() first"
        sampler = self._make_sampler(self.val_dataset, shuffle=False)
        nw = int(self.cfg.dataloader.get("num_workers", 4))
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            collate_fn=WingertEvalCollate(),
            num_workers=nw,
            persistent_workers=bool(self.cfg.dataloader.get("persistent_workers", True)) and nw > 0,
            pin_memory=True,
        )

    def val_average_dataloader(self) -> DataLoader:
        assert self.val_avg_dataset is not None, "call setup() first"
        sampler = self._make_sampler(self.val_avg_dataset, shuffle=False)
        nw = int(self.cfg.dataloader.get("num_workers", 4))
        return DataLoader(
            self.val_avg_dataset,
            batch_sampler=sampler,
            collate_fn=WingertEvalCollate(),
            num_workers=nw,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "call setup() first"
        sampler = self._make_sampler(self.test_dataset, shuffle=False)
        nw = int(self.cfg.dataloader.get("num_workers", 4))
        return DataLoader(
            self.test_dataset,
            batch_sampler=sampler,
            collate_fn=WingertEvalCollate(),
            num_workers=nw,
            pin_memory=True,
        )

    def get_session_info(self, split: str = "train") -> dict[str, int]:
        ds = {"train": self.train_dataset, "valid": self.val_dataset, "test": self.test_dataset}[split]
        assert ds is not None, "call setup() first"
        return {meta["session_id"]: meta["n_units"] for meta in ds.session_meta}

    def get_subject_info(self, split: str = "train") -> dict[str, str]:
        ds = {"train": self.train_dataset, "valid": self.val_dataset, "test": self.test_dataset}[split]
        assert ds is not None, "call setup() first"
        return {meta["session_id"]: meta["session_id"][:3] for meta in ds.session_meta}
