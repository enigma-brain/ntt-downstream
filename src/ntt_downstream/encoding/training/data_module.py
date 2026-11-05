"""Lightning DataModule for the Cowley V4 dataset."""

from __future__ import annotations

from functools import partial

import lightning as L
from torch.utils.data import DataLoader

from ntt_downstream.encoding.dataloading.cowley_v4 import (
    CowleyV4MultiSessionDataset,
    CowleyV4RepeatAveragedDataset,
    DistributedSessionBatchSampler,
    SessionBatchSampler,
    numpy_collate_fn,
)


class CowleyV4DataModule(L.LightningDataModule):
    """Lightning DataModule for multi-session Cowley V4 data.

    Creates train/val/test datasets with session-based batch sampling.
    Also provides a repeat-averaged validation dataset for eval against
    trial-averaged responses.

    Config expected:
        data.dataset_path: str — path to directory of HDF5 session files
        data.response_window: float — seconds (default 0.200)
        data.bin_size: float — seconds (default 0.001)
        data.author_split: str | None — session-level split filter
        data.image_size: list[int] | None — optional [H, W] for resize
        dataloader.batch_size: int
        dataloader.num_workers: int
        dataloader.prefetch_factor: int
        dataloader.persistent_workers: bool
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: CowleyV4MultiSessionDataset | CowleyV4RepeatAveragedDataset | None = None
        self.val_dataset: CowleyV4MultiSessionDataset | None = None
        self.val_avg_dataset: CowleyV4RepeatAveragedDataset | None = None
        self.test_dataset: CowleyV4MultiSessionDataset | None = None
        self.test_avg_dataset: CowleyV4RepeatAveragedDataset | None = None
        self._train_sampler: SessionBatchSampler | DistributedSessionBatchSampler | None = None

    def _common_kwargs(self) -> dict:
        data_cfg = self.cfg.data
        # Enable ImageNet normalization when using a pretrained backbone
        pretrained = self.cfg.model.get("pretrained_backbone", None)
        imagenet_normalize = pretrained is not None

        kwargs = {
            "data_dir": data_cfg.dataset_path,
            "response_window": data_cfg.get("response_window", 0.200),
            "bin_size": data_cfg.get("bin_size", 0.001),
            "response_offset": data_cfg.get("response_offset", 0.0),
            "author_split": data_cfg.get("author_split", None),
            "session_ids": data_cfg.get("session_ids", None),
            "subjects": data_cfg.get("subjects", None),
            "min_snr": data_cfg.get("min_snr", None),
            "imagenet_normalize": imagenet_normalize,
        }
        image_size = data_cfg.get("image_size", None)
        if image_size is not None:
            kwargs["image_size"] = tuple(image_size)
        return kwargs

    def setup(self, stage: str | None = None):
        common = self._common_kwargs()
        if stage == "fit" or stage is None:
            base_train = CowleyV4MultiSessionDataset(**common, split="train")
            if self.cfg.data.get("train_on_averaged", False):
                self.train_dataset = CowleyV4RepeatAveragedDataset(base_train)
            else:
                self.train_dataset = base_train
            self.val_dataset = CowleyV4MultiSessionDataset(**common, split="valid")
            self.val_avg_dataset = CowleyV4RepeatAveragedDataset(self.val_dataset)
        if stage == "test" or stage is None:
            self.test_dataset = CowleyV4MultiSessionDataset(**common, split="test")
            self.test_avg_dataset = CowleyV4RepeatAveragedDataset(self.test_dataset)

    def _make_loader(
        self,
        dataset,
        shuffle: bool = True,
        drop_last: bool = False,
        spike_augmentations=None,
    ) -> DataLoader:
        dl_cfg = self.cfg.dataloader
        batch_size = dl_cfg.get("batch_size", 64)
        num_workers = dl_cfg.get("num_workers", 4)

        # Use DDP-aware sampler if trainer has multiple devices
        if self.trainer is not None and getattr(self.trainer, "num_devices", 1) > 1:
            sampler = DistributedSessionBatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
            )
        else:
            sampler = SessionBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(numpy_collate_fn, spike_augmentations=spike_augmentations),
            prefetch_factor=dl_cfg.get("prefetch_factor", 2) if num_workers > 0 else None,
            persistent_workers=dl_cfg.get("persistent_workers", True) and num_workers > 0,
        )

    def train_dataloader(self):
        loader = self._make_loader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            spike_augmentations=self.cfg.data.get("spike_augmentations", None),
        )
        self._train_sampler = loader.batch_sampler  # type: ignore[assignment]
        return loader

    @property
    def train_sampler(self) -> SessionBatchSampler | DistributedSessionBatchSampler | None:
        """The training batch sampler (set after `train_dataloader()` is called)."""
        return self._train_sampler

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False, drop_last=False)

    def val_average_dataloader(self) -> DataLoader:
        """Dataloader for repeat-averaged validation (one item per unique image)."""
        if self.val_avg_dataset is None:
            raise ValueError("val_avg_dataset not available. Call setup('fit') first.")
        return self._make_loader(self.val_avg_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False, drop_last=False)

    def get_session_info(self, split: str = "train") -> dict[str, int]:
        """Return {session_id: n_units} for the given split's dataset.

        Must be called after setup().
        """
        ds = {"train": self.train_dataset, "valid": self.val_dataset, "test": self.test_dataset}.get(split)
        if ds is None:
            raise ValueError(f"Dataset for split '{split}' not available. Call setup() first.")
        return {meta["session_id"]: meta["n_units"] for meta in ds.session_meta}

    def get_subject_info(self, split: str = "train") -> dict[str, str]:
        """Return {session_id: subject_id} for the given split's dataset.

        Must be called after setup().
        """
        ds = {"train": self.train_dataset, "valid": self.val_dataset, "test": self.test_dataset}.get(split)
        if ds is None:
            raise ValueError(f"Dataset for split '{split}' not available. Call setup() first.")
        return {meta["session_id"]: meta["subject_id"] for meta in ds.session_meta}
