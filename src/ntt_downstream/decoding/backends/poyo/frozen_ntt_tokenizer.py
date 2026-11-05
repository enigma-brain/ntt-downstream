"""FrozenNTTTokenizer — wraps a frozen ntt BaseTokenizer for the POYO bypass.

Loads an ntt BaseTokenizer (from checkpoint or Poisson baseline factory),
freezes all parameters, and produces a TokenizerOutput with
flattened token sequences, timestamps, unit IDs, and masks.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from ntt.core.components.types import BinnedSpikesInput
from ntt.core.model.tokenizer.base import BaseTokenizer
from ntt.evaluation.utils.window_chunking import (
    _effective_window_length_bins,
    _slice_batch_time,
    requires_fixed_input_window,
)
from torch import nn

from ntt_downstream.decoding.backends.poyo.tokenizer_output import TokenizerOutput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DiscreteProbe:
    """Result of probing a tokenizer for a discrete bottleneck.

    ``indices_source`` is one of ``"encode" | "bottleneck" | "forward"``
    and names where token indices live: directly in ``encode()``'s
    return, in ``bottleneck.get_indices``, or pulled out of a full
    forward pass.
    """

    is_discrete: bool
    indices_source: str | None
    num_codebooks: int | None
    vocab_size: int | None


_NON_DISCRETE_PROBE = _DiscreteProbe(is_discrete=False, indices_source=None, num_codebooks=None, vocab_size=None)


def _loosen_legacy_spec_validation() -> None:
    """Switch all loaded ``TokenizerSpec`` subclasses to ``extra='ignore'``.

    Tolerates checkpoints whose pickled ``TokenizerSpec`` carries fields that
    the installed schema does not define (e.g. extra tiling fields on a
    ``WanNeuralTokenizerSpec``). The parent ``TokenizerSpec`` sets
    ``extra='forbid'``, so unpickling such a checkpoint raises
    ``pydantic.ValidationError`` inside ``torch.load(...)`` itself — before
    ``load_tokenizer_from_checkpoint_no_lightning``'s ``allow_spec_update``
    hook can fire (that hook only handles *added* fields, not *removed*
    ones).

    Loosening to ``extra='ignore'`` lets unpickling succeed at the cost of
    silently dropping fields the current schema doesn't know about. Whether
    that's safe depends on whether the dropped fields were
    architecture-affecting or just inference-time hyperparameters.

    Idempotent — re-running is cheap because ``model_rebuild(force=True)``
    is only called when a class is still on ``extra='forbid'``.
    """
    # Trigger registration of all contrib tokenizers so __subclasses__
    # returns the full set (otherwise lazy-imported subclasses are missed).
    import ntt.contrib.tokenizer  # noqa: F401
    from ntt.core.model.tokenizer.base import TokenizerSpec

    patched = []
    for cls in TokenizerSpec.__subclasses__():
        if cls.model_config.get("extra") == "forbid":
            cls.model_config["extra"] = "ignore"
            cls.model_rebuild(force=True)
            patched.append(cls.__name__)

    if patched:
        logger.warning(
            "Loosened pydantic 'extra' policy from 'forbid' to 'ignore' on %d "
            "TokenizerSpec subclasses (%s). Unknown fields in "
            "checkpoints will be silently dropped. If a dropped field affects "
            "architecture or inference behavior, the loaded tokenizer may "
            "behave incorrectly.",
            len(patched),
            ", ".join(sorted(patched)),
        )


def _resolve_ckpt_path(ckpt_path: str | Path) -> str | Path:
    """Resolve a tokenizer checkpoint path, downloading from MLflow if needed.

    Local filesystem paths pass through unchanged. URIs starting with
    ``mlflow-artifacts:`` or ``mlflow:`` are downloaded via
    ``mlflow.artifacts.download_artifacts`` (which reads
    ``MLFLOW_TRACKING_URI`` / ``MLFLOW_TRACKING_USERNAME`` /
    ``MLFLOW_TRACKING_PASSWORD`` from the environment).

    Mirrors ``ntt.jobs.eval.EvalContext._resolve_ckpt_path``: if the URI
    points to a directory containing exactly one ``*.ckpt`` file, return
    that file; if it points to a single artifact, return it directly.
    """
    path_str = str(ckpt_path)
    if not (path_str.startswith("mlflow-artifacts:") or path_str.startswith("mlflow:")):
        return ckpt_path

    import mlflow.artifacts

    logger.info(f"Downloading MLflow tokenizer artifact: {path_str}")
    resolved = Path(mlflow.artifacts.download_artifacts(path_str))
    if resolved.is_dir():
        ckpt_files = list(resolved.glob("*.ckpt"))
        if len(ckpt_files) == 1:
            return ckpt_files[0]
        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt file found in downloaded MLflow artifact: {resolved}")
        raise ValueError(
            f"Multiple .ckpt files in downloaded MLflow artifact {resolved}: {[f.name for f in ckpt_files]}"
        )
    return resolved


def _remap_wanneural_legacy_residual_keys(
    tokenizer: BaseTokenizer,
    tokenizer_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map WanNeural residual ``.6`` weights onto the ``.5`` slots.

    Some WanNeural checkpoints store the residual blocks with an extra
    non-parametric module, so the final affine/conv parameters sit at
    ``residual.6.{weight,bias}``, while the module built here keeps the same
    tensors at ``residual.5.{weight,bias}``.
    Only remap when the destination exists and shapes match.
    """
    expected = tokenizer.state_dict()
    remapped = dict(tokenizer_state)
    n_remapped = 0
    pattern = re.compile(r"(^|.*\.)residual\.6\.(weight|bias)$")

    for old_key, value in tokenizer_state.items():
        if not pattern.match(old_key):
            continue
        new_key = old_key.replace(".residual.6.", ".residual.5.")
        if new_key not in expected:
            continue
        if tuple(expected[new_key].shape) != tuple(value.shape):
            logger.warning(
                "Not remapping WanNeural key %s -> %s: shape %s != %s",
                old_key,
                new_key,
                tuple(value.shape),
                tuple(expected[new_key].shape),
            )
            continue
        remapped[new_key] = value
        del remapped[old_key]
        n_remapped += 1

    if n_remapped:
        logger.warning(
            "Remapped %d WanNeural residual state_dict tensor(s) from residual.6.* to residual.5.* for compatibility.",
            n_remapped,
        )
    return remapped


def _load_tokenizer_from_checkpoint_compat(
    ckpt_path: str | Path,
    device: str = "cpu",
    allow_spec_update: bool = True,
) -> BaseTokenizer:
    """Load an NTT tokenizer, with a narrow fallback for WanNeural residual-key checkpoints."""
    from ntt.core.model.tokenizer import create_tokenizer

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    hparams = ckpt.get("hyper_parameters", {})
    spec_dict = hparams.get("tokenizer_spec")
    if spec_dict is None:
        raise ValueError("No tokenizer_spec found in checkpoint hyper_parameters. Provide the spec explicitly.")

    import ntt.contrib.tokenizer  # noqa: F401
    from ntt.contrib import TokenizerSpecKind
    from pydantic import TypeAdapter, ValidationError

    original_spec = spec_dict
    if hasattr(spec_dict, "model_dump") and allow_spec_update:
        dumped = spec_dict.model_dump(by_alias=True, exclude_none=True)
        try:
            spec = TypeAdapter(TokenizerSpecKind).validate_python(dumped)
        except (TypeError, ValidationError) as exc:
            # Some specs use ``Dist`` subclass fields (e.g. ``UniformDist`` for
            # ``window_length_seconds``) whose discriminator does not survive
            # the ``model_dump`` round-trip, leaving pydantic unable to pick a
            # concrete subclass on re-validation. Fall back to the original
            # in-memory spec instance — schema migration is skipped, so any
            # field missing from the pickled spec keeps its default.
            logger.warning(
                "Tokenizer spec re-validation failed (%s); falling back to the "
                "original in-memory spec instance. Schema migration disabled.",
                exc.__class__.__name__,
            )
            spec = original_spec
    else:
        spec = TypeAdapter(TokenizerSpecKind).validate_python(spec_dict)
    tokenizer = create_tokenizer(spec)

    state_dict = ckpt.get("state_dict", ckpt)
    tokenizer_state = {k.replace("tokenizer.", "", 1): v for k, v in state_dict.items() if k.startswith("tokenizer.")}
    if not tokenizer_state:
        tokenizer_state = state_dict

    try:
        tokenizer.load_state_dict(tokenizer_state)
    except RuntimeError as exc:
        msg = str(exc)
        if "residual.5." in msg and "residual.6." in msg:
            logger.warning(
                "Tokenizer strict load failed with WanNeural residual key "
                "mismatch; retrying with a shape-checked residual.6 -> residual.5 "
                "remap."
            )
            tokenizer.load_state_dict(_remap_wanneural_legacy_residual_keys(tokenizer, tokenizer_state))
        else:
            # Some checkpoints carry extra state from training-only modules
            # attached to the tokenizer (e.g. ``sigreg.*`` regularizer buffers
            # used during training but not part of inference). Mirror
            # ``ntt.core.model``'s loose loss-module load: retry non-strict and
            # surface what was skipped, but still fail if any required tensor
            # is missing.
            missing, unexpected = tokenizer.load_state_dict(tokenizer_state, strict=False)
            if missing:
                raise
            if unexpected:
                logger.warning(
                    "Loaded tokenizer with non-strict matching: skipped %d unexpected key(s): %s",
                    len(unexpected),
                    ", ".join(sorted(unexpected)),
                )

    tokenizer.eval()
    return tokenizer


def _unflatten_tn_indices(indices: torch.Tensor, T: int, N: int) -> torch.Tensor:
    """Reshape ``(B, T*N[, Q])`` int indices to ``(B, T, N, Q)`` int64.

    Layout follows the wanneural ``VQPTBottleneck._to_seq`` convention
    ``"b c t n -> b (t n) c"`` — i.e. t-major, n-minor.

    Args:
        indices: ``(B, T*N)`` or ``(B, T*N, Q)`` int tensor.
        T: number of time bins.
        N: number of unit bins.

    Returns:
        ``(B, T, N, Q)`` int64 tensor with ``Q == 1`` when input is 3-D.
    """
    if indices.ndim == 2:
        B, S = indices.shape
        if S != T * N:
            raise ValueError(f"indices length {S} != T*N = {T}*{N}={T * N}")
        return indices.reshape(B, T, N, 1).to(torch.long)
    if indices.ndim == 3:
        B, S, q = indices.shape
        if S != T * N:
            raise ValueError(f"indices length {S} != T*N = {T}*{N}={T * N}")
        return indices.reshape(B, T, N, q).to(torch.long)
    raise ValueError(f"unexpected indices ndim={indices.ndim} (shape {tuple(indices.shape)})")


class FrozenNTTTokenizer(nn.Module):
    """Wraps a frozen ntt tokenizer for downstream evaluation.

    The ntt tokenizer's encode() produces [B, T', U, D] output.
    This wrapper:
    - Freezes all parameters (no gradients)
    - Runs encode_with_timestamps() to get latents + timestamps
    - Flattens T'*U -> N for per-unit tokenizers, or squeezes U=1 for population tokenizers
    - Computes per-token unit IDs and masks
    - Returns a TokenizerOutput

    Args:
        tokenizer: An ntt BaseTokenizer instance.
        freeze: If True (default), set tokenizer to eval mode and disable gradients.
            If False, leave tokenizer trainable and log a banner warning.
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        freeze: bool = True,
        discrete_input_mode: Literal["z_q", "embedding"] = "z_q",
        discrete_indices_source: Literal["auto", "encode", "bottleneck", "forward"] = "auto",
        fixed_window_seconds: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(tokenizer, BaseTokenizer):
            raise TypeError(f"Expected BaseTokenizer instance, got {type(tokenizer).__name__}")
        self.tokenizer = tokenizer
        self._freeze = freeze
        self.discrete_input_mode = discrete_input_mode
        self.discrete_indices_source = discrete_indices_source
        # Native input-window length (seconds) for fixed-input-window
        # tokenizers. ``None`` (default) means derive the chunk width from the
        # spec's effective window; an explicit config value overrides it when
        # the deployed window differs from the spec's. See chunk_width_bins.
        self.fixed_window_seconds = fixed_window_seconds
        # Global slot proxies: when set, _flatten_4d's
        # combines_units branch emits these M global indices as the
        # slot unit_ids instead of per-session `batch.unit_ids[:, :M]`.
        # Wrapper-level setup populates this buffer after registering
        # "_slot_k" tokens in the unit_emb vocab.
        self.slot_global_ids: torch.Tensor  # declared so nn.Module.__getattr__ doesn't widen it
        self.register_buffer("slot_global_ids", torch.empty(0, dtype=torch.long), persistent=False)
        if freeze:
            self.tokenizer.eval()
            self.tokenizer.requires_grad_(False)
        else:
            banner = "=" * 78
            logger.warning(
                "\n%s\n"
                "FrozenNTTTokenizer initialized with freeze=False — "
                "tokenizer params (%s) are TRAINABLE and will receive gradients.\n"
                "%s",
                banner,
                type(tokenizer).__name__,
                banner,
            )

        # Cache spec properties.
        # `combines_units=True` ⇔ the output U dim does NOT index real session
        # units. Two cases: (a) `compress_n=True` tokenizers (population/LDNS,
        # plus perceiver-style with abstract latent-slot dim where U_out =
        # latent_tokens, not U_input), and (b) `has_unit_dim=False` (no unit
        # dim at all, e.g. simple_vae). The adapter treats `combines_units=True`
        # as "no per-token unit IDs" and forbids `unit_embedding_mode∈{add,concat}`.
        spec = self.tokenizer.spec
        self._has_unit_dim = spec.has_unit_dim
        self._combines_units = spec.compress_n or not spec.has_unit_dim

        token_bin_ms = spec.token_bin_size_ms
        self._token_bin_size_s: float | None = token_bin_ms / 1000.0 if token_bin_ms is not None else None

        # For compress_n tokenizers with fixed neuron count, we need to pad
        # sessions that have fewer neurons than the tokenizer expects.
        self._requires_neuron_padding = spec.compress_n and not spec.variable_n and spec.n_neurons is not None
        self._expected_n_neurons: int | None = (
            cast(int | None, spec.n_neurons) if self._requires_neuron_padding else None
        )

        # Discrete-bottleneck probe — settles is_discrete + indices source + Q + K.
        bn = getattr(self.tokenizer, "bottleneck", None)
        self._has_indices_to_codes = bn is not None and hasattr(bn, "indices_to_codes")
        probe = self._probe_discrete_metadata(bn)
        self._is_discrete: bool = probe.is_discrete
        self._indices_source_resolved: str | None = probe.indices_source
        self._num_codebooks: int | None = probe.num_codebooks
        self._vocab_size: int | None = probe.vocab_size

        # Validation gates.
        if discrete_input_mode == "embedding" and not self._is_discrete:
            raise ValueError(
                f"discrete_input_mode='embedding' requires a tokenizer with a "
                f"discrete bottleneck, but {type(tokenizer).__name__} is not discrete. "
                f"Use discrete_input_mode='z_q' for continuous tokenizers."
            )

        logger.info(
            "FrozenNTTTokenizer init: discrete_input_mode=%s, is_discrete=%s, "
            "indices_source=%s, num_codebooks=%s, vocab_size=%s",
            self.discrete_input_mode,
            self._is_discrete,
            self._indices_source_resolved,
            self._num_codebooks,
            self._vocab_size,
        )

    def _probe_discrete_metadata(self, bn: object | None) -> _DiscreteProbe:
        """Detect a discrete bottleneck on the wrapped tokenizer.

        Two-stage detection:
          1. ``"auto"`` mode runs a dummy encode pass; if it returns int
             tensors the tokenizer is definitely discrete (the most direct
             test, catches transformer-vq tokenizers with no ``bottleneck``).
          2. Otherwise check the structural hint (``spec.latent_type``
             ends with ``"discrete"`` or the bottleneck has
             ``get_indices``) and resolve the indices source from there.
        """
        spec = self.tokenizer.spec

        if self.discrete_indices_source == "auto":
            probe = self._try_encode_probe(spec)
            if probe is not None:
                return probe

        return self._probe_from_structural_hint(spec, bn)

    def _try_encode_probe(self, spec) -> _DiscreteProbe | None:
        """Return a probe result if ``encode()`` itself yields int indices."""
        from ntt.core.model.tokenizer.validation import make_dummy_input

        try:
            dummy = make_dummy_input(spec)
            with torch.no_grad():
                z = self.tokenizer.encode(dummy)
            if isinstance(z, dict):
                z = z["latents"]
        except Exception as exc:
            logger.debug("Discrete probe (encode) failed: %s", exc)
            return None

        if torch.is_floating_point(z):
            return None

        num_codebooks = int(z.shape[-1]) if z.ndim == 4 else 1
        return _DiscreteProbe(
            is_discrete=True,
            indices_source="encode",
            num_codebooks=num_codebooks,
            vocab_size=self._infer_vocab_size(spec, indices=z),
        )

    def _probe_from_structural_hint(self, spec, bn: object | None) -> _DiscreteProbe:
        """Resolve discreteness from spec/bottleneck shape, no forward pass."""
        latent_type = str(getattr(spec, "latent_type", "") or "").lower()
        bn_has_get_indices = bn is not None and hasattr(bn, "get_indices")
        if not (latent_type.endswith("discrete") or bn_has_get_indices):
            return _NON_DISCRETE_PROBE

        if self.discrete_indices_source != "auto":
            indices_source = self.discrete_indices_source
        else:
            # The runtime "bottleneck" path mirrors WanNeural.encode's prelude
            # (_extract_data + _pre_encode); only pick it when the tokenizer
            # actually exposes that API. Tokenizers with a get_indices bottleneck
            # but no such prelude (e.g. the wavelet-* FSQ/VQPT tokenizers) emit indices via
            # forward().tokens — route them to "forward" instead, else they
            # AttributeError on the first batch.
            can_run_bottleneck = (
                bn_has_get_indices
                and hasattr(self.tokenizer, "_extract_data")
                and hasattr(self.tokenizer, "_pre_encode")
            )
            indices_source = "bottleneck" if can_run_bottleneck else "forward"

        return _DiscreteProbe(
            is_discrete=True,
            indices_source=indices_source,
            num_codebooks=self._infer_num_codebooks(spec, bn),
            vocab_size=self._infer_vocab_size(spec),
        )

    @staticmethod
    def _infer_num_codebooks(spec, bn) -> int:
        """Infer trailing-axis size Q from the spec or bottleneck."""
        vqpt_kwargs = getattr(spec, "vqpt_kwargs", None)
        if isinstance(vqpt_kwargs, dict) and "num_quantizers" in vqpt_kwargs:
            return int(vqpt_kwargs["num_quantizers"])
        if bn is not None and hasattr(bn, "num_quantizers"):
            return int(bn.num_quantizers)
        return 1

    def _infer_vocab_size(self, spec, indices: torch.Tensor | None = None) -> int:
        """Infer per-codebook vocab K from spec or, as last resort, from indices."""
        vqpt_kwargs = getattr(spec, "vqpt_kwargs", None)
        if isinstance(vqpt_kwargs, dict):
            if "codebook_size" in vqpt_kwargs:
                return int(vqpt_kwargs["codebook_size"])
            # FSQ-family (incl. residual_fsq): the per-codebook vocabulary is the
            # product of the per-dimension quantization levels — the same
            # derivation ntt itself uses (FSQ1d.codebook_size = prod(levels);
            # the 2D wavelet tokenizers require vqpt_kwargs to carry
            # "codebook_size" or "levels"). VQPTBottleneck2d exposes no codebook_size attribute, so
            # for residual_fsq this spec field is the only source. num_codebooks
            # is taken separately from "num_quantizers" (see _infer_num_codebooks).
            levels = vqpt_kwargs.get("levels")
            if levels:
                return int(math.prod(int(lvl) for lvl in levels))
        # transformer-vq family: spec has codebook_size as a top-level field.
        codebook_size = getattr(spec, "codebook_size", None)
        if isinstance(codebook_size, int) and codebook_size > 0:
            return int(codebook_size)
        bn = getattr(self.tokenizer, "bottleneck", None)
        if bn is not None and hasattr(bn, "codebook_size"):
            return int(bn.codebook_size)
        if indices is not None:
            max_idx = int(indices.max().item()) + 1
            return 1 << max(1, max_idx - 1).bit_length()
        raise ValueError(
            f"Could not infer vocab_size from spec ({type(spec).__name__}) "
            f"or bottleneck ({type(bn).__name__ if bn else None})."
        )

    @property
    def is_discrete(self) -> bool:
        """True if the wrapped tokenizer has a discrete bottleneck."""
        return self._is_discrete

    @property
    def num_codebooks(self) -> int:
        """Number of codebooks (Q). Raises if the tokenizer is not discrete."""
        if not self._is_discrete:
            raise ValueError("num_codebooks is undefined for a non-discrete tokenizer.")
        assert self._num_codebooks is not None
        return self._num_codebooks

    @property
    def vocab_size(self) -> int:
        """Per-codebook vocabulary size (K). Raises if the tokenizer is not discrete."""
        if not self._is_discrete:
            raise ValueError("vocab_size is undefined for a non-discrete tokenizer.")
        assert self._vocab_size is not None
        return self._vocab_size

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        device: str = "cpu",
        allow_spec_update: bool = True,
        freeze: bool = True,
        discrete_input_mode: Literal["z_q", "embedding"] = "z_q",
        discrete_indices_source: Literal["auto", "encode", "bottleneck", "forward"] = "auto",
        fixed_window_seconds: float | None = None,
    ) -> FrozenNTTTokenizer:
        """Load a frozen tokenizer from a Lightning checkpoint.

        Args:
            ckpt_path: Path to the .ckpt file saved by Lightning.
            device: Device to load weights onto (default "cpu").
            allow_spec_update: If True, allow the spec to be updated with new
                fields added since the checkpoint was saved.
            freeze: If True (default), freeze tokenizer params. If False,
                leave them trainable (and log a banner warning).
            discrete_input_mode: ``"z_q"`` returns continuous embeddings;
                ``"embedding"`` returns int64 codebook indices.
            discrete_indices_source: ``"auto"`` to detect, or one of
                ``"encode" | "bottleneck" | "forward"`` to pin the indices path.
            fixed_window_seconds: Optional override of the tokenizer's native
                input window (seconds). Used for checkpoints whose spec
                mis-declares ``window_length_seconds``. ``None`` reads from the
                spec.

        Returns:
            FrozenNTTTokenizer wrapping the loaded tokenizer.
        """
        resolved_path = _resolve_ckpt_path(ckpt_path)
        # Loosen pydantic 'extra' policy on TokenizerSpec subclasses before
        # torch.load runs unpickling (which validates the spec instance).
        # Without this, checkpoints whose spec carries fields the installed
        # schema does not define fail with `extra_forbidden`.
        _loosen_legacy_spec_validation()
        tokenizer = _load_tokenizer_from_checkpoint_compat(
            ckpt_path=str(resolved_path),
            device=device,
            allow_spec_update=allow_spec_update,
        )
        return cls(
            tokenizer,
            freeze=freeze,
            discrete_input_mode=discrete_input_mode,
            discrete_indices_source=discrete_indices_source,
            fixed_window_seconds=fixed_window_seconds,
        )

    @classmethod
    def from_poisson(
        cls,
        rate_bin_size_ms: float,
        input_bin_size_ms: float,
        window_length_seconds: float = 1.0,
        freeze: bool = True,
        discrete_input_mode: Literal["z_q", "embedding"] = "z_q",
        discrete_indices_source: Literal["auto", "encode", "bottleneck", "forward"] = "auto",
    ) -> FrozenNTTTokenizer:
        """Create a frozen Poisson baseline tokenizer.

        Args:
            rate_bin_size_ms: Temporal resolution for rate estimation (milliseconds).
            input_bin_size_ms: Bin size of the input spike data (milliseconds).
            window_length_seconds: Window length in seconds (default 1.0).
            freeze: If True (default), freeze tokenizer params. If False,
                leave them trainable (and log a banner warning).
            discrete_input_mode: ``"z_q"`` (default) for the Poisson baseline.
                ``"embedding"`` will raise — Poisson is non-discrete.
            discrete_indices_source: ignored for Poisson (non-discrete).

        Returns:
            FrozenNTTTokenizer wrapping a PoissonBaselineTokenizer.
        """
        from ntt.contrib.tokenizer.poisson_baseline import (
            PoissonBaselineTokenizer,
            PoissonBaselineTokenizerSpec,
        )

        spec = PoissonBaselineTokenizerSpec(
            rate_estimation_bin_ms=rate_bin_size_ms,
            bin_size_ms=input_bin_size_ms,
            window_length_seconds=window_length_seconds,
        )
        tokenizer = PoissonBaselineTokenizer(spec)
        return cls(
            tokenizer,
            freeze=freeze,
            discrete_input_mode=discrete_input_mode,
            discrete_indices_source=discrete_indices_source,
        )

    @property
    def combines_units(self) -> bool:
        """Whether this tokenizer produces population-level tokens (no per-unit dim)."""
        return self._combines_units

    @property
    def is_session_keyed(self) -> bool:
        """Whether the tokenizer looks units up by session + *local* unit id.

        True for tokenizers whose spec declares a per-unit ``embeddings_path``
        table (the Perceiver's NuCLR embeddings, keyed by ``session_id`` and
        ``0..N_session-1``). Such tokenizers must be fed local ids for every
        session unit, i.e. the full-window handoff
        (``FullWindowUnitV2TrainWrapper``): the Perceiver *clamps* out-of-range
        ids rather than raising, so POYO-global ids would silently map onto
        the wrong embeddings.
        """
        return bool(getattr(self.tokenizer.spec, "embeddings_path", None))

    @functools.cached_property
    def num_latent_slots(self) -> int | None:
        """Number of latent slots U for combines_units tokenizers with U>1.

        For abstract-latent tokenizers (e.g. Perceiver: U=latent_tokens),
        returns the slot count M. For pure population (U=1) or per-unit
        tokenizers (combines_units=False), returns None.

        Used by the FT warm-start logic to identify the M slot-proxy
        unit_emb rows that should be copied from the base session into
        a new session at calibration time.
        """
        if not self._combines_units:
            return None

        spec = self.tokenizer.spec
        # Family-specific shortcut (Perceiver):
        val = getattr(spec, "latent_tokens", None)
        if isinstance(val, int) and val > 1:
            return int(val)

        # Generic probe via dummy forward, mirroring `output_dim`.
        from ntt.core.model.tokenizer.validation import make_dummy_input

        try:
            dummy = make_dummy_input(spec)
            sess_offset = getattr(self.tokenizer, "_session_offset", None)
            if sess_offset:
                first_sid = next(iter(sess_offset))
                B = len(dummy.session_id)
                dummy = dummy._replace(session_id=(first_sid,) * B)
            with torch.no_grad():
                # ``make_dummy_input`` is typed as a binned-or-spike-times union, but
                # the probe only ever feeds binned specs through the 4D path.
                latents, _ = self._encode_with_timestamps_4d(cast(BinnedSpikesInput, dummy))
            # _encode_with_timestamps_4d asserts 4D layout, normalizing
            # has_time_dim=False tokenizers (PCA) by inserting T'=1.
            U = int(latents.shape[2])
            return U if U > 1 else None
        except Exception as exc:
            logger.warning(
                "num_latent_slots probe failed for %s: %s",
                type(self.tokenizer).__name__,
                exc,
            )
            return None

    @property
    def input_bin_size_ms(self) -> float:
        """Input bin size in milliseconds."""
        bin_ms = self.tokenizer.spec.bin_size_ms
        assert bin_ms is not False, "FrozenNTTTokenizer requires a tokenizer with bin_size_ms set (not spike-times)"
        return float(bin_ms)

    @property
    def token_bin_size_s(self) -> float | None:
        """Token time resolution in seconds (None for variable-length tokenizers)."""
        return self._token_bin_size_s

    @functools.cached_property
    def output_dim(self) -> int:
        """Latent dimension D of the tokenizer output.

        Tries three sources, in order:
          1. ``spec.z_dim`` / ``spec.embed_dim`` if present and >0. NOTE:
             ``spec.latent_dim`` is intentionally excluded — for Perceiver
             it reports the latent-slot count (e.g. 8), not the encode
             output dim (e.g. 100), so it can't be trusted as a shortcut.
          2. A dummy ``encode_with_timestamps`` (canonical ``[B, T', U, D]``
             layout — raw ``encode()`` may emit tokenizer-native layout like
             Perceiver's ``(B, M, C, L)``). For tokenizers with
             session-keyed lookups (e.g. Perceiver's per-session embedding
             table), substitute a real session id from
             ``tokenizer._session_offset`` — the synthetic ``__validation__``
             id from ``make_dummy_input`` would raise KeyError.
          3. Fall back to ``-1`` if the probe raises (e.g. transformer-vq with a
             ``Dist`` window-length that may sample a non-patch-aligned
             window). The train_wrapper bypasses this property entirely in
             embedding mode, so ``-1`` is harmless when output_dim isn't
             actually consumed.
        """
        spec = self.tokenizer.spec
        for attr in ("z_dim", "embed_dim"):
            val = getattr(spec, attr, None)
            if isinstance(val, int) and val > 0:
                return int(val)

        from ntt.core.model.tokenizer.validation import make_dummy_input

        try:
            dummy = make_dummy_input(spec)

            sess_offset = getattr(self.tokenizer, "_session_offset", None)
            if sess_offset:
                first_sid = next(iter(sess_offset))
                B = len(dummy.session_id)
                dummy = dummy._replace(session_id=(first_sid,) * B)

            with torch.no_grad():
                # ``make_dummy_input`` is typed as a binned-or-spike-times union, but
                # the probe only ever feeds binned specs through the 4D path.
                latents, _ = self._encode_with_timestamps_4d(cast(BinnedSpikesInput, dummy))
            # _encode_with_timestamps_4d asserts 4D layout, normalizing
            # has_time_dim=False tokenizers (PCA) by inserting T'=1.
            return int(latents.shape[-1])
        except Exception as exc:
            logger.warning(
                "output_dim probe failed for %s: %s. Returning -1; downstream "
                "consumers in z_q (continuous) mode will raise.",
                type(self.tokenizer).__name__,
                exc,
                exc_info=True,
            )
            return -1

    def _pad_neurons_if_needed(self, batch: BinnedSpikesInput) -> BinnedSpikesInput:
        """Pad batch to expected neuron count for compress_n tokenizers.

        When a session has fewer neurons than the tokenizer expects (e.g., 72 vs 128),
        pads by duplicating existing neurons to fill the gap. This follows the same
        strategy as NeuronCoveringWrapper._run_single_group in ntt's evaluation utils:
        filler neurons are copies of real neurons, keeping the input in-distribution.

        Mixed-session batches: a sample with 50 valid neurons in a batch where
        another sample has 128 has its columns 50..127 sitting as scatter_add
        zeros (see ``_ntt_frozen_tokenize``'s spike_matrix construction). The
        A batch whose column count already meets ``n_expected`` must not be
        returned unchanged, since those zeros would pollute LDNS-style
        ``compress_n`` tokenizers (which ignore ``pad_and_gap_mask``) by folding
        into the population latent. This implementation rebuilds each
        sample to ``max(U_current, n_expected)`` columns, packing real neurons
        to the front and duplicating into the rest, so no sample sees zero
        gaps regardless of batch composition.

        Args:
            batch: BinnedSpikesInput with potentially fewer neurons than expected.

        Returns:
            BinnedSpikesInput with each sample's column 0..(n_expected-1) backed
            by real or duplicated neurons. Unchanged if padding isn't needed.
        """
        if not self._requires_neuron_padding:
            return batch

        assert self._expected_n_neurons is not None
        n_expected = self._expected_n_neurons

        # Actual neuron count per sample (from unit_ids_pad_mask)
        actual_counts = batch.unit_ids_pad_mask.sum(dim=1)  # [B]
        min_actual = int(actual_counts.min().item())

        B, T, U_current = batch.spike_matrix.shape
        # True zero-work case: every sample already has enough valid neurons,
        # AND the column width matches — no rebuild needed.
        if min_actual >= n_expected and U_current == n_expected:
            return batch

        out_U = max(U_current, n_expected)
        device = batch.spike_matrix.device

        new_spike = torch.zeros(B, T, out_U, device=device, dtype=batch.spike_matrix.dtype)
        new_pad_mask = torch.zeros(B, T, out_U, device=device, dtype=batch.pad_and_gap_mask.dtype)
        new_unit_ids = torch.zeros(B, out_U, device=device, dtype=batch.unit_ids.dtype)
        new_unit_pad = torch.zeros(B, out_U, device=device, dtype=batch.unit_ids_pad_mask.dtype)

        for b in range(B):
            n_actual = int(actual_counts[b].item())
            if n_actual == 0:
                # No real data for this sample — leave zeros and mark all
                # columns invalid. Tokenizer will produce zero-population
                # output for this sample; downstream mask handling is
                # unchanged.
                continue

            # Real columns: pack the sample's first `n_actual` valid columns
            # to the front of the output. Upstream `_ntt_frozen_tokenize`
            # already places valid units at positions [0, n_actual) for each
            # sample, so a contiguous slice here is correct.
            new_spike[b, :, :n_actual] = batch.spike_matrix[b, :, :n_actual]
            new_pad_mask[b, :, :n_actual] = batch.pad_and_gap_mask[b, :, :n_actual]
            new_unit_ids[b, :n_actual] = batch.unit_ids[b, :n_actual]
            new_unit_pad[b, :n_actual] = 1

            n_fill = out_U - n_actual
            if n_fill <= 0:
                continue

            # Pick filler indices deterministically from this sample's unit
            # IDs. Seeding from the per-sample uid set keeps fillers stable
            # across batch composition: (a) the same session always gets the
            # same fillers regardless of which other samples it's batched
            # with, and (b) different sessions in the same batch get
            # different fillers. (Seeding with a constant per call would key
            # the filler pattern to position-in-batch, so the same sample
            # would get different fillers at different batch indices.)
            uids_b = batch.unit_ids[b, :n_actual].detach().cpu().numpy().tobytes()
            seed = int.from_bytes(
                hashlib.blake2s(uids_b, digest_size=8).digest(),
                "little",
                signed=False,
            )
            rng = np.random.default_rng(seed)
            filler_indices = rng.choice(n_actual, size=n_fill, replace=True)
            new_spike[b, :, n_actual:] = batch.spike_matrix[b, :, filler_indices]
            new_pad_mask[b, :, n_actual:] = batch.pad_and_gap_mask[b, :, filler_indices]
            new_unit_ids[b, n_actual:] = batch.unit_ids[b, filler_indices]
            # Mark fillers as valid so the tokenizer processes them
            new_unit_pad[b, n_actual:] = 1

        logger.debug(
            "Padded neurons to %d columns (expected %d, batch min/max actual = %d/%d) by per-sample duplication.",
            out_U,
            n_expected,
            min_actual,
            int(actual_counts.max().item()),
        )

        return BinnedSpikesInput(
            spike_matrix=new_spike,
            pad_and_gap_mask=new_pad_mask,
            sampling_rate=batch.sampling_rate,
            unit_ids=new_unit_ids,
            unit_ids_pad_mask=new_unit_pad,
            duration_sec=batch.duration_sec,
            session_id=batch.session_id,
            subject_id=batch.subject_id,
            species=batch.species,
            absolute_start=batch.absolute_start,
            absolute_end=batch.absolute_end,
            behavioral_data=getattr(batch, "behavioral_data", None),
            behavioral_intervals=getattr(batch, "behavioral_intervals", None),
        )

    def _encode_with_timestamps_4d(self, batch: BinnedSpikesInput) -> tuple[torch.Tensor, dict]:
        """Call ``tokenizer.encode_with_timestamps`` and return ``[B, T', U, D]``.

        Most tokenizers already emit canonical 4D latents. The exceptions are
        the ``has_time_dim=False`` tokenizers, which come in two shapes:

        * 3D ``[B, U, D]`` — e.g. the PCA baseline
          (``PCABaselineTokenizerSpec``), which pools time within the window.
          We splice ``T'=1`` back in via ``unsqueeze(1)`` and rewrite the
          timestamps (see below).
        * 4D ``[B, 1, U, D]`` with a singleton time axis — e.g. ``simple_vae``
          (``SimpleVAETokenizerSpec``), which emits ``[B, 1, 1, D]``. These are
          already canonical, so we pass them through unchanged; the upstream
          timestamps are valid because ``latents.shape[1] == 1`` means
          ``BaseTokenizer.encode_with_timestamps`` reads a single time token.

        POYO downstream code (input_proj, the 4D-assert in
        ``_encode_zq_with_timestamps``, the output_dim probe) all assume a time
        axis is present, so both shapes must end up 4D here.

        For the 3D case we *also* rewrite the timestamps. The upstream
        ``BaseTokenizer.encode_with_timestamps`` (ntt main) reads
        ``latents.shape[1]`` as the time-token count without checking
        ``spec.has_time_dim``, so for PCA it iterates the neuron axis and emits
        ``[B, N]`` fake timestamps. We replace those with ``[B, 1]``
        window-center timestamps from ``batch.absolute_start`` /
        ``absolute_end``.

        Shapes are asserted up front so a contract violation surfaces here with
        a clear error rather than failing deeper in the wrapper or model.

        Note: the discrete-indices path (``_encode_indices_with_timestamps``)
        does *not* route through here — it calls ``encode_with_timestamps``
        directly. No discrete ``has_time_dim=False`` tokenizer (e.g. fsqffn) is
        wired through that path today; routing it would need its own handling
        (the 3D timestamp-rewrite below is not meaningful for a discrete
        single-token-per-window tokenizer), so the two paths stay separate.
        """
        latents, ts_metadata = self.tokenizer.encode_with_timestamps(batch)
        has_time_dim = getattr(self.tokenizer.spec, "has_time_dim", True)
        if not has_time_dim:
            if latents.ndim == 4:
                # Already canonical [B, T'=1, U, D] (e.g. simple_vae emits
                # [B, 1, 1, D]). Upstream timestamps are valid because the
                # singleton time axis makes BaseTokenizer read one time token,
                # so pass through unchanged. Guard the singleton invariant: a
                # has_time_dim=False tokenizer with T' > 1 would break the
                # [B, 1] timestamp contract this branch relies on.
                assert latents.shape[1] == 1, (
                    f"Tokenizer {type(self.tokenizer).__name__} has "
                    f"spec.has_time_dim=False but emitted 4D latents with "
                    f"T'={latents.shape[1]} > 1; only a singleton time axis is "
                    f"supported here. Got {tuple(latents.shape)}."
                )
                return latents, ts_metadata

            assert latents.ndim == 3, (
                f"Tokenizer {type(self.tokenizer).__name__} has "
                f"spec.has_time_dim=False; expected 3D latents [B, U, D] "
                f"(or 4D with a singleton time axis), got {tuple(latents.shape)}"
            )
            latents = latents.unsqueeze(1)  # [B, U, D] -> [B, 1, U, D]

            # Override the upstream-broken timestamps with [B, 1] window
            # centers. ``absolute_start`` / ``absolute_end`` are [B] on the
            # batch; unsqueeze to align with the canonical [B, T'=1] shape
            # that ``_flatten_4d`` and POYO expect.
            abs_start = batch.absolute_start.to(dtype=torch.float64).unsqueeze(-1)
            abs_end = batch.absolute_end.to(dtype=torch.float64).unsqueeze(-1)
            ts_metadata = {
                "token_timestamps": (abs_start + abs_end) / 2,
                "token_start_time": abs_start,
                "token_end_time": abs_end,
            }
        else:
            assert latents.ndim == 4, (
                f"Tokenizer {type(self.tokenizer).__name__} has "
                f"spec.has_time_dim=True; expected 4D latents "
                f"[B, T', U, D], got {tuple(latents.shape)}"
            )
        return latents, ts_metadata

    def _encode_zq_with_timestamps(self, batch: BinnedSpikesInput) -> tuple[torch.Tensor, dict]:
        """Encode batch and return continuous latents (z_q) with timestamps.

        For tokenizers whose ``encode()`` returns int (transformer-vq family), falls back
        to ``bottleneck.indices_to_codes`` to recover a continuous embedding.

        Returns:
            ``(latents, ts_metadata)`` with ``latents`` shape ``[B, T', U, D]``
            floating-point.
        """
        ctx = torch.no_grad() if self._freeze else contextlib.nullcontext()
        with ctx:
            latents, ts_metadata = self._encode_with_timestamps_4d(batch)

        if not torch.is_floating_point(latents):
            # transformer-vq family: encode returned indices. Dequantize via the bottleneck.
            if not self._has_indices_to_codes:
                raise ValueError(
                    f"discrete_input_mode='z_q' on a tokenizer "
                    f"({type(self.tokenizer).__name__}) whose encode() returns int "
                    f"and whose bottleneck has no indices_to_codes() method. "
                    f"No continuous embedding is available; "
                    f"use discrete_input_mode='embedding' instead."
                )
            if latents.ndim == 3:
                B, T_out, U = latents.shape
                Q = 1
                flat = latents.reshape(B, T_out * U)
            else:
                B, T_out, U, Q = latents.shape
                flat = latents.reshape(B, T_out * U, Q) if Q > 1 else latents[..., 0].reshape(B, T_out * U)
            with ctx:
                # ``bottleneck`` is a dynamic submodule of the concrete tokenizer, not of BaseTokenizer.
                codes = cast(Any, self.tokenizer).bottleneck.indices_to_codes(flat, T=T_out)
            from einops import rearrange

            latents = rearrange(codes, "b c t n -> b t n c")

        assert latents.ndim == 4, f"Expected encode_with_timestamps to yield 4D output, got {tuple(latents.shape)}"
        return latents, ts_metadata

    def _encode_indices_with_timestamps(self, batch: BinnedSpikesInput) -> tuple[torch.Tensor, dict]:
        """Encode batch and return int64 codebook indices with timestamps.

        Returns:
            ``(indices, ts_metadata)`` with ``indices`` shape ``[B, T', U, Q]``
            int64.
        """
        if not self._is_discrete:
            raise ValueError("_encode_indices_with_timestamps called on non-discrete tokenizer.")

        source = self._indices_source_resolved
        assert source is not None

        # Always fetch ts_metadata via encode_with_timestamps (cheapest reliable path).
        ctx = torch.no_grad() if self._freeze else contextlib.nullcontext()
        with ctx:
            zq, ts_metadata = self.tokenizer.encode_with_timestamps(batch)

        T_out = ts_metadata["token_timestamps"].shape[1]
        U_from_zq = zq.shape[2] if zq.ndim >= 3 else 1

        if source == "encode":
            with ctx:
                indices = self.tokenizer.encode(batch)
            if isinstance(indices, dict):
                indices = indices["latents"]
            if torch.is_floating_point(indices):
                raise RuntimeError("indices_source='encode' but encode() returned float; probe is inconsistent.")
            if indices.ndim == 3:
                indices = indices.unsqueeze(-1)  # [B, T', U, 1]
            return indices.to(torch.int64), ts_metadata

        if source == "bottleneck":
            bn = getattr(self.tokenizer, "bottleneck", None)
            if bn is None or not hasattr(bn, "get_indices"):
                raise RuntimeError(
                    "indices_source='bottleneck' but tokenizer has no bottleneck.get_indices(); probe is inconsistent."
                )
            with ctx:
                # Mirror WanNeural.encode's prelude: _extract_data returns
                # (data, neuron_mask) and the encoder expects neuron_mask.
                # These hooks live on the concrete tokenizer (probed via hasattr
                # above), not on BaseTokenizer, so go through Any.
                tok = cast(Any, self.tokenizer)
                extracted = tok._extract_data(batch)
                if isinstance(extracted, tuple):
                    data, neuron_mask = extracted[0], extracted[1] if len(extracted) > 1 else None
                else:
                    data, neuron_mask = extracted, None
                pre = tok._pre_encode(data)
                try:
                    h = tok.encoder(pre, neuron_mask=neuron_mask)
                except TypeError:
                    # Encoders that do not accept neuron_mask kwarg.
                    h = tok.encoder(pre)
                flat_indices = bn.get_indices(h)  # [B, T*U] or [B, T*U, Q]
            indices = _unflatten_tn_indices(flat_indices, T=T_out, N=U_from_zq)
            return indices, ts_metadata

        if source == "forward":
            with ctx:
                out = self.tokenizer(batch)
            tokens = getattr(out, "tokens", None)
            if tokens is None:
                raise RuntimeError("indices_source='forward' but tokenizer.forward returned no .tokens.")
            if tokens.ndim == 3:
                tokens = tokens.unsqueeze(-1)
            if tokens.ndim == 4:
                return tokens.to(torch.int64), ts_metadata
            return _unflatten_tn_indices(tokens, T=T_out, N=U_from_zq), ts_metadata

        raise ValueError(f"Unknown indices source: {source!r}")

    def _pad_time_if_needed(self, batch: BinnedSpikesInput) -> BinnedSpikesInput:
        """Pad ``batch.spike_matrix`` along time to a multiple of the compression ratio.

        POYO's ``_ntt_frozen_tokenize`` derives ``n_bins`` from spike timestamp
        min/max — the resulting T may not be a multiple of
        ``token_bin_size_ms / input_bin_size_ms`` (e.g. 999 instead of 1000).
        Conv-based tokenizers (WanNeural) tolerate this by truncating the trailing
        partial token; transformer-style tokenizers (transformer-vq) strictly assert
        ``T % patch_length == 0`` and crash. Pad the trailing time slack with
        zeros and mark those bins invalid in ``pad_and_gap_mask``.
        """
        spec = self.tokenizer.spec
        token_bin_ms = getattr(spec, "token_bin_size_ms", None)
        input_bin_ms = getattr(spec, "bin_size_ms", None)
        if (
            not isinstance(token_bin_ms, (int, float))
            or not isinstance(input_bin_ms, (int, float))
            or input_bin_ms <= 0
            or token_bin_ms <= 0
        ):
            return batch

        ratio = int(round(token_bin_ms / input_bin_ms))
        if ratio <= 1:
            return batch

        T = batch.spike_matrix.shape[1]
        rem = T % ratio
        if rem == 0:
            return batch

        n_pad = ratio - rem
        device = batch.spike_matrix.device

        spike_pad = torch.zeros(
            batch.spike_matrix.shape[0],
            n_pad,
            batch.spike_matrix.shape[2],
            device=device,
            dtype=batch.spike_matrix.dtype,
        )
        mask_pad = torch.zeros(
            batch.pad_and_gap_mask.shape[0],
            n_pad,
            batch.pad_and_gap_mask.shape[2],
            device=device,
            dtype=batch.pad_and_gap_mask.dtype,
        )

        new_spike = torch.cat([batch.spike_matrix, spike_pad], dim=1)
        new_mask = torch.cat([batch.pad_and_gap_mask, mask_pad], dim=1)

        return BinnedSpikesInput(
            spike_matrix=new_spike,
            pad_and_gap_mask=new_mask,
            sampling_rate=batch.sampling_rate,
            unit_ids=batch.unit_ids,
            unit_ids_pad_mask=batch.unit_ids_pad_mask,
            duration_sec=batch.duration_sec,
            session_id=batch.session_id,
            subject_id=batch.subject_id,
            species=batch.species,
            absolute_start=batch.absolute_start,
            absolute_end=batch.absolute_end,
            behavioral_data=getattr(batch, "behavioral_data", None),
            behavioral_intervals=getattr(batch, "behavioral_intervals", None),
        )

    def forward(self, batch: BinnedSpikesInput) -> TokenizerOutput:
        """Encode a binned-spikes batch into a ``TokenizerOutput``.

        This is the tokenizer's public entry point. It runs the wrapped ntt
        tokenizer over ``batch`` and returns the flattened token sequence —
        continuous ``[B, N, D]`` or discrete ``[B, N, Q]`` codebook indices
        (per ``discrete_input_mode``) — with per-token timestamps, unit ids,
        and a validity mask. The encode itself lives in
        ``_forward_single_window``; ``forward`` only decides whether the
        incoming window can be encoded in one pass or must be chunked first.

        Fixed-input-window tokenizers (``spec.variable_t == False`` — e.g.
        cs-*, simple_vae, wavelet-*) build their architecture for
        one input length at construction time and cannot consume a longer
        window. POYO, however, hands us a window of its own choosing (often
        1 s). When the incoming window exceeds the tokenizer's native window we
        slice it into integer-multiple native chunks, encode each, and stitch
        the per-chunk outputs along the token axis (see ``_forward_chunked``).

        Variable-T tokenizers (``variable_t == True``)
        are unaffected: the chunking branch is gated entirely on
        ``requires_fixed_input_window(spec)``, and the exact-match case
        (``t_in == chunk_width``) also takes the single-window path.
        """
        if not isinstance(batch, BinnedSpikesInput):
            raise TypeError(f"Expected BinnedSpikesInput, got {type(batch).__name__}")
        spec = self.tokenizer.spec
        if requires_fixed_input_window(spec):
            chunk_width = self.chunk_width_bins()
            t_in = batch.spike_matrix.shape[1]
            if t_in > chunk_width:
                return self._forward_chunked(batch, chunk_width)
        return self._forward_single_window(batch)

    def chunk_width_bins(self) -> int:
        """Tokenizer native input window, in input bins.

        When ``fixed_window_seconds`` is set (an optional config override for
        deployments whose native window differs from the spec's declared
        value), convert it to bins via the input bin size. Otherwise defer to
        ntt's ``_effective_window_length_bins`` — the same quantity the setup
        validator (``assert_tokenizer_accepts_eval_window``) and ntt's canonical
        chunker use — so runtime chunking and setup validation never disagree on
        the chunk width.
        """
        if self.fixed_window_seconds is not None:
            return round(self.fixed_window_seconds / (self.input_bin_size_ms / 1000.0))
        return _effective_window_length_bins(self.tokenizer.spec)

    def _forward_chunked(self, batch: BinnedSpikesInput, chunk_width: int) -> TokenizerOutput:
        """Slice the window into native chunks, encode them in one pass, stitch.

        Reuses ntt's ``_slice_batch_time`` to carve adjacent ``chunk_width``-bin
        sub-windows (which also re-offsets each chunk's ``absolute_start`` /
        ``absolute_end``, so per-chunk token timestamps already sit at their
        correct absolute positions). The chunks are then stacked along the batch
        axis and encoded in a SINGLE ``_forward_single_window`` call
        (``n_chunks`` GPU launches → 1), split back, and ``_merge_chunk_outputs``
        concatenates along the token axis. Encoding is per-sample independent for
        these tokenizers, so the single-pass result is identical to encoding
        chunk-by-chunk (pinned by ``test_forward_chunked_matches_sequential_reference``).

        Trade-off: the single encode runs on a ``[n_chunks * B, ...]`` batch, so
        peak activation memory scales with ``n_chunks`` (total work is unchanged).
        For the usual 1 s / 0.1 s case that is a 10x batch inflation — fine for
        decoding batch sizes, but worth knowing if ``B`` is pushed very large.

        The window length must be an exact integer multiple of ``chunk_width``;
        setup validation is expected to guarantee this, so a violation here is a
        hard error rather than a silent trailing-remainder drop.
        """
        t_in = batch.spike_matrix.shape[1]
        n_chunks = t_in // chunk_width
        if t_in % chunk_width != 0:
            raise ValueError(
                f"window length {t_in} bins is not an integer multiple of the tokenizer "
                f"window {chunk_width} bins; setup validation should have caught this."
            )
        sr = float(batch.sampling_rate)
        chunks = [
            _slice_batch_time(
                batch,
                t_start=k * chunk_width,
                t_end=(k + 1) * chunk_width,
                sampling_rate=sr,
            )
            for k in range(n_chunks)
        ]
        stacked_out = self._forward_single_window(self._stack_batches(cast(list[BinnedSpikesInput], chunks)))
        return self._merge_chunk_outputs(self._split_output_along_batch(stacked_out, n_chunks))

    def _forward_single_window(self, batch: BinnedSpikesInput) -> TokenizerOutput:
        """Encode one native-window batch and produce a ``TokenizerOutput``.

        Dispatches on ``discrete_input_mode``:
          - ``"z_q"``: emit float ``[B, N, D]`` continuous tokens (default).
          - ``"embedding"``: emit int64 ``[B, N, Q]`` codebook indices.

        In both modes the per-unit-vs-population flatten is identical and
        shared via ``_flatten_4d``.

        Args:
            batch: A BinnedSpikesInput from the ntt dataloader.

        Returns:
            TokenizerOutput with flattened token sequence.
        """
        batch = self._pad_neurons_if_needed(batch)
        batch = self._pad_time_if_needed(batch)

        if self.discrete_input_mode == "embedding":
            latents, ts_metadata = self._encode_indices_with_timestamps(batch)
            assert self._num_codebooks is not None and self._vocab_size is not None
            vocab_sizes: tuple[int, ...] | None = (self._vocab_size,) * self._num_codebooks
        else:
            latents, ts_metadata = self._encode_zq_with_timestamps(batch)
            vocab_sizes = None

        tokens, timestamps, unit_ids, mask = self._flatten_4d(latents, batch, ts_metadata)

        return TokenizerOutput(
            tokens=tokens,
            timestamps=timestamps.float(),
            unit_ids=unit_ids,
            mask=mask,
            combines_units=self._combines_units,
            token_bin_size_s=self._token_bin_size_s,
            vocab_sizes=vocab_sizes,
        )

    @staticmethod
    def _merge_chunk_outputs(outputs: list[TokenizerOutput]) -> TokenizerOutput:
        """Concatenate per-chunk TokenizerOutputs along the token (N) axis.

        Timestamps are NOT re-offset: each chunk was produced from a
        time-sliced BinnedSpikesInput whose absolute_start/end were shifted by
        ``_slice_batch_time``, so per-chunk token timestamps already sit at
        their correct absolute positions. The merge is a pure concat. Scalar
        invariants (combines_units, token_bin_size_s, vocab_sizes, and whether
        unit_ids is present) must match across chunks.
        """
        if len(outputs) == 1:
            return outputs[0]
        first = outputs[0]
        # Fail loud (not ``assert``, which ``python -O`` strips) when chunks
        # carry incompatible scalar invariants — a corrupt concat would
        # otherwise pass silently under -O. Matches ntt's _concat_chunk_outputs
        # and the ValueError raised by _forward_chunked.
        for o in outputs[1:]:
            if o.combines_units != first.combines_units:
                raise ValueError("combines_units mismatch across chunks")
            if o.token_bin_size_s != first.token_bin_size_s:
                raise ValueError("token_bin_size_s mismatch across chunks")
            if o.vocab_sizes != first.vocab_sizes:
                raise ValueError("vocab_sizes mismatch across chunks")
            if (o.unit_ids is None) != (first.unit_ids is None):
                raise ValueError("unit_ids presence mismatch across chunks")
        tokens = torch.cat([o.tokens for o in outputs], dim=1)
        timestamps = torch.cat([o.timestamps for o in outputs], dim=1)
        mask = torch.cat([o.mask for o in outputs], dim=1)
        unit_ids = (
            None
            if first.unit_ids is None
            else torch.cat([o.unit_ids for o in outputs if o.unit_ids is not None], dim=1)
        )
        return TokenizerOutput(
            tokens=tokens,
            timestamps=timestamps,
            unit_ids=unit_ids,
            mask=mask,
            combines_units=first.combines_units,
            token_bin_size_s=first.token_bin_size_s,
            vocab_sizes=first.vocab_sizes,
        )

    @staticmethod
    def _stack_batches(chunks: list[BinnedSpikesInput]) -> BinnedSpikesInput:
        """Concatenate equal-shaped per-chunk batches along the batch (B) axis.

        Used to encode all chunks of a window in ONE forward pass: chunk ``k`` of
        sample ``b`` lands at row ``k * B + b``. Each chunk's
        ``absolute_start``/``absolute_end`` were already re-offset by
        ``_slice_batch_time``, so per-row token timestamps stay correct after the
        single encode. ``sampling_rate`` is identical across chunks (same source
        batch), so chunk 0's value is kept.
        """
        first = chunks[0]
        return BinnedSpikesInput(
            spike_matrix=torch.cat([c.spike_matrix for c in chunks], dim=0),
            pad_and_gap_mask=torch.cat([c.pad_and_gap_mask for c in chunks], dim=0),
            sampling_rate=first.sampling_rate,
            unit_ids=torch.cat([c.unit_ids for c in chunks], dim=0),
            unit_ids_pad_mask=torch.cat([c.unit_ids_pad_mask for c in chunks], dim=0),
            duration_sec=torch.cat([c.duration_sec for c in chunks], dim=0),
            session_id=tuple(s for c in chunks for s in c.session_id),
            subject_id=tuple(s for c in chunks for s in c.subject_id),
            species=tuple(s for c in chunks for s in c.species),
            absolute_start=torch.cat([c.absolute_start for c in chunks], dim=0),
            absolute_end=torch.cat([c.absolute_end for c in chunks], dim=0),
        )

    @staticmethod
    def _split_output_along_batch(out: TokenizerOutput, n_chunks: int) -> list[TokenizerOutput]:
        """Inverse of ``_stack_batches``: split a [n_chunks*B, ...] output into chunks.

        Row ``k * B + b`` of ``out`` is chunk ``k`` of sample ``b``, so slicing
        ``[k*B : (k+1)*B]`` recovers chunk ``k`` as a ``[B, ...]``
        ``TokenizerOutput``, ready for ``_merge_chunk_outputs`` to stitch along
        the token axis.
        """
        if out.tokens.shape[0] % n_chunks != 0:
            raise ValueError(
                f"stacked batch size {out.tokens.shape[0]} is not divisible by "
                f"n_chunks={n_chunks}; cannot split per-chunk."
            )
        b = out.tokens.shape[0] // n_chunks
        parts: list[TokenizerOutput] = []
        for k in range(n_chunks):
            sl = slice(k * b, (k + 1) * b)
            parts.append(
                TokenizerOutput(
                    tokens=out.tokens[sl],
                    timestamps=out.timestamps[sl],
                    unit_ids=None if out.unit_ids is None else out.unit_ids[sl],
                    mask=out.mask[sl],
                    combines_units=out.combines_units,
                    token_bin_size_s=out.token_bin_size_s,
                    vocab_sizes=out.vocab_sizes,
                )
            )
        return parts

    def _flatten_4d(
        self,
        latents: torch.Tensor,
        batch: BinnedSpikesInput,
        ts_metadata: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Reshape 4D ``[B, T', U, D]`` latents to ``[B, N, D]`` plus aux tensors.

        ``D`` here is interpreted as either the continuous latent dim or the
        trailing codebook axis Q; the function is dtype-agnostic.

        Returns:
            ``(tokens, timestamps, unit_ids, mask)`` where:
              * ``tokens`` is ``[B, N, D]`` with the input dtype.
              * ``timestamps`` is ``[B, N]``.
              * ``unit_ids`` is ``[B, N]`` (per-unit) or ``None`` (population).
              * ``mask`` is bool ``[B, N]``.
        """
        assert latents.ndim == 4, f"_flatten_4d expects 4D input [B, T, U, D], got {tuple(latents.shape)}"
        B, T_out, U, D = latents.shape

        token_timestamps = ts_metadata["token_timestamps"]
        assert token_timestamps.shape == (B, T_out), (
            f"token_timestamps shape {tuple(token_timestamps.shape)} != ({B}, {T_out})"
        )

        if self._combines_units:
            # Population / abstract-latent path. The U dim does NOT index
            # session units. For pure population (U=1) the mask is computed
            # at U=1; for abstract-latent tokenizers (e.g. Perceiver:
            # U=latent_tokens>1), we collapse the input mask across neurons
            # and broadcast to U at flatten.
            #
            # Per-slot unit_ids: we proxy slot k → batch.unit_ids[:, k] so
            # each (session, latent_slot) gets a distinct, learnable entry
            # in the unit_emb table — mirroring how spike_times and per-unit tokenizers get
            # one entry per (session, real_unit). The proxy slot↔unit
            # mapping is arbitrary but stable per session; what matters is
            # that the unit_emb sees M distinct, session-keyed IDs per
            # window, which is what unit_identification FT can train.
            mask_per_token = self._compute_token_mask(batch, T_out, 1)  # [B, T_out, 1]
            tokens = latents.reshape(B, T_out * U, D)
            timestamps = token_timestamps.unsqueeze(-1).expand(B, T_out, U).reshape(B, T_out * U)
            if self.slot_global_ids.numel() > 0:
                # Global slot proxies: universal slot tokens shared across sessions.
                # slot_global_ids[k] is the global vocab index of "_slot_k".
                if int(self.slot_global_ids.numel()) != U:
                    raise ValueError(
                        f"slot_global_ids has {int(self.slot_global_ids.numel())} "
                        f"entries but tokenizer emits U={U} slots."
                    )
                slot_ids_M = self.slot_global_ids.to(latents.device)  # [U]
                unit_ids = slot_ids_M.view(1, 1, U).expand(B, T_out, U).reshape(B, T_out * U)
            else:
                # Per-session slot proxies: real-unit_id slot proxies.
                if batch.unit_ids.shape[1] < U:
                    raise ValueError(
                        f"combines_units proxy needs ≥{U} input unit_ids per sample, got {batch.unit_ids.shape[1]}."
                    )
                slot_proxy_ids = batch.unit_ids[:, :U]  # [B, U]
                unit_ids = slot_proxy_ids.unsqueeze(1).expand(B, T_out, U).reshape(B, T_out * U)
            mask = mask_per_token.expand(B, T_out, U).reshape(B, T_out * U)
        elif self._has_unit_dim and U > 1:
            # Per-unit tokenizer: flatten T'*U -> N (t-major, n-minor).
            mask_per_token = self._compute_token_mask(batch, T_out, U)  # [B, T_out, U]
            tokens = latents.reshape(B, T_out * U, D)
            unit_ids = batch.unit_ids.unsqueeze(1).expand(B, T_out, U).reshape(B, T_out * U)
            timestamps = token_timestamps.unsqueeze(-1).expand(B, T_out, U).reshape(B, T_out * U)
            mask = mask_per_token.reshape(B, T_out * U)
        else:
            # Single-unit-dim edge case: U=1, has_unit_dim=True, compress_n=False.
            mask_per_token = self._compute_token_mask(batch, T_out, U)  # [B, T_out, 1]
            tokens = latents.squeeze(2)  # [B, T_out, D]
            timestamps = token_timestamps  # [B, T_out]
            unit_ids = None
            mask = mask_per_token.squeeze(2)  # [B, T_out]

        return tokens, timestamps, unit_ids, mask

    def _compute_token_mask(self, batch: BinnedSpikesInput, T_out: int, U: int) -> torch.Tensor:
        """Compute per-token validity mask from the input pad_and_gap_mask.

        The input mask has shape [B, T_input, U] at input bin resolution.
        We need to reduce it to [B, T_out, U] at the tokenizer's rate-bin resolution.
        A token is valid if ANY input bin within its range is valid for that unit.

        Args:
            batch: The input BinnedSpikesInput.
            T_out: Number of output time bins.
            U: Number of units.

        Returns:
            Boolean mask of shape [B, T_out, U]. True = valid token.
        """
        pad_mask = batch.pad_and_gap_mask  # [B, T_input, U_input], int, 1=valid

        B, T_input, U_input = pad_mask.shape

        # Compute ratio from input bins to token bins
        assert T_out > 0, "T_out must be > 0"
        ratio = T_input // T_out
        assert ratio > 0, f"Invalid ratio: T_input={T_input}, T_out={T_out}"

        # Truncate to the bins actually used by encode
        truncated = pad_mask[:, : T_out * ratio, :]  # [B, T_out*ratio, U_input]

        if U == 1 and U_input > 1:
            # Population tokenizer (compress_n): collapse all neurons into one.
            # A token is valid if ANY neuron in ANY input bin within that
            # token's time range is valid.
            reshaped = truncated.reshape(B, T_out, ratio, U_input)
            # sum over both ratio (time) and neurons, then check > 0
            token_mask = reshaped.sum(dim=(2, 3)) > 0  # [B, T_out], bool
            token_mask = token_mask.unsqueeze(2)  # [B, T_out, 1]
        else:
            assert U_input == U, (
                f"Unit dimension mismatch: pad_and_gap_mask has {U_input} units, but encode produced {U} units"
            )
            reshaped = truncated.reshape(B, T_out, ratio, U)
            # A token is valid if ANY of its constituent input bins is valid
            token_mask = reshaped.sum(dim=2) > 0  # [B, T_out, U], bool

        return token_mask
