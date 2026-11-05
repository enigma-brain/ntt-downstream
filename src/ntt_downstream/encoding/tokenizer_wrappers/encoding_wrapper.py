from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from ntt.core.components.types import BinnedSpikesInput


def build_binned_spikes_input(
    *,
    responses: torch.Tensor,
    session_key: str,
    sampling_rate: float,
    subject_id: str,
    species: str,
) -> Any:
    """
    Convert a ntt-downstream response batch into NTT `BinnedSpikesInput`.

    All fields are required by the NTT NamedTuple contract. ``pad_mask`` uses
    NTT convention where ``True`` = valid position, so we set all-ones for
    fixed-size inputs (no padding).
    """
    from ntt.core.components.types import BinnedSpikesInput

    if responses.ndim != 3:
        raise ValueError(f"Expected responses with shape (B, T, N), got {tuple(responses.shape)}")

    batch_size, num_bins, num_units = responses.shape
    device = responses.device

    return BinnedSpikesInput(
        spike_matrix=responses,
        pad_and_gap_mask=torch.ones((batch_size, num_bins, num_units), device=device, dtype=torch.bool),
        sampling_rate=float(sampling_rate),
        unit_ids=torch.arange(num_units, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1),
        unit_ids_pad_mask=torch.zeros((batch_size, num_units), device=device, dtype=torch.bool),
        duration_sec=torch.full(
            (batch_size,), float(num_bins) / float(sampling_rate), device=device, dtype=torch.float32
        ),
        session_id=session_key,
        subject_id=subject_id,
        species=species,
        absolute_start=torch.Tensor([0]),
        absolute_end=torch.Tensor([num_bins]),
    )


@dataclass
class CompressNSessionPlan:
    """Pre-computed chunking plan for encoding a session with a compress_n tokenizer.

    When a session has more neurons than the tokenizer's capacity (``spec.n_neurons``),
    neurons are split into balanced chunks.  Each chunk is zero-padded to
    ``pad_size`` before encoding.  This dataclass stores everything needed to
    reassemble the original neuron ordering at decode time.
    """

    session_key: str
    n_neurons_total: int  # Original N for this session
    chunk_neuron_slices: list[tuple[int, int]]  # [(start, end), ...] per chunk
    pad_size: int  # = spec.n_neurons — each chunk padded to this
    latent_n: int  # Latent channels produced per chunk
    num_chunks: int
    total_latent_queries: int  # = num_chunks * latent_n


def _make_balanced_slices(n_total: int, max_per_chunk: int) -> list[tuple[int, int]]:
    """Split ``n_total`` items into balanced chunks of at most ``max_per_chunk``.

    Returns a list of ``(start, end)`` slices whose sizes differ by at most 1.
    """
    num_chunks = math.ceil(n_total / max_per_chunk)
    base_size = n_total // num_chunks
    remainder = n_total % num_chunks
    slices: list[tuple[int, int]] = []
    cursor = 0
    for i in range(num_chunks):
        size = base_size + (1 if i < remainder else 0)
        slices.append((cursor, cursor + size))
        cursor += size
    return slices


@dataclass
class TokenizerWrapper:
    """Wraps an NTT tokenizer for use in the ntt-downstream training pipeline.

    Handles conversion between ntt-downstream tensors (B, T, N) and NTT's
    BinnedSpikesInput/TokenizerOutput types, plus discrete→float conversion
    for diffusion targets.

    NTT tokenizers return latents in a unified 4D shape ``[B, T, U, D]``.
    This wrapper handles neuron chunking (when a session has more neurons
    than the tokenizer capacity) and reshapes the 4D latents into the flat
    target format expected by downstream models.

    Supports two tokenizer families:

    * **compress_t** (``spec.compress_t=True, compress_n=False``): compresses the
      temporal dimension per neuron independently.  Neurons are processed in
      chunks of ``neuron_batch_size``.
    * **compress_n** (``spec.compress_n=True, compress_t=False``): compresses the
      neuron dimension while preserving time.  Neurons are split into balanced
      chunks of ``spec.n_neurons`` and each chunk is encoded separately.
    """

    # Typed ``Any`` on purpose: the wrapped tokenizer is one of several ntt
    # architectures (wanneural, ldns, conv_vae, wavelet-*, ...) with different
    # sub-module layouts, dispatched on ``tokenizer_kind``. Annotating it as
    # ``nn.Module`` makes pyright resolve every attribute via
    # ``Module.__getattr__`` to ``Tensor | Module``, which is wrong for all of them.
    tokenizer: Any
    mode: str  # "frozen", "trainable", or "finetune_decoder"
    sampling_rate: float
    reconstruction_rate_hz: float
    subject_id: str
    species: str
    tokenizer_kind: str | None
    discrete_target_mode: str
    discrete_target_vocab_size: int | None
    neuron_batch_size: int = 100
    latent_scale: float = 1.0
    latent_channel_means: torch.Tensor | None = None
    latent_channel_stds: torch.Tensor | None = None
    bsq_codebook_scale: float | None = None
    tokenizer_window_length_seconds: float | None = None
    # ---- Perceiver-style (compress_n + variable_n) extensions -------------
    #
    # The Perceiver-AE tokenizer produces a 4D output ``(B, M, C, L)`` instead
    # of the WanNeural/LDNS convention ``(B, T, U=1, D)``. M is the
    # population-compression factor (fixed, e.g. 16), C is the per-token
    # latent dim, and L is time. ``latent_layout`` controls how that gets
    # mapped to the unified ``(B, S, T, D)`` format expected by the
    # downstream encoding models:
    #
    #   - ``"m_as_spatial"``: permute to ``(B, S=M, T=L, D=C)``. Each of the
    #     M latent tokens is treated as a "neuron-like" spatial slot.
    #   - ``"m_flat"``: reshape to ``(B, S=1, T=L, D=M*C)``. The whole M-block
    #     is folded into the channel dim — a single high-dim population
    #     feature per time step.
    #
    # ``perceiver_rebin_factor`` is inferred from ``sampling_rate /
    # reconstruction_rate_hz`` at construction time. It determines how many
    # adjacent 1-ms bins get summed into one tokenizer input bin (1 = no
    # rebinning).
    latent_layout: str = "m_as_spatial"
    perceiver_rebin_factor: int = 1
    # ---- Discrete-indices extraction (CE loss) -----------------------------
    #
    # transformer-vq's ``tokenizer.encode()`` returns integer codebook indices already in
    # 4D ``(B, T, N, 1)`` format, which the wrapper just hands back when
    # ``raw_discrete=True``. WanNeural with FSQ/BSQ/VQPT bottlenecks instead
    # returns the *quantized continuous embedding* (``z_q``, float), and the
    # discrete indices have to be fetched via a separate call. ``vqpt`` with
    # ``num_quantizers > 1`` (residual VQ) also produces a trailing codebook
    # axis ``Q``, which the unified layout exposes as ``D=Q`` int64.
    #
    # ``discrete_indices_source`` controls how indices are reached. ``"auto"``
    # probes the tokenizer once at construction time; explicit overrides
    # (``"encode" | "bottleneck" | "forward"``) are useful when the auto-probe
    # cannot be run (e.g. tokenizers that need real session embeddings).
    #
    #   - ``"encode"``    → ``tokenizer.encode(...)`` already returns int indices.
    #   - ``"bottleneck"``→ run ``_extract_data``/``_pre_encode``/``encoder``
    #     manually and call ``bottleneck.get_indices(h)``. Skips the decoder.
    #   - ``"forward"``   → fall back to ``tokenizer(inp).tokens``. Wastes a
    #     decoder pass but is the most generic public API.
    #
    # ``num_codebooks`` is the trailing-axis size ``Q`` in the unified
    # ``(B, T, N, Q)`` layout. Grouped residual quantizers expose indices as
    # ``(G, B, T*N, Q_per_group)`` upstream; the wrapper packs those groups into
    # the trailing axis as ``Q = G * Q_per_group`` and unpacks them again for
    # decode.
    discrete_indices_source: str = "auto"
    num_codebooks: int | None = None
    # ---- Wavelet (compress_n + compress_t) extensions ----------------------
    #
    # Wavelet tokenizers (wavelet-*) are rigid at a fixed window
    # (e.g. 1s @ 1ms) and compress BOTH neurons and time. Cowley V4's response
    # is shorter than the tokenizer window (200ms vs 1s), so the wrapper
    # zero-pads each response to the tokenizer window before encoding. The
    # encoder is causal in time, so the leading K latent positions do not
    # depend on the zero-pad tail.
    #
    # `cosmos_target_slice` is the K (number of leading latent t positions
    # the model predicts). The wrapper slices encode targets to [0..K) and
    # at decode expands [K..T_lat_full) from a cached zero-pad-encoding bank.
    #
    # `cosmos_t_safe` is the decoder-safe horizon — the number of output
    # bins (in the response window) that are unaffected by the decoder's
    # small lookahead at the splice boundary. Used by the
    # encoding module to slice predictions and targets before metric
    # computation.
    cosmos_target_slice: int | None = None
    cosmos_t_safe: int | None = None

    def __post_init__(self):
        self._boilerplate_cache: dict = {}
        self._compress_n_plans: dict[str, CompressNSessionPlan] = {}
        self._compress_t_session_meta: dict[str, dict[str, int]] = {}
        self._tokenizer_device: torch.device | None = None
        # Resolved at first encode (or eagerly via _detect_discrete_metadata)
        self._indices_source_resolved: str | None = None
        self._num_codebooks_resolved: int | None = None
        # Lazy cache of zero-pad-encoding latents (wavelet). Shape
        # (1, T_lat_full, U_per_chunk, D); session-independent (the wavelet
        # tokenizer has no session conditioning).
        self._cosmos_zero_pad_latents: torch.Tensor | None = None
        self._cosmos_zero_pad_tokens: torch.Tensor | None = None
        self._init_window_and_validate()
        self._detect_discrete_metadata()

    def _align_tokenizer_device(self, target_device: torch.device) -> None:
        """Move the wrapped tokenizer onto ``target_device`` if it isn't already.

        ``TokenizerWrapper`` is a dataclass, not an ``nn.Module`` — Lightning's
        ``self.to(device)`` on the outer LightningModule never recurses into
        ``self.tokenizer_wrapper.tokenizer``. This helper snaps the tokenizer
        onto the input's device lazily, once per device change; same idea as
        the explicit ``to("cuda") / to("cpu")`` dance in the training modules'
        calibration blocks, but applied to every forward call.
        """
        try:
            current = next(self.tokenizer.parameters()).device
        except StopIteration:
            try:
                current = next(self.tokenizer.buffers()).device
            except StopIteration:
                current = torch.device("cpu")
        if current != target_device:
            self.tokenizer.to(target_device)
            self._tokenizer_device = target_device

    def _init_window_and_validate(self) -> None:
        """Validate ``latent_layout`` and default/validate the tokenizer window.

        Must run before ``_detect_discrete_metadata``: its probe calls
        ``_set_window_length_seconds_for_current_batch``, which overwrites
        ``spec.window_length_seconds`` with the probe length. Reading the spec
        after that would seed a variable-window (``Dist``) tokenizer's window
        from the probe — one token bin — and make ``_encode_compress_t`` chunk
        every response into one-token windows.
        """
        # Note: tokenizers with compress_n=True AND compress_t=True (wavelet
        # family) go through the dedicated _encode_cosmos / _decode_cosmos
        # arm, dispatched on `is_cosmos` before the compress_n / compress_t
        # paths.
        spec = getattr(self.tokenizer, "spec", None)
        # compress_n + variable_n=True is handled by the perceiver-style
        # path (_encode_perceiver / _decode_perceiver). Only the
        # fixed-n compress_n path goes through _encode_compress_n.

        if self.latent_layout not in ("m_as_spatial", "m_flat"):
            raise ValueError(f"latent_layout must be 'm_as_spatial' or 'm_flat', got {self.latent_layout!r}")

        # Default tokenizer_window_length_seconds from spec if not set.
        # Only seed when the spec value is a fixed scalar. If the tokenizer was
        # trained with a sampling distribution (``Dist`` subclass like
        # ``UniformDist``) for ``window_length_seconds``, leave the wrapper's
        # value as None — ``_set_window_length_seconds_for_current_batch``
        # overrides ``spec.window_length_seconds`` per batch with the actual
        # ``T / sampling_rate``, which is the correct behavior for variable-
        # window tokenizers. Callers that want explicit sub-window slicing
        # should set ``tokenizer.tokenizer_window_length_seconds`` in config.
        if self.tokenizer_window_length_seconds is None:
            if spec is not None:
                wls = getattr(spec, "window_length_seconds", None)
                if isinstance(wls, (int, float)) and not isinstance(wls, bool):
                    object.__setattr__(self, "tokenizer_window_length_seconds", float(wls))

        # Validate: must be a positive multiple of token_bin_size_ms / 1000
        if self.tokenizer_window_length_seconds is not None:
            token_bin_s = None
            if spec is not None:
                tbms = getattr(spec, "token_bin_size_ms", None)
                if tbms is not None:
                    token_bin_s = float(tbms) / 1000.0
            if token_bin_s is not None and token_bin_s > 0:
                ratio = self.tokenizer_window_length_seconds / token_bin_s
                if abs(ratio - round(ratio)) > 1e-6:
                    raise ValueError(
                        f"tokenizer_window_length_seconds ({self.tokenizer_window_length_seconds}) "
                        f"must be a multiple of token_bin_size_ms/1000 ({token_bin_s}). "
                        f"Ratio is {ratio:.4f}, expected integer."
                    )

    # -- Decoder fine-tuning support ------------------------------------------

    def get_decoder_params(self) -> list[torch.nn.Parameter]:
        """Return trainable decoder parameters for the wrapped tokenizer.

        Dispatches on ``tokenizer_kind`` to identify decoder-only sub-modules.
        The bottleneck is excluded for tokenizer types where it is shared between
        encode and decode paths (e.g., WanNeural's ``VAEBottleneck.post_conv``).
        """
        import logging

        logger = logging.getLogger(__name__)
        kind = (self.tokenizer_kind or "").lower()

        if kind == "wanneural":
            return list(self.tokenizer.decoder.parameters())

        if kind == "simple_vae":
            return list(self.tokenizer.decoder.parameters())

        if kind == "conv_vae":
            params = list(self.tokenizer.decoder.parameters())
            proj_out = getattr(self.tokenizer, "neuron_proj_out", None)
            if proj_out is not None:
                params.extend(proj_out.parameters())
            return params

        if kind == "ldns":
            ae = self.tokenizer.count_wrapper.ae_net
            params: list[torch.nn.Parameter] = []
            params.extend(ae.unbottleneck.parameters())
            for block in ae.decoder:
                params.extend(block.parameters())
            params.extend(ae.decoder_out.parameters())
            return params

        if kind == "compressed_sensing":
            logger.warning(
                "CompressedSensingTokenizer has no learned decoder parameters (FISTA solver). "
                "Decoder fine-tuning has no effect."
            )
            return []

        raise NotImplementedError(
            f"Decoder parameter identification not implemented for tokenizer kind '{kind}'. "
            f"Add support in TokenizerWrapper.get_decoder_params()."
        )

    def enable_decoder_finetuning(self) -> None:
        """Selectively unfreeze decoder parameters and set decoder modules to train mode.

        Call after ``_configure_tokenizer_mode(tokenizer, 'finetune_decoder')`` which
        first freezes everything. This method then re-enables gradients for decoder-only
        parameters.
        """
        import logging

        logger = logging.getLogger(__name__)
        decoder_params = self.get_decoder_params()
        if not decoder_params:
            logger.warning("No decoder parameters found for fine-tuning.")
            return
        for p in decoder_params:
            p.requires_grad = True

        # Put decoder sub-modules in train mode for dropout / batchnorm
        kind = (self.tokenizer_kind or "").lower()
        if kind == "wanneural":
            self.tokenizer.decoder.train()
        elif kind == "simple_vae":
            self.tokenizer.decoder.train()
        elif kind == "conv_vae":
            self.tokenizer.decoder.train()
            proj_out = getattr(self.tokenizer, "neuron_proj_out", None)
            if proj_out is not None:
                proj_out.train()
        elif kind == "ldns":
            ae = self.tokenizer.count_wrapper.ae_net
            ae.unbottleneck.train()
            for block in ae.decoder:
                block.train()
            ae.decoder_out.train()

        n_params = sum(p.numel() for p in decoder_params)
        logger.info(f"Enabled decoder fine-tuning: {n_params:,} parameters unfrozen")

    def calibrate_latent_scale(self, dataloader, max_batches: int = 50) -> None:
        """Compute ``latent_scale`` from a sample of training data.

        Encodes ``max_batches`` of responses, measures the global std of the
        resulting latent tokens, and sets ``self.latent_scale`` to that value.
        Call this after construction when ``latent_scale`` was set to ``"auto"``.
        """
        import logging

        logger = logging.getLogger(__name__)
        # Temporarily force scale=1.0 so we measure the raw (unscaled) token std
        object.__setattr__(self, "latent_scale", 1.0)

        # Infer device: try tokenizer parameters first, fall back to buffers, then CPU
        device = next(
            (p.device for p in self.tokenizer.parameters()),
            next((b.device for b in self.tokenizer.buffers()), torch.device("cpu")),
        )

        # Accumulate sufficient statistics for global variance
        total_sum = 0.0
        total_sum_sq = 0.0
        total_count = 0
        n_batches = 0
        with torch.no_grad():
            for i, (k, b) in enumerate(dataloader):
                if i >= max_batches:
                    break
                responses = b["response"].to(device, torch.float32)
                # Dataset returns (B, N, T); encode expects (B, T, N)
                tokens = self.encode(responses.transpose(1, 2), k, raw_discrete=False)["target"]
                flat = tokens.float()
                total_sum += flat.sum().item()
                total_sum_sq += flat.pow(2).sum().item()
                total_count += flat.numel()
                n_batches += 1

        if total_count == 0:
            logger.warning("calibrate_latent_scale: no data, keeping scale=1.0")
            return

        global_mean = total_sum / total_count
        global_var = total_sum_sq / total_count - global_mean**2
        global_std = max(global_var, 0.0) ** 0.5

        if global_std < 1e-6:
            logger.warning(f"calibrate_latent_scale: computed std={global_std:.6f} is near zero, keeping scale=1.0")
            return
        object.__setattr__(self, "latent_scale", global_std)
        logger.info(f"calibrate_latent_scale: latent_scale={self.latent_scale:.4f} (from {n_batches} batches)")

    def calibrate_latent_scale_per_channel(self, dataloader, max_batches: int = 50) -> None:
        """Compute per-channel mean/std from training data.

        For tokens in unified ``(B, S, T, D)`` format, computes mean and std
        per latent channel D (across samples, spatial units, and time).
        """
        import logging

        logger = logging.getLogger(__name__)
        # Temporarily disable any scaling so we measure raw tokens
        old_means = self.latent_channel_means
        old_stds = self.latent_channel_stds
        old_scale = self.latent_scale
        object.__setattr__(self, "latent_channel_means", None)
        object.__setattr__(self, "latent_channel_stds", None)
        object.__setattr__(self, "latent_scale", 1.0)

        device = next(
            (p.device for p in self.tokenizer.parameters()),
            next((b.device for b in self.tokenizer.buffers()), torch.device("cpu")),
        )

        all_tokens = []
        with torch.no_grad():
            for i, (k, b) in enumerate(dataloader):
                if i >= max_batches:
                    break
                responses = b["response"].to(device, torch.float32)
                # Dataset returns (B, N, T); encode expects (B, T, N)
                tokens = self.encode(responses.transpose(1, 2), k, raw_discrete=False)["target"]
                all_tokens.append(tokens.cpu())

        if not all_tokens:
            logger.warning("calibrate_latent_scale_per_channel: no data, keeping defaults")
            object.__setattr__(self, "latent_channel_means", old_means)
            object.__setattr__(self, "latent_channel_stds", old_stds)
            object.__setattr__(self, "latent_scale", old_scale)
            return

        cat = torch.cat(all_tokens, dim=0)  # (B, S, T, D) unified format

        # D is always the last dim; compute mean/std over all other dims
        reduce_dims = tuple(range(cat.ndim - 1))  # (0, 1, 2) for 4D
        means = cat.mean(dim=reduce_dims)  # (D,)
        stds = cat.std(dim=reduce_dims).clamp(min=1e-6)  # (D,)
        # Reshape for broadcasting: (1, 1, 1, D)
        shape = [1] * (cat.ndim - 1) + [-1]
        object.__setattr__(self, "latent_channel_means", means.reshape(*shape))
        object.__setattr__(self, "latent_channel_stds", stds.reshape(*shape))

        # Set via object.__setattr__ above, which pyright can't see through.
        assert self.latent_channel_means is not None and self.latent_channel_stds is not None
        logger.info(
            f"calibrate_latent_scale_per_channel: "
            f"means={self.latent_channel_means.flatten().tolist()}, "
            f"stds={self.latent_channel_stds.flatten().tolist()} "
            f"(from {len(all_tokens)} batches)"
        )

    # -- Discrete-indices detection -------------------------------------------

    def _make_probe_input(self, num_bins: int):
        """Build a tiny ``BinnedSpikesInput`` for one-shot tokenizer probes.

        Used by :meth:`get_t_comp` and :meth:`_detect_discrete_metadata` to
        run the encoder once without involving the dataloader. ``session_id``
        is the placeholder ``"_probe_"`` — fine for tokenizers (like wanneural)
        whose encoder doesn't look up per-session embeddings, but fails for
        Perceiver / NuCLR-style models.
        """
        from ntt.core.components.types import BinnedSpikesInput

        nbs = self.neuron_batch_size
        device = next(self.tokenizer.parameters()).device
        dummy = torch.zeros(1, num_bins, nbs, device=device)
        pad_mask, unit_ids, uid_pm, dur = self._get_boilerplate_tensors(1, num_bins, nbs, device)
        return BinnedSpikesInput(
            spike_matrix=dummy,
            pad_and_gap_mask=pad_mask,
            sampling_rate=float(self.sampling_rate),
            unit_ids=unit_ids,
            unit_ids_pad_mask=uid_pm,
            duration_sec=dur,
            session_id="_probe_",
            subject_id=self.subject_id,
            species=self.species,
            absolute_start=torch.Tensor([0]),
            absolute_end=torch.Tensor([num_bins]),
        )

    def _detect_discrete_metadata(self) -> None:
        """Probe the tokenizer once to settle indices source and ``num_codebooks``.

        Runs eagerly during ``__post_init__`` so model-construction code can
        read :attr:`num_codebooks_effective` synchronously (e.g. to size the
        cross-entropy head).

        For continuous-latent tokenizers and Perceiver-style ones (which need
        real session embeddings to encode), detection is skipped and
        single-codebook ``encode`` semantics are assumed — those cases never
        hit the discrete-indices path anyway.
        """
        import logging

        logger = logging.getLogger(__name__)

        spec = getattr(self.tokenizer, "spec", None)
        latent_type = str(getattr(spec, "latent_type", "")).lower() if spec is not None else ""
        is_discrete = latent_type.endswith("discrete")

        # Defaults for single-codebook tokenizers (transformer-vq, internal name CALM)
        resolved_source = self.discrete_indices_source if self.discrete_indices_source != "auto" else "encode"
        resolved_q = int(self.num_codebooks) if self.num_codebooks is not None else 1

        if not is_discrete or self.is_perceiver:
            self._indices_source_resolved = resolved_source
            self._num_codebooks_resolved = resolved_q
            return

        inferred_q = self._infer_bottleneck_num_codebooks()
        if self.num_codebooks is None and inferred_q is not None:
            resolved_q = inferred_q

        # Resolve a probe window length: prefer the spec's natural window, else
        # token_bin_size_ms × 1, else 1 bin. The exact value doesn't matter for
        # source/Q detection — only the dtype and last-dim shape do.
        num_bins = 1
        wls = getattr(spec, "window_length_seconds", None) if spec is not None else None
        if isinstance(wls, (int, float)) and not isinstance(wls, bool) and wls > 0:
            num_bins = max(1, int(round(float(wls) * float(self.sampling_rate))))
        else:
            tbms = getattr(spec, "token_bin_size_ms", None) if spec is not None else None
            if isinstance(tbms, (int, float)) and not isinstance(tbms, bool) and tbms > 0:
                num_bins = max(1, int(round(float(tbms) / 1000.0 * float(self.sampling_rate))))

        try:
            inp = self._make_probe_input(num_bins)
            self._set_window_length_seconds_for_current_batch(num_bins)
        except Exception as e:
            logger.warning(f"discrete-metadata probe: could not build probe input ({e}); using defaults.")
            self._indices_source_resolved = resolved_source
            self._num_codebooks_resolved = resolved_q
            return

        # If user pinned the source, skip auto-detection and only probe Q
        if self.discrete_indices_source != "auto":
            try:
                q = self._probe_num_codebooks(inp, source=self.discrete_indices_source)
            except Exception as e:
                logger.warning(
                    f"discrete-metadata probe failed for source={self.discrete_indices_source!r}: {e}; "
                    f"falling back to num_codebooks={resolved_q}."
                )
                q = resolved_q
            self._indices_source_resolved = self.discrete_indices_source
            self._num_codebooks_resolved = int(self.num_codebooks) if self.num_codebooks is not None else q
            return

        # Auto: probe encode dtype first
        try:
            with torch.no_grad():
                z = self.tokenizer.encode(inp)
        except Exception as e:
            logger.warning(f"discrete-metadata probe: tokenizer.encode failed ({e}); using defaults.")
            self._indices_source_resolved = "encode"
            self._num_codebooks_resolved = resolved_q
            return

        if not z.is_floating_point():
            # transformer-vq-style: encode already returns indices in unified 4D format
            q = int(z.shape[-1]) if z.ndim == 4 else 1
            self._indices_source_resolved = "encode"
            self._num_codebooks_resolved = int(self.num_codebooks) if self.num_codebooks is not None else q
            logger.info(
                f"tokenizer indices source: encode (transformer-vq-style). num_codebooks={self._num_codebooks_resolved}"
            )
            return

        # Encode returned float (e.g. wanneural-VQ): need separate indices fetch
        bn = getattr(self.tokenizer, "bottleneck", None)
        can_use_bottleneck = (
            bn is not None
            and hasattr(bn, "get_indices")
            and hasattr(self.tokenizer, "_extract_data")
            and hasattr(self.tokenizer, "_pre_encode")
            and hasattr(self.tokenizer, "encoder")
        )
        if can_use_bottleneck:
            try:
                q = self._probe_num_codebooks(inp, source="bottleneck")
                self._indices_source_resolved = "bottleneck"
                self._num_codebooks_resolved = int(self.num_codebooks) if self.num_codebooks is not None else q
                logger.info(
                    "tokenizer indices source: bottleneck.get_indices "
                    f"(wanneural-style). num_codebooks={self._num_codebooks_resolved}"
                )
                return
            except Exception as e:
                logger.warning(f"bottleneck.get_indices probe failed ({e}); falling back to forward().")

        # Last resort: tokenizer.forward(inp).tokens (wastes a decoder pass)
        try:
            q = self._probe_num_codebooks(inp, source="forward")
            self._indices_source_resolved = "forward"
            self._num_codebooks_resolved = int(self.num_codebooks) if self.num_codebooks is not None else q
            logger.info(f"tokenizer indices source: forward().tokens. num_codebooks={self._num_codebooks_resolved}")
        except Exception as e:
            logger.warning(
                f"forward().tokens probe failed ({e}); CE-loss training on this tokenizer is unlikely to work."
            )
            self._indices_source_resolved = "encode"
            self._num_codebooks_resolved = resolved_q

    def _probe_num_codebooks(self, inp, source: str) -> int:
        """Run a tiny encode through ``source`` and report the trailing Q dim."""
        with torch.no_grad():
            if source == "encode":
                z = self.tokenizer.encode(inp)
                return int(z.shape[-1]) if z.ndim == 4 else 1
            if source == "bottleneck":
                data, neuron_mask = self.tokenizer._extract_data(inp)
                h = self.tokenizer._pre_encode(data)
                h = self.tokenizer.encoder(h, neuron_mask=neuron_mask)
                indices = self.tokenizer.bottleneck.get_indices(h)
                if indices.ndim == 4:
                    return int(indices.shape[0]) * int(indices.shape[-1])
                return int(indices.shape[-1]) if indices.ndim >= 3 else 1
            if source == "forward":
                tokens = self.tokenizer(inp).tokens
                if tokens is None:
                    raise RuntimeError("tokenizer.forward() returned no .tokens")
                return int(tokens.shape[-1]) if tokens.ndim >= 3 else 1
        raise ValueError(f"Unknown indices source: {source!r}")

    def _infer_bottleneck_num_codebooks(self) -> int | None:
        """Infer unified trailing Q from vqpt bottleneck metadata, if available."""
        bn = getattr(self.tokenizer, "bottleneck", None)
        quantizer = getattr(bn, "quantizer", None)
        if quantizer is None:
            return None

        groups = int(getattr(quantizer, "groups", 1) or 1)
        q_per_group = getattr(quantizer, "num_quantizers", None)
        if q_per_group is None:
            rvqs = getattr(quantizer, "rvqs", None)
            if rvqs is not None and len(rvqs) > 0:
                q_per_group = getattr(rvqs[0], "num_quantizers", None)
        if q_per_group is None:
            q_per_group = 1
        return groups * int(q_per_group)

    @property
    def num_codebooks_effective(self) -> int:
        """Resolved trailing-axis size ``Q`` for discrete tokens (1 = single codebook)."""
        if self._num_codebooks_resolved is None:
            self._detect_discrete_metadata()
        assert self._num_codebooks_resolved is not None
        return self._num_codebooks_resolved

    @property
    def indices_source(self) -> str:
        """Resolved indices source: ``"encode" | "bottleneck" | "forward"``."""
        if self._indices_source_resolved is None:
            self._detect_discrete_metadata()
        assert self._indices_source_resolved is not None
        return self._indices_source_resolved

    def _extract_chunk_indices(self, inp) -> torch.Tensor:
        """Return discrete indices for one chunk in ``(B, T_comp, N_pad, Q)`` int64.

        ``T_comp`` and ``N_pad`` are inferred from the tokenizer's own output —
        callers don't need to pass them in.

        Sources:
          - ``"encode"``    : ``tokenizer.encode(inp)`` already returns 4D int.
          - ``"bottleneck"``: run encoder manually, call ``bottleneck.get_indices(h)``.
            ``h`` has shape ``(B, C, T_comp, N_pad)`` so ``T_comp``/``N_pad``
            come from there.
          - ``"forward"``   : run ``tokenizer(inp)``; use ``out.latents`` shape
            ``(B, T_comp, N_pad, C)`` to recover ``T_comp``/``N_pad``.
        """
        source = self.indices_source

        if source == "encode":
            z = self.tokenizer.encode(inp)
            if z.ndim != 4:
                raise ValueError(f"encode-source indices: expected 4D, got {tuple(z.shape)}")
            return z.to(torch.long)

        if source == "bottleneck":
            data, neuron_mask = self.tokenizer._extract_data(inp)
            h = self.tokenizer._pre_encode(data)
            h = self.tokenizer.encoder(h, neuron_mask=neuron_mask)
            # h is (B, C, T_comp, N_pad)
            t_comp = int(h.shape[2])
            n_pad = int(h.shape[3])
            indices = self.tokenizer.bottleneck.get_indices(h)
            return self._unflatten_tn_indices(indices, t_comp, n_pad)

        if source == "forward":
            out = self.tokenizer(inp)
            if out.tokens is None:
                raise RuntimeError("forward-source indices: tokenizer.forward returned no .tokens")
            if out.latents is None or out.latents.ndim != 4:
                raise RuntimeError(
                    "forward-source indices: tokenizer.forward returned no usable latents shape; "
                    "cannot recover (T_comp, N_pad)."
                )
            t_comp = int(out.latents.shape[1])
            n_pad = int(out.latents.shape[2])
            return self._unflatten_tn_indices(out.tokens, t_comp, n_pad)

        raise ValueError(f"Unknown indices_source: {source!r}")

    @staticmethod
    def _unflatten_tn_indices(indices: torch.Tensor, T: int, N: int) -> torch.Tensor:
        """Reshape flattened int indices to unified ``(B, T, N, Q)`` int64.

        Layout follows the wanneural ``VQPTBottleneck._to_seq`` convention
        (``rearrange("b c t n -> b (t n) c")``): time is the major axis, neuron
        the minor axis inside the flattened sequence.

        Grouped vqpt variants return ``(G, B, T*N, Q_per_group)``; pack the
        leading group axis into the unified trailing codebook axis so the CE
        head learns one categorical distribution per group/codebook slot.
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
        if indices.ndim == 4:
            groups, B, S, q_per_group = indices.shape
            if S != T * N:
                raise ValueError(f"indices length {S} != T*N = {T}*{N}={T * N}")
            packed = indices.permute(1, 2, 0, 3).reshape(B, S, groups * q_per_group)
            return packed.reshape(B, T, N, groups * q_per_group).to(torch.long)
        raise ValueError(f"unexpected indices ndim={indices.ndim} (shape {tuple(indices.shape)})")

    # -- compress_t token layout helpers ---------------------------------------

    def get_t_comp(self, num_bins: int) -> int:
        """Return the temporal compression factor for a given input length.

        Performs a quick dummy encode with a single sample to measure the
        actual number of latent time steps produced by the tokenizer.
        """
        from ntt.core.components.types import BinnedSpikesInput

        # Perceiver-style tokenizers preserve time and rebin the input by
        # ``perceiver_rebin_factor``. Compute this analytically instead of
        # probing — the probe would fail because a dummy session_id is not in
        # the NuCLR embeddings lookup.
        if self.is_perceiver:
            return max(1, num_bins // max(1, int(self.perceiver_rebin_factor)))

        nbs = self.neuron_batch_size
        self._set_window_length_seconds_for_current_batch(num_bins)
        device = next(self.tokenizer.parameters()).device
        dummy = torch.zeros(1, num_bins, nbs, device=device)
        pad_mask, unit_ids, uid_pm, dur = self._get_boilerplate_tensors(1, num_bins, nbs, device)
        inp = BinnedSpikesInput(
            spike_matrix=dummy,
            pad_and_gap_mask=pad_mask,
            sampling_rate=float(self.sampling_rate),
            unit_ids=unit_ids,
            unit_ids_pad_mask=uid_pm,
            duration_sec=dur,
            session_id="_probe_",
            subject_id=self.subject_id,
            species=self.species,
            absolute_start=torch.Tensor([0]),
            absolute_end=torch.Tensor([num_bins]),
        )
        with torch.no_grad():
            z = self.tokenizer.encode(inp)  # [1, T_latent, nbs, D]
        if z.ndim != 4:
            raise ValueError(f"Expected 4D encode output [B, T, U, D], got shape {tuple(z.shape)}")
        return z.shape[1]

    # -- compress_n properties ------------------------------------------------

    @property
    def is_compress_n(self) -> bool:
        spec = getattr(self.tokenizer, "spec", None)
        if spec is None:
            return False
        return bool(getattr(spec, "compress_n", False))

    @property
    def is_perceiver(self) -> bool:
        """Return True for Perceiver-family tokenizers (variable-n compress_n)."""
        spec = getattr(self.tokenizer, "spec", None)
        if spec is None:
            return False
        kind = str(getattr(spec, "KIND", "")).lower()
        if "perceiver" in kind:
            return True
        # Fallback heuristic: compress_n with variable_n=True behaves like a
        # Perceiver for our purposes even if KIND changes upstream.
        return bool(getattr(spec, "compress_n", False)) and bool(getattr(spec, "variable_n", False))

    @property
    def is_cosmos(self) -> bool:
        """Return True for wavelet-family tokenizers (compress_n AND compress_t)."""
        spec = getattr(self.tokenizer, "spec", None)
        if spec is None:
            return False
        kind = str(getattr(spec, "KIND", "")).lower()
        if kind.startswith("cosmos"):
            return True
        # Defensive: any non-perceiver tokenizer that compresses both axes
        # should route through the wavelet arm too.
        return (
            bool(getattr(spec, "compress_n", False))
            and bool(getattr(spec, "compress_t", False))
            and not bool(getattr(spec, "variable_n", False))
        )

    @property
    def produces_discrete_tokens(self) -> bool:
        """Return True when the tokenizer's bottleneck emits discrete codebook indices.

        This is the single runtime signal separating the discrete family
        (transformer-vq / BSQ / FSQ / VQPT) from the continuous family (VAE / flow /
        diffusion / mse). It is the exact ``spec.latent_type`` predicate used by
        :meth:`_target_to_tokens` and :meth:`_detect_discrete_metadata`, so the
        roundtrip ceiling diagnostic branches on the same notion of discreteness
        the training/loss code already relies on. For discrete tokenizers the
        ``raw_discrete=True`` encode → decode-from-indices path is correct; for
        continuous ones the roundtrip must instead mirror the training decode
        path (``_target_to_tokens`` → ``decode_differentiable``).
        """
        spec = getattr(self.tokenizer, "spec", None)
        latent_type = str(getattr(spec, "latent_type", "")).lower() if spec is not None else ""
        return latent_type.endswith("discrete")

    @property
    def perceiver_M(self) -> int:
        """Number of latent tokens (M) the perceiver encoder uses."""
        spec = self.tokenizer.spec
        val = getattr(spec, "latent_tokens", None)
        if val is None:
            raise ValueError("perceiver spec is missing 'latent_tokens'")
        return int(val)

    @property
    def perceiver_C(self) -> int:
        """Per-token latent feature dim (C). For perceiver this is ``latent_dim``."""
        spec = self.tokenizer.spec
        val = getattr(spec, "latent_dim", None)
        if val is None:
            raise ValueError("perceiver spec is missing 'latent_dim'")
        return int(val)

    @property
    def unified_latent_dim(self) -> int:
        """Last-dim (D) of the unified ``(B, S, T, D)`` layout produced by encode().

        Resolves to:
          - ``C`` when ``latent_layout='m_as_spatial'``
          - ``M * C`` when ``latent_layout='m_flat'``
        """
        if not self.is_perceiver:
            return self.latent_dim
        if self.latent_layout == "m_as_spatial":
            return self.perceiver_C
        return self.perceiver_M * self.perceiver_C

    @property
    def unified_num_spatial(self) -> int:
        """S (spatial dim) of the unified layout — matches ``n_neurons_override``."""
        if not self.is_perceiver:
            raise ValueError("unified_num_spatial is only defined for Perceiver tokenizers")
        return self.perceiver_M if self.latent_layout == "m_as_spatial" else 1

    @property
    def latent_dim(self) -> int:
        """Return the latent dimension D of the tokenizer output ``[B, T, U, D]``.

        For compress_t tokenizers this is ``spec.z_dim``.
        For compress_n tokenizers this is ``spec.latent_dim`` (same as :attr:`latent_n`).
        For Perceiver-family tokenizers this is layout-aware: returns ``C`` for
        ``m_as_spatial`` and ``M*C`` for ``m_flat`` — i.e. the last-dim actually
        emitted by :meth:`encode` in the unified ``(B, S, T, D)`` format. This
        matters for the audio encoding module, which auto-derives
        ``token_prediction_dim`` from this property.
        """
        spec = getattr(self.tokenizer, "spec", None)
        if spec is None:
            raise ValueError("Cannot determine latent_dim: tokenizer has no spec")
        # Perceiver: reflect the unified-layout D the wrapper actually emits.
        if self.is_perceiver:
            return self.unified_latent_dim
        # compress_n uses latent_dim
        if self.is_compress_n:
            latent_n = self.latent_n
            assert latent_n is not None  # latent_n is only None when not is_compress_n
            return latent_n
        # compress_t / default: z_dim
        z_dim = getattr(spec, "z_dim", None)
        if z_dim is not None:
            return int(z_dim)
        # Fallback: try latent_dim
        latent_dim = getattr(spec, "latent_dim", None)
        if latent_dim is not None:
            return int(latent_dim)
        if getattr(spec, "has_latent_dim", None) is False:
            return 1
        # Fallback: embed_dim (used by transformer-vq and other transformer-based tokenizers)
        embed_dim = getattr(spec, "embed_dim", None)
        if embed_dim is not None:
            return int(embed_dim)
        raise ValueError(
            "Cannot determine latent_dim from tokenizer spec (no z_dim, latent_dim, or embed_dim attribute)"
        )

    @property
    def temporal_compression_factor(self) -> int | None:
        """Return the total temporal compression ratio, or None if unavailable.

        For compress_n tokenizers (which preserve time) this returns None.
        """
        if self.is_compress_n:
            return None
        spec = getattr(self.tokenizer, "spec", None)
        if spec is None:
            return None
        val = getattr(spec, "temporal_compression", None)
        if val is not None:
            return int(val)
        token_bin_size_ms = getattr(spec, "token_bin_size_ms", None)
        bin_size_ms = getattr(spec, "bin_size_ms", None)
        if (
            isinstance(token_bin_size_ms, (int, float))
            and not isinstance(token_bin_size_ms, bool)
            and isinstance(bin_size_ms, (int, float))
            and not isinstance(bin_size_ms, bool)
            and bin_size_ms > 0
        ):
            ratio = float(token_bin_size_ms) / float(bin_size_ms)
            if abs(ratio - round(ratio)) <= 1e-6:
                return int(round(ratio))
        return None

    @property
    def latent_n(self) -> int | None:
        """Number of latent neuron channels per chunk (only for compress_n)."""
        if not self.is_compress_n:
            return None
        spec = self.tokenizer.spec
        val = getattr(spec, "latent_dim", None)
        if val is None:
            raise ValueError("compress_n tokenizer spec is missing 'latent_dim'")
        return int(val)

    @property
    def max_neurons_per_chunk(self) -> int | None:
        """Tokenizer capacity — max neurons per encode call (only for compress_n)."""
        if not self.is_compress_n:
            return None
        spec = self.tokenizer.spec
        val = getattr(spec, "n_neurons", None)
        if val is None or val is False:
            raise ValueError("compress_n tokenizer spec must have a positive integer 'n_neurons'")
        return int(val)

    def get_or_create_plan(self, session_key: str, n_neurons: int) -> CompressNSessionPlan:
        """Return the chunking plan for *session_key*, creating it if needed."""
        if session_key in self._compress_n_plans:
            plan = self._compress_n_plans[session_key]
            if plan.n_neurons_total != n_neurons:
                raise ValueError(
                    f"Session '{session_key}' was registered with {plan.n_neurons_total} neurons "
                    f"but encode/decode called with {n_neurons}."
                )
            return plan

        # Perceiver-style: variable_n means the whole population goes through
        # one encode call (no chunking). The plan just records what the
        # unified layout's S will be so readout sizing stays consistent.
        if self.is_perceiver:
            s = self.unified_num_spatial  # M or 1 depending on latent_layout
            d = self.unified_latent_dim
            plan = CompressNSessionPlan(
                session_key=session_key,
                n_neurons_total=n_neurons,
                chunk_neuron_slices=[(0, n_neurons)],  # one chunk = full population
                pad_size=n_neurons,
                latent_n=d,
                num_chunks=s,
                total_latent_queries=s * d,
            )
            self._compress_n_plans[session_key] = plan
            return plan

        M = self.max_neurons_per_chunk
        latent_n = self.latent_n
        # Both are None only for non-compress_n tokenizers, which never reach this
        # chunking path (None would already fail in _make_balanced_slices).
        assert M is not None and latent_n is not None
        slices = _make_balanced_slices(n_neurons, M)
        plan = CompressNSessionPlan(
            session_key=session_key,
            n_neurons_total=n_neurons,
            chunk_neuron_slices=slices,
            pad_size=M,
            latent_n=latent_n,
            num_chunks=len(slices),
            total_latent_queries=len(slices) * latent_n,
        )
        self._compress_n_plans[session_key] = plan
        return plan

    def get_num_chunks(self, session_key: str, n_neurons: int) -> int:
        """Return the number of neuron chunks for a session (creates plan if needed)."""
        return self.get_or_create_plan(session_key, n_neurons).num_chunks

    def _get_boilerplate_tensors(self, batch_size: int, num_bins: int, num_units: int, device: torch.device):
        """Return cached boilerplate tensors for BinnedSpikesInput construction."""
        key = (batch_size, num_bins, num_units, str(device))
        if key not in self._boilerplate_cache:
            self._boilerplate_cache[key] = (
                torch.ones((batch_size, num_bins, num_units), device=device, dtype=torch.bool),
                torch.arange(num_units, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1),
                torch.zeros((batch_size, num_units), device=device, dtype=torch.bool),
                torch.full(
                    (batch_size,), float(num_bins) / float(self.sampling_rate), device=device, dtype=torch.float32
                ),
            )
        return self._boilerplate_cache[key]

    @torch.compiler.disable
    def encode(
        self,
        responses: torch.Tensor,
        session_key: str,
        raw_discrete: bool = False,
        skip_target_scaling: bool = False,
    ) -> dict[str, Any]:
        """Encode responses to training targets via the tokenizer.

        Returns tokens in the unified ``(B, S, T, D)`` layout:

        * **S** = spatial units (N neurons for compress_t, num_chunks for compress_n)
        * **T** = temporal tokens (t_comp for compress_t, n_time_bins for compress_n)
        * **D** = token dimension (z_dim for compress_t, latent_n for compress_n)

        Args:
            responses: ``(B, T_data, N)`` neural responses.
            session_key: Session identifier.
            raw_discrete: If True and the tokenizer produces discrete tokens,
                return raw integer indices without float conversion.
            skip_target_scaling: If True, return raw float tokens without
                ``_tokens_to_target`` normalization. Use this for BCE loss
                where the binarization threshold must operate on unscaled latents.

        Returns dict with:
            target: ``(B, S, T, D)`` float tensor (or long if raw_discrete)
            n_spatial: int — S for this session
            n_temporal: int — T
            aux_loss: None (encode path has no auxiliary loss)
        """
        if responses.ndim != 3:
            raise ValueError(f"Expected responses with shape (B, T, N), got {tuple(responses.shape)}")

        # The wrapper is not an nn.Module, so Lightning's .to(device) on the
        # LightningModule does not recurse into ``self.tokenizer``. If the
        # LightningModule moved to CUDA but the tokenizer is still on CPU
        # (as left by ``build_tokenizer_wrapper(..., device="cpu")``), the
        # forward would either error on device mismatch or — worse — silently
        # run on CPU for minutes per step. Snap the tokenizer to the input's
        # device once, lazily.
        self._align_tokenizer_device(responses.device)

        if self.is_perceiver:
            return self._encode_perceiver(
                responses, session_key, raw_discrete=raw_discrete, skip_target_scaling=skip_target_scaling
            )
        if self.is_cosmos:
            return self._encode_cosmos(
                responses, session_key, raw_discrete=raw_discrete, skip_target_scaling=skip_target_scaling
            )
        if self.is_compress_n:
            return self._encode_compress_n(
                responses, session_key, raw_discrete=raw_discrete, skip_target_scaling=skip_target_scaling
            )
        return self._encode_compress_t(
            responses, session_key, raw_discrete=raw_discrete, skip_target_scaling=skip_target_scaling
        )

    def encode_windowed(
        self,
        responses: torch.Tensor,
        session_key: str,
        raw_discrete: bool = False,
        skip_target_scaling: bool = False,
    ) -> dict[str, Any]:
        """Encode responses in fixed-size windows.

        Slices the full response tensor into chunks of ``tokenizer_window_length_seconds``
        duration, batch-encodes all chunks via the existing ``encode()`` method, and
        reassembles the output into a single temporal sequence.

        Args:
            responses: ``(B, T, N)`` full response tensor.
            session_key: Session identifier.
            raw_discrete: If True, return raw discrete indices.
            skip_target_scaling: If True, return raw float tokens without normalization.

        Returns dict with:
            target: ``(B, W * T_tok, S, D)`` unified latent target
            n_windows: int
            tokens_per_window: int
            n_spatial: int
            n_temporal: int
        """
        import logging

        logger = logging.getLogger(__name__)

        if self.tokenizer_window_length_seconds is None:
            raise ValueError(
                "tokenizer_window_length_seconds is not set. "
                "Set it explicitly or ensure tokenizer spec has window_length_seconds."
            )

        B, T_full, N = responses.shape
        M = int(round(self.tokenizer_window_length_seconds * self.sampling_rate))
        if M <= 0:
            raise ValueError(
                f"Computed window bins M={M} from "
                f"tokenizer_window_length_seconds={self.tokenizer_window_length_seconds}, "
                f"sampling_rate={self.sampling_rate}"
            )

        W = T_full // M
        if W == 0:
            raise ValueError(
                f"Response length T={T_full} is shorter than one window M={M} bins "
                f"(tokenizer_window_length_seconds={self.tokenizer_window_length_seconds})"
            )
        T_trunc = W * M

        # Reshape: (B, T_trunc, N) -> (B*W, M, N)
        windowed = responses[:, :T_trunc, :].reshape(B * W, M, N)

        # Encode all windows as one big batch
        result = self.encode(windowed, session_key, raw_discrete=raw_discrete, skip_target_scaling=skip_target_scaling)
        target_flat = result["target"]  # (B*W, S, T_tok, D)

        T_tok = target_flat.shape[2]
        if not hasattr(self, "_windowed_validated"):
            spec = getattr(self.tokenizer, "spec", None)
            tbms = getattr(spec, "token_bin_size_ms", None) if spec else None
            if tbms is not None:
                expected_coverage = T_tok * float(tbms) / 1000.0
                if abs(expected_coverage - self.tokenizer_window_length_seconds) > 1e-4:
                    msg = (
                        f"Tokenizer produced {T_tok} tokens covering "
                        f"{expected_coverage:.4f}s, but tokenizer_window_length_seconds="
                        f"{self.tokenizer_window_length_seconds}s."
                    )
                    spec_default = getattr(spec, "window_length_seconds", None)
                    if spec_default is not None and abs(self.tokenizer_window_length_seconds - spec_default) > 1e-6:
                        raise ValueError(msg + " User-specified window does not match tokenizer output.")
                    else:
                        logger.warning(msg + " Proceeding with default window.")
            self._windowed_validated = True
            logger.info(
                f"encode_windowed: {W} windows of {M} bins "
                f"({self.tokenizer_window_length_seconds}s), {T_tok} token(s)/window"
            )

        S = target_flat.shape[1]
        D = target_flat.shape[3]

        # Reshape: (B*W, S, T_tok, D) -> (B, W, S, T_tok, D) -> (B, W*T_tok, S, D)
        target = target_flat.view(B, W, S, T_tok, D)
        target = target.permute(0, 1, 3, 2, 4).reshape(B, W * T_tok, S, D)

        return {
            "target": target,
            "n_windows": W,
            "tokens_per_window": T_tok,
            "n_spatial": S,
            "n_temporal": W * T_tok,
        }

    def _encode_compress_t(
        self,
        responses: torch.Tensor,
        session_key: str,
        raw_discrete: bool = False,
        skip_target_scaling: bool = False,
    ) -> dict[str, Any]:
        """Encode with a compress_t tokenizer (per-neuron, temporal compression).

        NTT encode() returns 4D ``[B, T_latent, U, D]``. We chunk neurons,
        encode each chunk, strip padding, and concatenate along the U dim.
        Result: ``(B, S=N, T=t_comp, D)`` in the unified format.
        """
        from ntt.core.components.types import BinnedSpikesInput

        B, T, N = responses.shape
        window_bins = self._tokenizer_window_bins()
        if window_bins is not None and window_bins < T:
            if T % window_bins != 0:
                raise ValueError(
                    f"Response length T={T} is not divisible by tokenizer window "
                    f"{window_bins} bins (tokenizer_window_length_seconds="
                    f"{self.tokenizer_window_length_seconds})."
                )
            n_windows = T // window_bins
            windowed = responses.reshape(B, n_windows, window_bins, N).reshape(B * n_windows, window_bins, N)
            flat = self._encode_compress_t(
                windowed,
                session_key,
                raw_discrete=raw_discrete,
                skip_target_scaling=skip_target_scaling,
            )
            target_flat = flat["target"]  # (B*W, N, T_tok, D)
            tokens_per_window = int(target_flat.shape[2])
            latent_dim = int(target_flat.shape[3])
            target = target_flat.reshape(B, n_windows, N, tokens_per_window, latent_dim)
            target = target.permute(0, 2, 1, 3, 4).reshape(B, N, n_windows * tokens_per_window, latent_dim)
            self._compress_t_session_meta[session_key] = {
                "t_comp": n_windows * tokens_per_window,
                "target_t": T,
                "n_total": N,
                "chunk_width": int(self.neuron_batch_size),
                "latent_dim": latent_dim,
                "window_bins": int(window_bins),
                "n_windows": int(n_windows),
                "tokens_per_window": int(tokens_per_window),
            }
            return {
                "target": target,
                "n_spatial": N,
                "n_temporal": n_windows * tokens_per_window,
                "aux_loss": flat.get("aux_loss"),
            }

        nbs = self.neuron_batch_size
        self._set_window_length_seconds_for_current_batch(T)
        latent_chunks = []

        # When the tokenizer's encode() returns float quantized embeddings but
        # we asked for raw_discrete=True (e.g. wanneural-VQ), we have to fetch
        # codebook indices via a separate path. The trailing axis becomes Q
        # (residual-VQ codebook count) instead of z_dim. The output is int64.
        use_indices_path = raw_discrete and self.indices_source != "encode"

        for start in range(0, N, nbs):
            end = min(start + nbs, N)
            chunk = responses[:, :, start:end]  # (B, T, chunk_n)
            chunk_n = chunk.shape[2]

            # Pad last chunk to neuron_batch_size
            if chunk_n < nbs:
                pad = torch.zeros(B, T, nbs - chunk_n, device=chunk.device, dtype=chunk.dtype)
                chunk = torch.cat([chunk, pad], dim=2)

            pad_mask, unit_ids, unit_ids_pad_mask, duration_sec = self._get_boilerplate_tensors(
                B,
                T,
                nbs,
                chunk.device,
            )
            inp = BinnedSpikesInput(
                spike_matrix=chunk,
                pad_and_gap_mask=pad_mask,
                sampling_rate=float(self.sampling_rate),
                unit_ids=unit_ids,
                unit_ids_pad_mask=unit_ids_pad_mask,
                duration_sec=duration_sec,
                session_id=session_key,
                subject_id=self.subject_id,
                species=self.species,
                absolute_start=torch.Tensor([0]),
                absolute_end=torch.Tensor([T]),
            )

            if use_indices_path:
                # (B, T_latent, nbs, Q) int64
                z = self._extract_chunk_indices(inp)
            else:
                z = self.tokenizer.encode(inp)  # [B, T_latent, nbs, D]

            if z.ndim != 4:
                raise ValueError(
                    f"Expected 4D encode output [B, T, U, D], got shape {tuple(z.shape)}. "
                    f"Ensure your NTT tokenizer follows the unified latent shape convention."
                )

            # Strip padding neurons
            latent_chunks.append(z[:, :, :chunk_n, :])

        # Concatenate along neuron (U) dim → [B, T_latent, N_total, D]
        latents = torch.cat(latent_chunks, dim=2)
        t_comp = latents.shape[1]
        latent_dim = latents.shape[3]

        self._compress_t_session_meta[session_key] = {
            "t_comp": t_comp,
            "target_t": T,
            "n_total": N,
            "chunk_width": nbs,
            "latent_dim": latent_dim,
        }

        # NTT gives [B, T_latent, N, D] → unified (B, S=N, T=t_comp, D)
        tokens = latents.permute(0, 2, 1, 3)  # (B, N, t_comp, D)

        if raw_discrete:
            # tokens are int64 here when use_indices_path=True (or already int
            # when source="encode" / transformer-vq). Hand back as-is — _tokens_to_target
            # would corrupt them for residual-VQ (D=Q != 1).
            target = tokens
        elif skip_target_scaling:
            target = tokens.to(torch.float32)
        else:
            target = self._tokens_to_target(tokens)
        return {"target": target, "n_spatial": N, "n_temporal": t_comp, "aux_loss": None}

    def _encode_compress_n(
        self,
        responses: torch.Tensor,
        session_key: str,
        raw_discrete: bool = False,
        skip_target_scaling: bool = False,
    ) -> dict[str, Any]:
        """Encode with a compress_n tokenizer (population compression, time preserved).

        NTT encode() returns 4D ``[B, T, U=1, D]``. We squeeze U, stack chunks.
        Result: ``(B, S=num_chunks, T, D=latent_n)`` in the unified format.
        """
        from ntt.core.components.types import BinnedSpikesInput

        B, T, N = responses.shape
        plan = self.get_or_create_plan(session_key, N)
        latent_chunks = []

        # See _encode_compress_t: same dispatch — fetch indices via the
        # wanneural-style bottleneck path when encode() returns floats but
        # raw_discrete was requested.
        use_indices_path = raw_discrete and self.indices_source != "encode"

        for start, end in plan.chunk_neuron_slices:
            chunk = responses[:, :, start:end]  # (B, T, n_real)
            n_real = end - start

            # Pad neuron dim to tokenizer capacity
            if n_real < plan.pad_size:
                pad = torch.zeros(B, T, plan.pad_size - n_real, device=chunk.device, dtype=chunk.dtype)
                chunk = torch.cat([chunk, pad], dim=2)

            # Build unit_ids: real neuron indices + sentinel -1 for padding
            real_ids = torch.arange(start, end, device=chunk.device, dtype=torch.long)
            if n_real < plan.pad_size:
                pad_ids = torch.full((plan.pad_size - n_real,), -1, device=chunk.device, dtype=torch.long)
                unit_ids = torch.cat([real_ids, pad_ids])
            else:
                unit_ids = real_ids
            unit_ids = unit_ids.unsqueeze(0).expand(B, -1)

            # unit_ids_pad_mask: True for padded positions
            uid_pad_mask = torch.zeros(B, plan.pad_size, device=chunk.device, dtype=torch.bool)
            if n_real < plan.pad_size:
                uid_pad_mask[:, n_real:] = True

            # pad_mask: True = valid (NTT convention)
            pad_mask = torch.ones(B, T, plan.pad_size, device=chunk.device, dtype=torch.bool)
            if n_real < plan.pad_size:
                pad_mask[:, :, n_real:] = False
            duration_sec = torch.full(
                (B,), float(T) / float(self.sampling_rate), device=chunk.device, dtype=torch.float32
            )

            inp = BinnedSpikesInput(
                spike_matrix=chunk,
                pad_and_gap_mask=pad_mask,
                sampling_rate=float(self.sampling_rate),
                unit_ids=unit_ids,
                unit_ids_pad_mask=uid_pad_mask,
                duration_sec=duration_sec,
                session_id=session_key,
                subject_id=self.subject_id,
                species=self.species,
                absolute_start=torch.Tensor([0]),
                absolute_end=torch.Tensor([T]),
            )

            if use_indices_path:
                # _extract_chunk_indices returns (B, T_comp, N_pad, Q) int64.
                # For compress_n, T_comp == T (time preserved) and N_pad == 1
                # (population compressed to a single "neuron" slot).
                z = self._extract_chunk_indices(inp)
            else:
                z = self.tokenizer.encode(inp)  # [B, T, U=1, D]

            if z.ndim != 4:
                raise ValueError(
                    f"Expected 4D encode output [B, T, U, D], got shape {tuple(z.shape)}. "
                    f"Ensure your NTT tokenizer follows the unified latent shape convention."
                )

            # For compress_n: U is 1 (compressed). Squeeze U → (B, T, D)
            latent_chunks.append(z.squeeze(2))

        # Stack chunks → (B, num_chunks, T, D) — already in unified format
        tokens = torch.stack(latent_chunks, dim=1)
        if raw_discrete:
            target = tokens
        elif skip_target_scaling:
            target = tokens.to(torch.float32)
        else:
            target = self._tokens_to_target(tokens)
        return {"target": target, "n_spatial": plan.num_chunks, "n_temporal": T, "aux_loss": None}

    # -- Wavelet paths (compress_n + compress_t) -------------------------------

    @property
    def cosmos_window_bins(self) -> int:
        """Number of input bins per tokenizer window for wavelet (e.g. 1000).

        Reads ``window_length_seconds`` × ``1000 / bin_size_ms`` from the spec.
        """
        spec = self.tokenizer.spec
        wls = float(spec.window_length_seconds)
        bms = float(spec.bin_size_ms)
        return int(round(wls * 1000.0 / bms))

    @property
    def cosmos_n_chunk(self) -> int:
        """Per-chunk neuron capacity for wavelet (= ``spec.n_neurons``)."""
        return int(self.tokenizer.spec.n_neurons)

    def _cosmos_chunk_slices(self, n_total: int) -> list[tuple[int, int]]:
        """Disjoint ``[start, end)`` neuron slices of width ``cosmos_n_chunk``.

        Last chunk is padded internally to ``cosmos_n_chunk`` if it falls
        short. We don't reuse ``CompressNSessionPlan`` because that struct
        requires ``spec.latent_dim`` which wavelet's spec doesn't carry —
        wavelet's per-chunk latent dim is ``z_channels`` (continuous) or
        ``num_codebooks`` (discrete), depending on the bottleneck.
        """
        n_chunk = self.cosmos_n_chunk
        slices: list[tuple[int, int]] = []
        cursor = 0
        while cursor < n_total:
            end = min(cursor + n_chunk, n_total)
            slices.append((cursor, end))
            cursor = end
        return slices

    def _cosmos_make_input(self, spikes_btn: torch.Tensor, session_key: str) -> BinnedSpikesInput:
        from ntt.core.components.types import BinnedSpikesInput

        B, T, N = spikes_btn.shape
        device = spikes_btn.device
        return BinnedSpikesInput(
            spike_matrix=spikes_btn,
            pad_and_gap_mask=torch.ones_like(spikes_btn, dtype=torch.bool),
            sampling_rate=float(self.sampling_rate),
            unit_ids=torch.arange(N, device=device).unsqueeze(0).expand(B, -1).contiguous(),
            unit_ids_pad_mask=torch.ones((B, N), device=device, dtype=torch.bool),
            duration_sec=torch.full((B,), float(T) / float(self.sampling_rate), device=device, dtype=torch.float32),
            session_id=(session_key,) * B,
            subject_id=(self.subject_id,) * B,
            species=(self.species,) * B,
            absolute_start=torch.zeros(B, device=device),
            absolute_end=torch.full((B,), float(T), device=device),
        )

    def _ensure_cosmos_zero_pad_bank(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Lazily encode an all-zeros 1s window through the tokenizer once,
        cache the latent tail used to expand sliced predictions back to the
        full latent window. Session-independent (T4 verified). Returns
        latents shape ``(1, T_lat_full, U_per_chunk, D)``.
        """
        if self._cosmos_zero_pad_latents is not None and self._cosmos_zero_pad_latents.device == device:
            return self._cosmos_zero_pad_latents
        T_full = self.cosmos_window_bins
        n_chunk = self.cosmos_n_chunk
        zeros = torch.zeros(1, T_full, n_chunk, device=device, dtype=dtype)
        with torch.no_grad():
            lat = self.tokenizer.encode(self._cosmos_make_input(zeros, session_key="_cosmos_bg_"))
        bank: torch.Tensor = lat.detach()
        self._cosmos_zero_pad_latents = bank
        return bank

    def _encode_cosmos(
        self,
        responses: torch.Tensor,
        session_key: str,
        raw_discrete: bool = False,
        skip_target_scaling: bool = False,
    ) -> dict[str, Any]:
        """Encode (B, T_resp, N) → unified (B, U_total, T_lat_slice, D).

        Pipeline:
          1. Zero-pad responses on the time dim from T_resp to ``cosmos_window_bins``.
          2. Split N into disjoint ``n_chunk``-neuron groups (last padded if needed).
          3. Per chunk: tokenizer.encode → (B, T_lat_full, U_per_chunk, D)
             (or discrete indices if raw_discrete).
          4. Concat chunks on the U axis → (B, T_lat_full, U_total, D).
          5. If ``cosmos_target_slice`` is set, slice T to [0..K).
          6. Permute to unified-S-major (B, U_total, T_slice, D).

        N_total can exceed ``n_chunk`` (wavelet is N-rigid only per chunk).
        """
        if responses.ndim != 3:
            raise ValueError(f"Expected (B, T, N), got {tuple(responses.shape)}")

        B, T_resp, N = responses.shape
        T_full = self.cosmos_window_bins
        n_chunk = self.cosmos_n_chunk
        device = responses.device
        dtype = responses.dtype

        # 1. Pad time → T_full.
        if T_resp > T_full:
            padded = responses[:, :T_full, :]
        elif T_resp < T_full:
            pad = torch.zeros(B, T_full - T_resp, N, device=device, dtype=dtype)
            padded = torch.cat([responses, pad], dim=1)
        else:
            padded = responses

        # 2. Chunk N into 32-groups. Cache chunk slices per session so the
        #    decoder can reverse the chunking with the same boundaries.
        chunk_slices = self._cosmos_chunk_slices(N)
        # Stash on a wavelet-specific dict so we don't shadow compress_n plans.
        if not hasattr(self, "_cosmos_chunk_cache"):
            self._cosmos_chunk_cache = {}
        cached = self._cosmos_chunk_cache.get(session_key)
        if cached is not None and cached != (N, chunk_slices):
            raise ValueError(
                f"wavelet chunk slices for session={session_key!r} changed: old={cached}, new=({N}, {chunk_slices})"
            )
        self._cosmos_chunk_cache[session_key] = (N, chunk_slices)

        use_indices_path = raw_discrete and self.indices_source != "encode"
        chunks_out: list[torch.Tensor] = []
        for start, end in chunk_slices:
            n_real = end - start
            chunk = padded[:, :, start:end]
            if n_real < n_chunk:
                pad = torch.zeros(B, T_full, n_chunk - n_real, device=device, dtype=dtype)
                chunk = torch.cat([chunk, pad], dim=2)

            inp = self._cosmos_make_input(chunk, session_key=session_key)
            if use_indices_path:
                # Discrete indices via the tokenizer's forward path.
                # Wavelet's vqpt/fsq bottlenecks expose tokens via tokenizer(inp).tokens
                # in shape (B, T_lat, U_lat, Q) int — same as our unified layout.
                with torch.no_grad():
                    out = self.tokenizer(inp)
                if out.tokens is None:
                    raise RuntimeError("wavelet forward did not emit discrete tokens (continuous bottleneck?)")
                z = out.tokens.to(torch.long)  # (B, T_lat_full, U_per_chunk, Q)
            else:
                z = self.tokenizer.encode(inp)  # (B, T_lat_full, U_per_chunk, D)

            if z.ndim != 4:
                raise ValueError(f"Expected 4D encode output, got {tuple(z.shape)} from wavelet chunk.")
            chunks_out.append(z)

        # 4. Concat chunks on the U axis.
        # Each chunk: (B, T_lat_full, U_per_chunk, D)
        # Concat:     (B, T_lat_full, U_total, D)
        z_full = torch.cat(chunks_out, dim=2)

        # 5. Slice on T to [0..K) if cosmos_target_slice is set.
        K = self.cosmos_target_slice
        if K is not None:
            T_lat_full = z_full.shape[1]
            if K > T_lat_full:
                raise ValueError(f"cosmos_target_slice={K} > T_lat_full={T_lat_full}")
            z_sliced = z_full[:, :K, :, :]
        else:
            z_sliced = z_full

        # 6. Permute to unified S-major: (B, U_total, T_slice, D).
        tokens = z_sliced.permute(0, 2, 1, 3).contiguous()

        # Discrete tokens: keep as-is; continuous tokens: optionally scale.
        if raw_discrete:
            target = tokens
        elif skip_target_scaling:
            target = tokens.to(torch.float32)
        else:
            target = self._tokens_to_target(tokens)

        n_spatial = int(target.shape[1])
        n_temporal = int(target.shape[2])
        return {"target": target, "n_spatial": n_spatial, "n_temporal": n_temporal, "aux_loss": None}

    def _decode_cosmos(self, tokens: torch.Tensor, session_key: str) -> torch.Tensor:
        """Decode unified (B, U_total, T_slice, D) → (B, T_full, N_total) rates.

        Steps:
          1. Permute back to T-major: (B, T_slice, U_total, D).
          2. If T_slice < T_lat_full, expand by filling [T_slice..T_lat_full)
             from the cached zero-pad bank (broadcast across U slots and B).
          3. Split chunks on U axis into ``num_chunks`` groups of
             ``U_per_chunk`` slots.
          4. Per chunk: tokenizer.reconstruct(...).reconstruction → (B, T_full, n_chunk).
          5. Concat on N axis, trim to true N_total from the chunking plan.

        The caller is responsible for slicing the output to the
        decoder-safe horizon (``cosmos_t_safe``) before computing metrics —
        that's an encoding-module-level concern.
        """
        return self._decode_cosmos_impl(tokens, session_key=session_key, differentiable=False)

    def _decode_differentiable_cosmos(self, tokens: torch.Tensor, session_key: str) -> torch.Tensor:
        """Differentiable decode for wavelet. Uses ``tokenizer.decode`` (already softplus-rates)."""
        return self._decode_cosmos_impl(tokens, session_key=session_key, differentiable=True)

    def _decode_cosmos_impl(
        self,
        tokens: torch.Tensor,
        session_key: str,
        differentiable: bool,
    ) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"Expected (B, S, T, D), got {tuple(tokens.shape)}")

        cache = getattr(self, "_cosmos_chunk_cache", {})
        cached = cache.get(session_key)
        if cached is None:
            raise RuntimeError(
                f"wavelet decode: no chunking plan for session={session_key!r}. "
                f"call encode(...) first to populate the plan."
            )
        N_total, chunk_slices = cached
        num_chunks_plan = len(chunk_slices)

        # 1. Permute to T-major.
        # tokens shape: (B, U_total, T_slice, D) → (B, T_slice, U_total, D)
        tk_t = tokens.permute(0, 2, 1, 3).contiguous()
        B, T_slice, U_total, D = tk_t.shape

        # Need to know U_per_chunk to split the U axis. Probe via the zero-pad bank.
        bg = self._ensure_cosmos_zero_pad_bank(device=tk_t.device, dtype=tk_t.dtype)
        # bg shape: (1, T_lat_full, U_per_chunk, D)
        T_lat_full = int(bg.shape[1])
        U_per_chunk = int(bg.shape[2])
        if U_total % U_per_chunk != 0:
            raise ValueError(f"U_total={U_total} not divisible by U_per_chunk={U_per_chunk}")
        num_chunks_in = U_total // U_per_chunk
        if num_chunks_in != num_chunks_plan:
            raise ValueError(f"wavelet decode: latent has {num_chunks_in} chunks but plan expects {num_chunks_plan}")

        # 2. Expand T from T_slice to T_lat_full if needed.
        if T_slice < T_lat_full:
            # bg is the zero-pad encoding for ONE 32-neuron chunk. Replicate
            # along the U axis (once per chunk) and along the batch axis.
            bg_tail = bg[:, T_slice:, :, :]  # (1, T_lat_full - T_slice, U_per_chunk, D)
            bg_tail_tiled = bg_tail.repeat(B, 1, num_chunks_in, 1)
            tk_full = torch.cat([tk_t, bg_tail_tiled], dim=1)
        elif T_slice == T_lat_full:
            tk_full = tk_t
        else:
            raise ValueError(f"T_slice={T_slice} > T_lat_full={T_lat_full}")

        # 3. Split chunks on U.
        # tk_full shape: (B, T_lat_full, U_total, D)
        T_full_input = self.cosmos_window_bins
        chunk_recons: list[torch.Tensor] = []
        for c, (start, end) in enumerate(chunk_slices):
            u0, u1 = c * U_per_chunk, (c + 1) * U_per_chunk
            chunk_lat = tk_full[:, :, u0:u1, :]  # (B, T_lat_full, U_per_chunk, D)
            # Tokenizer.decode expects the same shape it emits from encode.
            if differentiable:
                rec = self.tokenizer.decode(chunk_lat)  # (B, T_full_input, n_chunk) softplus-rates
            else:
                # Use tokenizer.reconstruct: same decode path but may include
                # Poisson sampling depending on tokenizer.training mode.
                rec = self.tokenizer.reconstruct(chunk_lat).reconstruction.float()
            # Trim padded neurons.
            n_real = end - start
            chunk_recons.append(rec[:, :, :n_real])

        # 4. Concat on N and return.
        recon = torch.cat(chunk_recons, dim=2)  # (B, T_full_input, N_total)
        if recon.shape[1] != T_full_input:
            raise RuntimeError(f"wavelet decode produced T={recon.shape[1]} but expected T_full={T_full_input}")
        return recon

    # -- Decode paths ---------------------------------------------------------

    def decode(self, tokens: torch.Tensor, session_key: str) -> torch.Tensor:
        """Decode raw latent tokens back to ``(B, T_recon, N)`` neural responses.

        Uses ``tokenizer.reconstruct()`` which may include Poisson sampling.

        Args:
            tokens: ``(B, S, T, D)`` raw latent tensor in the unified format.
                Should be in raw tokenizer latent space (call ``_target_to_tokens``
                first if tokens are in scaled target space).
            session_key: Session identifier (used to look up chunking plan).
        """
        self._align_tokenizer_device(tokens.device)
        if self.is_perceiver:
            return self._decode_perceiver(tokens, session_key, differentiable=False)
        if self.is_cosmos:
            return self._decode_cosmos(tokens, session_key)
        if self.is_compress_n:
            return self._decode_compress_n(tokens, session_key)
        return self._decode_compress_t(tokens, session_key=session_key)

    def decode_differentiable(self, tokens: torch.Tensor, session_key: str) -> torch.Tensor:
        """Decode latents to firing rates using the differentiable decoder path.

        Unlike ``decode()`` which uses ``tokenizer.reconstruct()`` (includes Poisson
        sampling), this calls ``tokenizer.decode()`` which returns differentiable
        softplus rates.

        Args:
            tokens: ``(B, S, T, D)`` raw latent tensor in the unified format.
            session_key: Session identifier.

        Returns:
            Differentiable firing rates ``(B, T, N)``.
        """
        self._align_tokenizer_device(tokens.device)
        if self.is_perceiver:
            return self._decode_perceiver(tokens, session_key, differentiable=True)
        if self.is_cosmos:
            return self._decode_differentiable_cosmos(tokens, session_key)
        if self.is_compress_n:
            return self._decode_differentiable_compress_n(tokens, session_key)
        return self._decode_differentiable_compress_t(tokens, session_key=session_key)

    def _decode_compress_t(self, tokens: torch.Tensor, session_key: str | None = None) -> torch.Tensor:
        """Decode with a compress_t tokenizer via ``reconstruct()``.

        Accepts unified ``(B, S=N, T=t_comp, D)`` tokens. Converts to NTT's
        ``[B, T_latent, nbs, D]`` chunks and calls ``reconstruct()``.
        """
        meta = self._compress_t_session_meta.get(session_key) if session_key is not None else None
        target_t = int(meta["target_t"]) if meta is not None else None

        B, n_total, t_comp, latent_dim = tokens.shape
        nbs = int(meta["chunk_width"]) if meta is not None and "chunk_width" in meta else self.neuron_batch_size

        if meta is not None and int(meta.get("n_windows", 1)) > 1:
            n_windows = int(meta["n_windows"])
            tokens_per_window = int(meta["tokens_per_window"])
            window_bins = int(meta["window_bins"])
            if t_comp != n_windows * tokens_per_window:
                raise ValueError(
                    f"Token temporal dim {t_comp} does not match windowed tokenizer metadata "
                    f"{n_windows} windows * {tokens_per_window} tokens/window."
                )
            flat = (
                tokens.reshape(B, n_total, n_windows, tokens_per_window, latent_dim)
                .permute(0, 2, 1, 3, 4)
                .reshape(B * n_windows, n_total, tokens_per_window, latent_dim)
            )
            decoded = self._decode_compress_t_single(
                flat,
                target_t=window_bins,
                nbs=nbs,
                differentiable=False,
            )
            return decoded.reshape(B, n_windows, window_bins, n_total).reshape(B, n_windows * window_bins, n_total)

        if target_t is not None:
            self._set_window_length_seconds_for_current_batch(target_t)

        return self._decode_compress_t_single(tokens, target_t=target_t, nbs=nbs, differentiable=False)

    def _decode_compress_t_single(
        self,
        tokens: torch.Tensor,
        *,
        target_t: int | None,
        nbs: int,
        differentiable: bool,
    ) -> torch.Tensor:
        """Decode one fixed tokenizer window for a compress_t tokenizer.

        Accepts either continuous-latent ``(B, N, t_comp, D=z_dim)`` floats or
        discrete-index ``(B, N, t_comp, D=Q)`` int64. Integer inputs are routed
        through ``bottleneck.indices_to_codes`` (differentiable) or the
        tokenizer's own ``reconstruct`` (non-differentiable, auto-detects int).
        """
        if target_t is not None:
            self._set_window_length_seconds_for_current_batch(target_t)

        B, n_total, t_comp, latent_dim = tokens.shape
        is_int = not tokens.is_floating_point()
        # (B, N, t_comp, D) -> NTT format (B, t_comp, N, D)
        tokens_4d = tokens.permute(0, 2, 1, 3).contiguous()

        decoded_chunks = []
        for start in range(0, n_total, nbs):
            end = min(start + nbs, n_total)
            chunk_n = end - start
            chunk = tokens_4d[:, :, start:end, :]  # (B, T, chunk_n, D)

            if chunk_n < nbs:
                pad = torch.zeros(B, t_comp, nbs - chunk_n, latent_dim, device=chunk.device, dtype=chunk.dtype)
                chunk = torch.cat([chunk, pad], dim=2)

            if is_int:
                decoded = self._decode_int_chunk(chunk, t_comp=t_comp, n_pad=nbs, differentiable=differentiable)
            elif differentiable:
                decoded = self.tokenizer.decode(chunk)  # (B, T_recon, nbs)
            else:
                decoded = self.tokenizer.reconstruct(chunk).reconstruction  # (B, T_recon, nbs)
            decoded_chunks.append(decoded[:, :, :chunk_n])

        return torch.cat(decoded_chunks, dim=2).to(torch.float32)  # (B, T, N)

    def _decode_int_chunk(
        self,
        chunk: torch.Tensor,
        *,
        t_comp: int,
        n_pad: int,
        differentiable: bool,
    ) -> torch.Tensor:
        """Decode a single int-dtype chunk ``(B, T, N_pad, Q)`` to rates ``(B, T_recon, N_pad)``.

        Two decode-from-indices conventions are supported:

          - **wanneural-style** (``tokenizer.bottleneck.indices_to_codes`` exists):
            flatten the (T, N) axes and pass ``(B, T*N[, Q])`` indices to the
            bottleneck. Returns ``(B, C, T, N_pad)`` codes which we feed to the
            decoder. Non-differentiable variant uses ``tokenizer.reconstruct``
            on the same flat layout (auto-routes int → indices_to_codes).
          - **transformer-vq-style** (no ``bottleneck`` attr; ``tokenizer.decode`` takes
            4D int tokens directly): pass the 4D ``(B, T, N_pad, 1)`` chunk as-is.
        """
        B, T, N_pad, Q = chunk.shape
        if T != t_comp or N_pad != n_pad:
            raise ValueError(
                f"_decode_int_chunk shape mismatch: got (T={T}, N_pad={N_pad}), "
                f"expected (t_comp={t_comp}, n_pad={n_pad})"
            )

        bn = getattr(self.tokenizer, "bottleneck", None)
        bn_supports_indices = bn is not None and hasattr(bn, "indices_to_codes")

        if bn_supports_indices:
            # Flatten to (B, T*N) or (B, T*N, Q) per VQPTBottleneck.indices_to_codes
            # API. VQPTBottleneck._to_seq uses "b c t n -> b (t n) c" — same
            # (t major, n minor) ordering we used for unflattening.
            if Q == 1:
                flat = chunk[..., 0].reshape(B, T * N_pad)
            else:
                flat = chunk.reshape(B, T * N_pad, Q)

            # vqpt grouped residual quantizers expect their group axis restored
            # as (G, B, T*N, Q_per_group). During CE training/eval we pack that
            # axis into the unified trailing Q dimension.
            quantizer = getattr(bn, "quantizer", None)
            groups = int(getattr(quantizer, "groups", 1) or 1)
            if groups > 1:
                if Q % groups != 0:
                    raise ValueError(
                        f"grouped tokenizer decode expected Q divisible by groups={groups}, got Q={Q}. "
                        "Check tokenizer.num_codebooks / model.num_codebooks metadata."
                    )
                q_per_group = Q // groups
                flat = flat.reshape(B, T * N_pad, groups, q_per_group)
                flat = flat.permute(2, 0, 1, 3).contiguous()

            if differentiable:
                assert bn is not None  # implied by bn_supports_indices
                z = bn.indices_to_codes(flat, T=t_comp)  # (B, C, T, N_pad)
                # tokenizer.decode wants (B, T, N, C); see WanNeuralTokenizer.decode.
                from einops import rearrange

                z_4d = rearrange(z, "b c t n -> b t n c")
                return self.tokenizer.decode(z_4d)  # (B, T_recon, N_pad)
            # Non-differentiable: tokenizer.reconstruct auto-routes int → indices_to_codes.
            return self.tokenizer.reconstruct(flat).reconstruction  # (B, T_recon, N_pad)

        # transformer-vq-style: tokenizer.decode / tokenizer.reconstruct accept 4D int
        # tokens directly and handle the codebook lookup internally.
        if differentiable:
            return self.tokenizer.decode(chunk)
        return self.tokenizer.reconstruct(chunk).reconstruction

    def _decode_compress_n(self, tokens: torch.Tensor, session_key: str) -> torch.Tensor:
        """Decode with a compress_n tokenizer via ``reconstruct()``.

        Accepts unified ``(B, S=num_chunks, T, D=latent_n)`` tokens. Supports
        both float (continuous latents) and int64 (codebook indices, e.g.
        compress_n VQ); the latter is routed through ``_decode_int_chunk``.
        """
        if session_key not in self._compress_n_plans:
            raise ValueError(
                f"No chunking plan found for session '{session_key}'. "
                f"encode() must be called before decode(), or register the plan via get_or_create_plan()."
            )

        plan = self._compress_n_plans[session_key]
        is_int = not tokens.is_floating_point()
        decoded_chunks = []

        for i, (start, end) in enumerate(plan.chunk_neuron_slices):
            n_real = end - start
            chunk = tokens[:, i, :, :]  # (B, T, D)
            chunk_4d = chunk.unsqueeze(2)  # (B, T, 1, D) — NTT 4D format
            if is_int:
                # compress_n: U=1 spatial slot, T preserved → t_comp=T, n_pad=1.
                reconstructed = self._decode_int_chunk(
                    chunk_4d, t_comp=int(chunk.shape[1]), n_pad=1, differentiable=False
                )
            else:
                reconstructed = self.tokenizer.reconstruct(chunk_4d).reconstruction  # (B, T, pad_size)
            decoded_chunks.append(reconstructed[:, :, :n_real])

        return torch.cat(decoded_chunks, dim=2).to(torch.float32)  # (B, T, N)

    def _decode_differentiable_compress_t(self, tokens: torch.Tensor, session_key: str | None = None) -> torch.Tensor:
        """Differentiable decode for compress_t: ``tokenizer.decode()`` (no Poisson).

        Accepts unified ``(B, S=N, T=t_comp, D)`` tokens.
        """
        meta = self._compress_t_session_meta.get(session_key) if session_key is not None else None
        B, n_total, t_comp, latent_dim = tokens.shape
        nbs = int(meta["chunk_width"]) if meta is not None and "chunk_width" in meta else self.neuron_batch_size

        if meta is not None and int(meta.get("n_windows", 1)) > 1:
            n_windows = int(meta["n_windows"])
            tokens_per_window = int(meta["tokens_per_window"])
            window_bins = int(meta["window_bins"])
            if t_comp != n_windows * tokens_per_window:
                raise ValueError(
                    f"Token temporal dim {t_comp} does not match windowed tokenizer metadata "
                    f"{n_windows} windows * {tokens_per_window} tokens/window."
                )
            flat = (
                tokens.reshape(B, n_total, n_windows, tokens_per_window, latent_dim)
                .permute(0, 2, 1, 3, 4)
                .reshape(B * n_windows, n_total, tokens_per_window, latent_dim)
            )
            decoded = self._decode_compress_t_single(
                flat,
                target_t=window_bins,
                nbs=nbs,
                differentiable=True,
            )
            return decoded.reshape(B, n_windows, window_bins, n_total).reshape(B, n_windows * window_bins, n_total)

        target_t = int(meta["target_t"]) if meta is not None and "target_t" in meta else None
        return self._decode_compress_t_single(tokens, target_t=target_t, nbs=nbs, differentiable=True)

    def _decode_differentiable_compress_n(self, tokens: torch.Tensor, session_key: str) -> torch.Tensor:
        """Differentiable decode for compress_n: ``tokenizer.decode()`` (no Poisson).

        Accepts unified ``(B, S=num_chunks, T, D=latent_n)`` tokens. Int-dtype
        inputs go through ``_decode_int_chunk`` which converts indices to codes
        via ``bottleneck.indices_to_codes`` before the decoder.
        """
        plan = self._compress_n_plans[session_key]
        is_int = not tokens.is_floating_point()
        decoded_chunks = []
        for i, (start, end) in enumerate(plan.chunk_neuron_slices):
            n_real = end - start
            chunk = tokens[:, i, :, :]  # (B, T, D)
            chunk_4d = chunk.unsqueeze(2)  # (B, T, 1, D)
            if is_int:
                rates = self._decode_int_chunk(chunk_4d, t_comp=int(chunk.shape[1]), n_pad=1, differentiable=True)
            else:
                rates = self.tokenizer.decode(chunk_4d)  # (B, T, pad_size)
            decoded_chunks.append(rates[:, :, :n_real])
        return torch.cat(decoded_chunks, dim=2)  # (B, T, N)

    # -- Perceiver-style paths (compress_n + variable_n) ----------------------

    def _rebin_perceiver_counts(self, responses: torch.Tensor) -> torch.Tensor:
        """Sum-pool responses along the time axis by ``perceiver_rebin_factor``.

        ``responses`` is ``(B, T_data, N)``. Returns ``(B, T_data // f, N)``.
        No-op when the factor is 1.
        """
        f = int(self.perceiver_rebin_factor)
        if f <= 1:
            return responses
        B, T, N = responses.shape
        T_new = T // f
        if T_new == 0:
            raise ValueError(
                f"perceiver_rebin_factor={f} is larger than the input time dim {T}; "
                f"the response window is shorter than one tokenizer bin."
            )
        return responses[:, : T_new * f, :].reshape(B, T_new, f, N).sum(dim=2)

    def _encode_perceiver(
        self,
        responses: torch.Tensor,
        session_key: str,
        raw_discrete: bool = False,
        skip_target_scaling: bool = False,
    ) -> dict[str, Any]:
        """Encode with a Perceiver-style tokenizer (variable_n + compress_n).

        The tokenizer returns ``(B, M, C, L)`` — M latent tokens, each a C-dim
        feature evolving over L time bins. We rebin the input from the
        downstream sampling rate (e.g. 1 kHz Cowley) to the tokenizer's native
        bin size (``spec.bin_size_ms``), pack the whole population into one
        ``BinnedSpikesInput`` with the NTT mask convention (``1 = real``),
        then reshape the 4D output to the unified ``(B, S, T, D)`` layout.
        """
        from ntt.core.components.types import BinnedSpikesInput

        B, T_in, N = responses.shape
        responses = self._rebin_perceiver_counts(responses)  # (B, T, N) at tokenizer rate
        _, T, _ = responses.shape

        # No need to touch spec.window_length_seconds: the extracted
        # Perceiver is length-agnostic (the spec window only sizes the build).
        bin_size_s = 1.0 / float(self.reconstruction_rate_hz)

        device = responses.device
        unit_ids = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
        # NTT convention for Perceiver: True = real. No padding here since
        # every Cowley session feeds a fixed, fully-observed population.
        unit_ids_pad_mask = torch.ones(B, N, device=device, dtype=torch.bool)
        pad_and_gap_mask = torch.ones(B, T, N, device=device, dtype=torch.bool)
        duration_sec = torch.full((B,), float(T) * bin_size_s, device=device, dtype=torch.float32)

        # Per-batch-element session ids: PerceiverTokenizer._get_embeddings
        # iterates over ``x.session_id`` (one entry per batch element) to look
        # up each sample's NuCLR slice. Passing a scalar string collapses to a
        # 1-tuple → only e[0] gets filled; e[1..B-1] remain zeros and the
        # decoder produces an identical, neuron-agnostic rate for those
        # elements. Always pass a B-tuple.
        session_ids = (session_key,) * B
        subject_ids = (self.subject_id,) * B
        species_ids = (self.species,) * B

        inp = BinnedSpikesInput(
            spike_matrix=responses.to(torch.float32),
            pad_and_gap_mask=pad_and_gap_mask,
            sampling_rate=float(self.reconstruction_rate_hz),
            unit_ids=unit_ids,
            unit_ids_pad_mask=unit_ids_pad_mask,
            duration_sec=duration_sec,
            session_id=session_ids,
            subject_id=subject_ids,
            species=species_ids,
            absolute_start=torch.zeros(B, device=device, dtype=torch.float32),
            absolute_end=torch.full((B,), float(T) * bin_size_s, device=device, dtype=torch.float32),
        )

        z = self.tokenizer.encode(inp)  # (B, M, C, L)
        if z.ndim != 4:
            raise ValueError(f"Expected 4D encode output (B, M, C, L) from Perceiver, got {tuple(z.shape)}")
        B_, M, C, L = z.shape
        assert B_ == B, f"batch size mismatch in perceiver encode: {B_} vs {B}"

        # Ensure any session_key we haven't seen yet gets a plan recorded
        # (so downstream num_chunks lookups are consistent).
        self.get_or_create_plan(session_key, N)

        if self.latent_layout == "m_as_spatial":
            # (B, M, C, L) → (B, S=M, T=L, D=C)
            tokens = z.permute(0, 1, 3, 2).contiguous()
            n_spatial = M
        else:  # "m_flat"
            # (B, M, C, L) → (B, L, M*C) → (B, 1, L, M*C)
            tokens = z.permute(0, 3, 1, 2).reshape(B, L, M * C).unsqueeze(1).contiguous()
            n_spatial = 1

        if raw_discrete:
            target = tokens
        elif skip_target_scaling:
            target = tokens.to(torch.float32)
        else:
            target = self._tokens_to_target(tokens)

        return {
            "target": target,
            "n_spatial": n_spatial,
            "n_temporal": L,
            "aux_loss": None,
        }

    def _decode_perceiver(
        self,
        tokens: torch.Tensor,
        session_key: str,
        differentiable: bool = True,
    ) -> torch.Tensor:
        """Decode perceiver latents back to rates at the tokenizer's native rate.

        ``tokens`` is in the unified ``(B, S, T, D)`` layout produced by
        :meth:`_encode_perceiver`. We reshape back to ``(B, M, C, L)``, look up
        the session's per-neuron NuCLR embeddings ``e`` via
        ``tokenizer._get_embeddings`` (local unit ids ``0..N-1``, as in encode),
        and decode with ``tokenizer.count_wrapper.decode(z, e)`` — the same
        softplus-rate decoder ``reconstruct_from_input`` uses. The extracted
        ``PerceiverTokenizer.decode(tokens)`` can't be used: it takes no
        embeddings and always raises.

        ``differentiable=False`` returns the same rates under ``no_grad``.
        Returns rates in ``(B, L, N)`` — N is the session's neuron count.
        """
        from ntt.core.components.types import BinnedSpikesInput

        if session_key not in self._compress_n_plans:
            raise ValueError(
                f"No chunking plan for session '{session_key}'. "
                f"encode() must be called first, or register the plan via get_or_create_plan()."
            )
        plan = self._compress_n_plans[session_key]
        N = plan.n_neurons_total

        B = tokens.shape[0]
        M = self.perceiver_M
        C = self.perceiver_C

        if self.latent_layout == "m_as_spatial":
            # (B, M, L, C) → (B, M, C, L)
            B_, S, L, D = tokens.shape
            if S != M or D != C:
                raise ValueError(f"Expected tokens (B, S={M}, T, D={C}) for m_as_spatial, got {tuple(tokens.shape)}")
            z = tokens.permute(0, 1, 3, 2).contiguous()
        else:  # m_flat
            # (B, 1, L, M*C) → (B, L, M, C) → (B, M, C, L)
            B_, S, L, D = tokens.shape
            if S != 1 or D != M * C:
                raise ValueError(f"Expected tokens (B, 1, T, {M * C}) for m_flat, got {tuple(tokens.shape)}")
            z = tokens.squeeze(1).reshape(B, L, M, C).permute(0, 2, 3, 1).contiguous()

        # Build a minimal BinnedSpikesInput to resolve the session's embeddings.
        # _get_embeddings only reads session_id + unit_ids; spike_matrix is
        # required by the type, so we pass zeros.
        device = z.device
        bin_size_s = 1.0 / float(self.reconstruction_rate_hz)
        unit_ids = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
        unit_ids_pad_mask = torch.ones(B, N, device=device, dtype=torch.bool)
        pad_and_gap_mask = torch.ones(B, L, N, device=device, dtype=torch.bool)
        spikes_placeholder = torch.zeros(B, L, N, device=device, dtype=torch.float32)
        session_ids = (session_key,) * B
        subject_ids = (self.subject_id,) * B
        species_ids = (self.species,) * B
        inp = BinnedSpikesInput(
            spike_matrix=spikes_placeholder,
            pad_and_gap_mask=pad_and_gap_mask,
            sampling_rate=float(self.reconstruction_rate_hz),
            unit_ids=unit_ids,
            unit_ids_pad_mask=unit_ids_pad_mask,
            duration_sec=torch.full((B,), float(L) * bin_size_s, device=device, dtype=torch.float32),
            session_id=session_ids,
            subject_id=subject_ids,
            species=species_ids,
            absolute_start=torch.zeros(B, device=device, dtype=torch.float32),
            absolute_end=torch.full((B,), float(L) * bin_size_s, device=device, dtype=torch.float32),
        )
        # _get_embeddings returns (e, mask)
        e, _mask = self.tokenizer._get_embeddings(inp)
        e = e.to(z.dtype)

        # count_wrapper.decode returns softplus rates in the tokenizer-native
        # (B, N, L) layout; transpose to the wrapper's (B, L, N).
        with torch.set_grad_enabled(differentiable and torch.is_grad_enabled()):
            rates = self.tokenizer.count_wrapper.decode(z, e)
        return rates.transpose(-1, -2).to(torch.float32)

    # -- Internal helpers -----------------------------------------------------

    def _set_window_length_seconds_for_current_batch(self, num_bins: int) -> None:
        """Align spec.window_length_seconds with the current batch length, if present."""
        spec = getattr(self.tokenizer, "spec", None)
        if spec is None or not hasattr(spec, "window_length_seconds"):
            return
        object.__setattr__(spec, "window_length_seconds", float(num_bins) / float(self.sampling_rate))

    def _tokenizer_window_bins(self) -> int | None:
        """Return configured tokenizer window size in input bins, if available."""
        if self.tokenizer_window_length_seconds is None:
            return None
        bins = int(round(float(self.tokenizer_window_length_seconds) * float(self.sampling_rate)))
        return bins if bins > 0 else None

    def _target_to_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Invert target encoding: undo latent_scale and discrete encoding.

        Works with any shape — scaling broadcasts over all dims except the last (D).

        Note: The discrete branches (one_hot / float_index) below serve BSQ-style
        targets. For cross-entropy with discrete tokenizers (e.g.
        transformer-vq, internal name CALM), use ``raw_discrete=True`` in encode and bypass this method entirely.
        """
        if self.latent_channel_means is not None:
            # Means and stds are always calibrated together.
            assert self.latent_channel_stds is not None
            tokens = tokens * self.latent_channel_stds.to(tokens.device) + self.latent_channel_means.to(tokens.device)
        elif self.latent_scale != 1.0:
            tokens = tokens * self.latent_scale

        spec = getattr(self.tokenizer, "spec", None)
        latent_type = getattr(spec, "latent_type", None) if spec is not None else None
        is_discrete = str(latent_type).lower().endswith("discrete")
        if not is_discrete:
            return tokens

        # Discrete target modes — not used by the cross_entropy loss path
        if self.discrete_target_mode == "one_hot":
            return tokens.argmax(dim=-1).long()
        if self.discrete_target_mode == "float_index":
            if tokens.ndim >= 2 and tokens.shape[-1] == 1:
                return tokens.squeeze(-1).round().long()
            return tokens.round().long()
        raise ValueError(
            f"Unsupported discrete_target_mode='{self.discrete_target_mode}'. Use 'float_index' or 'one_hot'."
        )

    def _tokens_to_target(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert raw tokenizer output to a scaled float target for loss computation.

        Continuous tokens are divided by ``latent_scale`` so that the model
        operates on ~unit-variance targets. Scaling broadcasts over all dims
        except the last (D). Works with any shape including unified ``(B, S, T, D)``.

        Note: The discrete integer branches (one_hot / float_index) below serve
        BSQ-style targets. For cross-entropy with discrete tokenizers
        (e.g. transformer-vq, internal name CALM), use ``raw_discrete=True`` in encode to get raw indices directly.
        """
        if tokens.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            # Discrete target modes — not used by the cross_entropy loss path
            if self.discrete_target_mode == "one_hot":
                vocab_size = self.discrete_target_vocab_size
                if vocab_size is None:
                    vocab_size = int(self.tokenizer.get_vocab_size())
                return torch.nn.functional.one_hot(tokens.long(), num_classes=vocab_size).to(torch.float32)
            if self.discrete_target_mode != "float_index":
                raise ValueError(
                    f"Unsupported discrete_target_mode='{self.discrete_target_mode}'. Use 'float_index' or 'one_hot'."
                )
            return tokens.to(torch.float32).unsqueeze(-1)
        target = tokens.to(torch.float32)
        if self.latent_channel_means is not None:
            # Means and stds are always calibrated together.
            assert self.latent_channel_stds is not None
            target = (target - self.latent_channel_means.to(target.device)) / self.latent_channel_stds.to(target.device)
        elif self.latent_scale != 1.0:
            target = target / self.latent_scale
        return target


# ---------------------------------------------------------------------------
# MLflow artifact helpers
# ---------------------------------------------------------------------------


def _looks_like_mlflow_artifact_uri(path: str) -> bool:
    """Return True if `path` looks like an MLflow artifact URI."""
    if not path:
        return False
    parsed = urlparse(path)
    if parsed.scheme in {"runs", "models", "mlflow-artifacts"}:
        return True
    return path.startswith(("runs:", "models:", "mlflow-artifacts:"))


def _download_mlflow_artifact(artifact_uri: str) -> str:
    """Download an MLflow artifact URI and return a local filesystem path."""
    from mlflow import artifacts

    local_path = artifacts.download_artifacts(artifact_uri=artifact_uri)
    return local_path


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    """Resolve a checkpoint path; MLflow artifact URIs (optional) are downloaded first."""
    if _looks_like_mlflow_artifact_uri(checkpoint_path):
        return _download_mlflow_artifact(checkpoint_path)
    return checkpoint_path


def _configure_tokenizer_mode(tokenizer: torch.nn.Module, mode: str) -> None:
    """Set tokenizer to frozen, trainable, or finetune_decoder mode.

    For ``"finetune_decoder"``, all parameters are frozen and the model is set
    to eval mode (same as ``"frozen"``).  The selective unfreezing of decoder
    parameters happens later via ``TokenizerWrapper.enable_decoder_finetuning()``.
    """
    if mode not in {"frozen", "trainable", "finetune_decoder"}:
        raise ValueError(
            f"Unsupported tokenizer mode: '{mode}'. Expected one of ['frozen', 'trainable', 'finetune_decoder']."
        )
    is_trainable = mode == "trainable"
    for param in tokenizer.parameters():
        param.requires_grad = is_trainable
    if is_trainable:
        tokenizer.train()
    else:
        tokenizer.eval()


# ---------------------------------------------------------------------------
# NTT tokenizer loading
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _allow_abstract_dist_load(fallback_seconds: float):
    """Temporarily allow instantiating ``ntt.core.components.dist.Dist``.

    Some checkpoints store ``float | Dist`` (or ``int | Dist``) spec fields in a
    form that collapses to the base ``Dist`` schema and drops every
    subclass-specific attribute. On unpickle, pydantic then tries to
    construct the abstract ``Dist`` directly and fails.

    We clear ``__abstractmethods__`` so the unpickle can produce a stub ``Dist``
    instance (which ``_repair_stub_dist_fields`` later detects via
    ``type(x) is Dist``), and install concrete ``min`` / ``max`` / ``sample``
    on the base class so any ``@model_validator`` that peeks at those during
    spec reconstruction (e.g. the transformer-vq spec's ``_validate_config`` multiplies
    ``.min`` by ``bin_size_ms`` for a divisibility check) doesn't blow up
    before the repair step can run. ``fallback_seconds`` should be the
    downstream window so the divisibility math sees a consistent value.
    Concrete subclasses (``UniformDist`` etc.) override these, so good
    checkpoints are unaffected.
    """
    from ntt.core.components.dist import Dist

    saved_abstract = Dist.__abstractmethods__
    saved_min = Dist.__dict__.get("min")
    saved_max = Dist.__dict__.get("max")
    saved_sample = Dist.__dict__.get("sample")

    Dist.__abstractmethods__ = frozenset()
    Dist.min = property(lambda self: fallback_seconds)  # pyright: ignore[reportAttributeAccessIssue] # abstract property patched on purpose
    Dist.max = property(lambda self: fallback_seconds)  # pyright: ignore[reportAttributeAccessIssue] # abstract property patched on purpose
    Dist.sample = lambda self: fallback_seconds
    try:
        yield
    finally:
        Dist.__abstractmethods__ = saved_abstract
        if saved_min is not None:
            Dist.min = saved_min  # pyright: ignore[reportAttributeAccessIssue] # abstract property patched on purpose
        if saved_max is not None:
            Dist.max = saved_max  # pyright: ignore[reportAttributeAccessIssue] # abstract property patched on purpose
        if saved_sample is not None:
            Dist.sample = saved_sample


def _repair_stub_dist_fields(
    spec: Any,
    fallback_window_seconds: float,
    fallback_n_neurons: int | None,
) -> Any:
    """Replace any stub ``Dist`` fields on a TokenizerSpec with scalars.

    A stub is identified as ``type(value) is Dist`` — the exact abstract base
    class, which appears when a checkpoint stored the field without its
    concrete subclass data. Concrete subclass instances
    (``UniformDist`` etc.) are left untouched.

    Returns a new spec with the replaced fields, or the original if nothing
    needed repairing.
    """
    from ntt.core.components.dist import Dist

    updates: dict[str, Any] = {}

    wls = getattr(spec, "window_length_seconds", None)
    if type(wls) is Dist:
        updates["window_length_seconds"] = float(fallback_window_seconds)

    n = getattr(spec, "n_neurons", None)
    if type(n) is Dist:
        if fallback_n_neurons is None:
            raise ValueError(
                "Checkpoint tokenizer_spec has a stub Dist in n_neurons but "
                "no fallback was provided. Set `tokenizer.n_neurons` in the config."
            )
        updates["n_neurons"] = int(fallback_n_neurons)

    if not updates:
        return spec
    warnings.warn(
        f"Checkpoint tokenizer_spec contains placeholder Dist fields; replaced with values from the config: {updates}.",
        stacklevel=2,
    )
    return spec.model_copy(update=updates)


def _load_from_checkpoint(
    checkpoint_path: str,
    cfg: DictConfig | None = None,
) -> torch.nn.Module | None:
    """Load tokenizer from checkpoint via Lightning's ``_load_state``.

    Uses the hyperparameters stored in the checkpoint to reconstruct the
    model, then loads the state dict with ``strict=False`` so any residual
    key mismatch only produces a warning.

    The checkpoint path may be a local file or an MLflow artifact URI (optional;
    MLflow is imported on demand). When ``cfg`` is provided, placeholder
    ``Dist`` fields in the checkpoint's tokenizer_spec are replaced with scalars
    from ``cfg.data.response_window`` and ``cfg.tokenizer.n_neurons``.
    """
    from lightning.fabric.utilities.cloud_io import _load as pl_load
    from lightning.pytorch.core.saving import _load_state
    from lightning.pytorch.utilities.migration.utils import _pl_migrate_checkpoint, pl_legacy_patch
    from ntt.core.model.litmodule import NTTLitModule

    resolved_path = _resolve_checkpoint_path(checkpoint_path)

    fallback_window = float(cfg.data.get("response_window", 0.200)) if cfg is not None else 0.200

    with pl_legacy_patch(), _allow_abstract_dist_load(fallback_window):
        checkpoint = pl_load(resolved_path, map_location="cpu", weights_only=False)

    checkpoint = _pl_migrate_checkpoint(checkpoint, checkpoint_path=resolved_path)
    checkpoint.setdefault(NTTLitModule.CHECKPOINT_HYPER_PARAMS_KEY, {})

    # Lightning types the checkpoint dict loosely; setdefault above guarantees
    # the key exists, and the hasattr check below guards non-dict values.
    hparams = cast("dict[str, Any]", checkpoint.get(NTTLitModule.CHECKPOINT_HYPER_PARAMS_KEY))
    spec = hparams.get("tokenizer_spec") if hasattr(hparams, "get") else None
    if spec is not None and cfg is not None:
        fallback_n_cfg = cfg.get("tokenizer", {}).get("n_neurons") if cfg.get("tokenizer") else None
        fallback_n = int(fallback_n_cfg) if fallback_n_cfg is not None else None
        repaired = _repair_stub_dist_fields(spec, fallback_window, fallback_n)
        if repaired is not spec:
            hparams["tokenizer_spec"] = repaired
            spec = repaired

        # Perceiver-family: the checkpoint stores the path of its NuCLR
        # embeddings file, which may not exist on this machine. Override it
        # with ``tokenizer.embeddings_path`` from the config; without an
        # override, fall back to a file with the same basename under
        # ``./tokenizer_ckpts`` when the stored path is missing.
        if hasattr(spec, "embeddings_path"):
            cfg_emb = cfg.get("tokenizer", {}).get("embeddings_path") if cfg.get("tokenizer") else None
            cur = getattr(spec, "embeddings_path", None)
            new_paths: list[str] | None = None
            if cfg_emb:
                # OmegaConf ListConfig isn't `isinstance(cfg_emb, list)` but it's
                # iterable and not a string — treat any non-string iterable as a list.
                if isinstance(cfg_emb, str):
                    new_paths = [cfg_emb]
                else:
                    try:
                        new_paths = [str(p) for p in cfg_emb]
                    except TypeError:
                        new_paths = [str(cfg_emb)]
            elif cur:
                cur_list = list(cur) if isinstance(cur, (list, tuple)) else [cur]
                new_paths = []
                for p in cur_list:
                    if p and not Path(str(p)).exists():
                        # Try ./tokenizer_ckpts/<basename> as a local-share fallback.
                        basename = Path(str(p)).name
                        candidate = Path("./tokenizer_ckpts") / basename
                        if candidate.exists():
                            new_paths.append(str(candidate))
                            continue
                    new_paths.append(str(p))
            if new_paths is not None and tuple(new_paths) != tuple(cur or ()):
                try:
                    updated = spec.model_copy(update={"embeddings_path": new_paths})
                    hparams["tokenizer_spec"] = updated
                    spec = updated
                except Exception:
                    pass

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        return None

    lit_module = _load_state(NTTLitModule, checkpoint, strict=False)
    assert isinstance(lit_module, NTTLitModule)
    if lit_module.tokenizer is None:
        lit_module.configure_model()

    return lit_module.tokenizer


def _create_from_spec(spec_cfg: Any, cfg: DictConfig) -> torch.nn.Module:
    """Create tokenizer from spec using NTT's create_tokenizer."""
    from ntt.core.model.tokenizer import create_tokenizer
    from ntt.core.model.tokenizer.base import TokenizerSpec

    # If already a TokenizerSpec instance (Hydra resolved it), use directly
    if isinstance(spec_cfg, TokenizerSpec):
        return create_tokenizer(spec_cfg)

    # Otherwise, prepare spec dict with defaults from config
    if not isinstance(spec_cfg, dict):
        spec_dict = dict(spec_cfg)
    else:
        spec_dict = spec_cfg

    # Extract sampling parameters from data config
    bin_size = float(cfg.data.get("bin_size", 0.001))
    response_window = float(cfg.data.get("response_window", 0.200))
    spec_dict.setdefault("window_length_seconds", response_window)
    spec_dict.setdefault("bin_size_ms", bin_size * 1000.0)
    spec_dict.setdefault("n_neurons", 1)

    # Create spec and tokenizer
    if "_target_" in spec_dict:
        import hydra

        spec = hydra.utils.instantiate(spec_dict)
    else:
        from ntt.core.model.tokenizer import get_tokenizer_class

        kind = spec_dict.get("KIND")
        if kind is None:
            raise ValueError("Tokenizer spec must contain 'KIND' field")
        spec_class, _ = get_tokenizer_class(str(kind))
        spec = spec_class.model_validate(spec_dict)

    return create_tokenizer(spec)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_tokenizer_wrapper(cfg: DictConfig, device: torch.device | str) -> TokenizerWrapper | None:
    """Build a TokenizerWrapper from a Hydra config.

    Supports two loading paths:
    1. From checkpoint using NTT's load_tokenizer_from_checkpoint
    2. From spec using NTT's create_tokenizer

    Use `checkpoint_first` to control precedence when both are available.
    Returns None if tokenizer is not enabled in the config.
    """
    tokenizer_cfg = cfg.get("tokenizer")
    if tokenizer_cfg is None or not tokenizer_cfg.get("enabled", False):
        return None

    checkpoint_path = tokenizer_cfg.get("checkpoint_path")
    spec_cfg = tokenizer_cfg.get("spec")
    checkpoint_first = bool(tokenizer_cfg.get("checkpoint_first", True))

    # Try loading in order of preference
    tokenizer = None
    if checkpoint_first and checkpoint_path:
        tokenizer = _load_from_checkpoint(str(checkpoint_path), cfg=cfg)

    if tokenizer is None and spec_cfg is not None:
        tokenizer = _create_from_spec(spec_cfg, cfg)

    if tokenizer is None and checkpoint_path:
        tokenizer = _load_from_checkpoint(str(checkpoint_path), cfg=cfg)

    if tokenizer is None:
        raise ValueError(
            "Could not load tokenizer. Provide either:\n"
            "  - tokenizer.checkpoint_path (for trained tokenizers)\n"
            "  - tokenizer.spec (for spec-based tokenizers)"
        )

    # Extract tokenizer kind from spec
    tokenizer_kind = None
    tokenizer_spec = getattr(tokenizer, "spec", None)
    if tokenizer_spec is not None:
        tokenizer_kind = str(getattr(tokenizer_spec, "KIND", None))

    # Configure frozen/trainable mode
    mode = str(cfg.trainer.get("tokenizer_mode", "frozen"))
    _configure_tokenizer_mode(tokenizer, mode)
    tokenizer = tokenizer.to(device)

    # Extract sampling rate from data config.
    # Use response_bin_size when set — it controls the actual resolution of neural
    # response data passed to the tokenizer, which can differ from the spectrogram
    # bin_size (e.g. 1ms responses vs 10ms spectrograms).
    response_bin_size = cfg.data.get("response_bin_size", None)
    if response_bin_size is not None:
        bin_size = float(response_bin_size)
    else:
        bin_size = float(cfg.data.get("bin_size", 0.001))
    sampling_rate = 1.0 / bin_size  # bin_size in seconds → Hz
    meta = cfg.get("meta", {})
    subject_id = str(meta.get("animal", "monkey"))
    species = str(meta.get("species", "monkey"))
    discrete_target_mode = str(tokenizer_cfg.get("discrete_target_mode", "float_index"))
    discrete_target_vocab_size = tokenizer_cfg.get("discrete_target_vocab_size")
    if discrete_target_vocab_size is not None:
        discrete_target_vocab_size = int(discrete_target_vocab_size)

    # Determine reconstruction rate from tokenizer spec
    reconstruction_rate_hz = sampling_rate
    if tokenizer_spec is not None:
        bin_size_ms = getattr(tokenizer_spec, "bin_size_ms", None)
        if isinstance(bin_size_ms, (int, float)) and not isinstance(bin_size_ms, bool) and bin_size_ms > 0:
            reconstruction_rate_hz = 1000.0 / float(bin_size_ms)

    default_neuron_batch_size = 100
    if tokenizer_spec is not None:
        spec_n = getattr(tokenizer_spec, "n_neurons", None)
        if isinstance(spec_n, int) and spec_n > 0:
            default_neuron_batch_size = spec_n
    neuron_batch_size = int(tokenizer_cfg.get("neuron_batch_size", default_neuron_batch_size))
    if tokenizer_spec is not None:
        # For fixed-width compress_t tokenizers, decode semantics depend on the
        # internal neuron width. Force chunk width to spec.n_neurons.
        if bool(getattr(tokenizer_spec, "compress_t", False)) and not bool(
            getattr(tokenizer_spec, "compress_n", False)
        ):
            spec_n = getattr(tokenizer_spec, "n_neurons", None)
            if isinstance(spec_n, int) and spec_n > 0:
                neuron_batch_size = spec_n

    # Latent scale: normalizes continuous token targets to ~unit variance for diffusion.
    # "auto" defers to calibrate_latent_scale() which the training script calls after
    # dataloader construction. "auto_per_channel" uses per-channel mean/std (for compress_n).
    # A numeric value is used directly.
    latent_scale_cfg = tokenizer_cfg.get("latent_scale", "auto")
    if isinstance(latent_scale_cfg, str) and latent_scale_cfg.lower() == "auto_per_channel":
        latent_scale = 1.0
        _needs_calibration = True
        _per_channel_calibration = True
    elif isinstance(latent_scale_cfg, str) and latent_scale_cfg.lower() == "auto":
        latent_scale = 1.0  # placeholder — caller must call calibrate_latent_scale()
        _needs_calibration = True
        _per_channel_calibration = False
    else:
        latent_scale = float(latent_scale_cfg)
        _needs_calibration = False
        _per_channel_calibration = False

    # Extract BSQ codebook scale if applicable
    bsq_codebook_scale = None
    if tokenizer_spec is not None and str(getattr(tokenizer_spec, "bottleneck_type", "")).lower() == "bsq":
        bottleneck = getattr(tokenizer, "bottleneck", None)
        if bottleneck is not None and hasattr(bottleneck, "codebook_scale"):
            bsq_codebook_scale = float(bottleneck.codebook_scale)

    # Perceiver-family options: latent_layout + rebin factor (data → tokenizer bin size).
    latent_layout = str(tokenizer_cfg.get("latent_layout", "m_as_spatial"))
    rebin_factor = 1
    if reconstruction_rate_hz > 0 and sampling_rate > 0:
        ratio = sampling_rate / reconstruction_rate_hz
        if ratio > 1.0:
            rounded = int(round(ratio))
            if abs(rounded - ratio) > 1e-6:
                warnings.warn(
                    f"Non-integer rebin factor {ratio:.4f} from sampling_rate={sampling_rate} / "
                    f"reconstruction_rate_hz={reconstruction_rate_hz}; rounding to {rounded}.",
                    stacklevel=2,
                )
            rebin_factor = rounded

    # Wavelet-specific config: target slice (latent T cut for the encoder)
    # and the decoder-safe horizon (output T to use for metrics). Both are
    # optional — default to None and the wrapper falls back to no-slice
    # (full latent / full output) behavior.
    cosmos_cfg = cfg.get("tokenizer", {}) if cfg.get("tokenizer") else {}
    cosmos_target_slice = cosmos_cfg.get("cosmos_target_slice", None)
    cosmos_t_safe = cosmos_cfg.get("cosmos_t_safe", None)
    cosmos_target_slice = int(cosmos_target_slice) if cosmos_target_slice is not None else None
    cosmos_t_safe = int(cosmos_t_safe) if cosmos_t_safe is not None else None

    wrapper = TokenizerWrapper(
        tokenizer=tokenizer,
        mode=mode,
        sampling_rate=sampling_rate,
        reconstruction_rate_hz=reconstruction_rate_hz,
        subject_id=subject_id,
        species=species,
        tokenizer_kind=tokenizer_kind,
        discrete_target_mode=discrete_target_mode,
        discrete_target_vocab_size=discrete_target_vocab_size,
        neuron_batch_size=neuron_batch_size,
        latent_scale=latent_scale,
        bsq_codebook_scale=bsq_codebook_scale,
        latent_layout=latent_layout,
        perceiver_rebin_factor=rebin_factor,
        cosmos_target_slice=cosmos_target_slice,
        cosmos_t_safe=cosmos_t_safe,
    )
    # Not dataclass fields: consumers read these via ``getattr(..., False)``,
    # so they are attached dynamically.
    wrapper._needs_calibration = _needs_calibration  # pyright: ignore[reportAttributeAccessIssue] # dynamic attr, read via getattr
    wrapper._per_channel_calibration = _per_channel_calibration  # pyright: ignore[reportAttributeAccessIssue] # dynamic attr, read via getattr

    if mode == "finetune_decoder":
        wrapper.enable_decoder_finetuning()

    return wrapper
