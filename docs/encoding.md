# Encoding

The encoding subpackage trains stimulus → neural-response models on Cowley V4
(image → V4 spike rates) and Wingert auditory cortex (audio → A1 spike rates).
Both branches share a single Hydra entry point that dispatches by
`model.model_type`.

Entry point: `python -m ntt_downstream.encoding.train`
Default config: `configs/encoding/cowley_v4_base.yaml`
Override with: `--config-name <name>` (any YAML in `configs/encoding/`).

> **For running evals, prefer the stable CLI contract:** see [cli.md](cli.md)
> for the `ntt-downstream-encode` launcher (promoted flags, escape hatches).
> This doc explains the internals behind that launcher.

## Pipeline

```
       stimulus                            target neural responses
          │                                          │
          ▼                                          ▼
   ┌────────────────┐               (optional) ┌──────────────────┐
   │  backbone       │                          │ frozen tokenizer │
   │  Hiera (image)  │                          │ (compress_t or   │
   │  CausalConv     │                          │  compress_n)      │
   │  (audio)        │                          └────────┬─────────┘
   └────────┬───────┘                                    │
            │ stimulus latent                            │ z* (B, S, T, D)
            ▼                                            │
   ┌─────────────────┐                                   │
   │ cross-attention │  per-neuron queries               │
   │ readout         │  + session/subject emb            │
   └────────┬───────┘                                    │
            │ per-neuron latent                          │
            ▼                                            │
   ┌──────────────────────────────────────────────────────┐
   │ prediction head — loss_mode-dependent                │
   │   rates  : softplus → Poisson NLL or MSE             │
   │   latent : MSE / smooth_L1 / BCE on tokenizer z*     │
   │   gen    : flow / diffusion / EAR-energy / MAR /     │
   │            cross-entropy (residual VQ)               │
   └────────┬─────────────────────────────────────────────┘
            │
            ▼
       loss → backbone + readout (+ generative head)
```

The cross-attention readout queries the backbone latent with a
*per-neuron query token* (built from `unit_emb`, optionally
`session_emb` and `subject_emb`). This decouples the backbone from
session/subject identity: the same backbone serves multiple sessions, and
per-session readouts only differ in their query embedding tables.

## Two model branches

| Branch | Backbone | DataModule | Lightning module | Default config |
|---|---|---|---|---|
| Image (Cowley V4) | `Hiera` (vision transformer with hierarchical pooling) or `ConvCoreReadout` (Cowley-paper-matched 2D conv) | `CowleyV4DataModule` | `EncodingLitModule` | `cowley_v4_base.yaml` |
| Audio (Wingert auditory) | dilated depthwise-separable TCN (`CausalConv`) over the spectrogram | `WingertAuditoryDataModule` | `AudioEncodingLitModule` | `wingert_base.yaml` |

Dispatch is by `cfg.model.model_type`. `image_encoding_*` and `conv_core_readout`
go through `_train_image`; `audio_encoding` goes through `_train_audio`.

## Config tree

```
configs/encoding/
├── data/                # dataset paths + windowing
│   ├── cowley_v4.yaml
│   ├── cowley_v4_paper_matched.yaml
│   └── wingert_auditory.yaml
├── backbone/            # backbone architecture
│   ├── causal_conv.yaml             # audio
│   ├── hiera_small_112.yaml         # 4.7M-param Hiera, scratch, 112×112
│   ├── hiera_base_112.yaml
│   ├── hiera_pretrained_224.yaml    # pretrained Hiera, 224×224 (frozen by default)
│   └── hiera_pretrained_tiny_224.yaml
├── model/               # readout + loss head
│   ├── conv_core_readout.yaml
│   ├── image_encoding.yaml          # default Cowley readout
│   ├── image_encoding_bce.yaml
│   ├── image_encoding_diffusion.yaml
│   ├── image_encoding_energy.yaml
│   ├── image_encoding_mar.yaml
│   ├── image_encoding_mar_energy.yaml
│   ├── audio_encoding.yaml          # rate-mode (no tokenizer)
│   └── audio_encoding_tokenizer.yaml  # tokenizer-target modes
├── trainer/
│   └── default.yaml                 # optimizer / scheduler / logging / eval cadence
├── cowley_v4_*.yaml     # named experiment recipes (image)
└── wingert_*.yaml       # named experiment recipes (audio)
```

`@package _global_` headers in `data/`, `backbone/`, `model/`, and `trainer/`
let those groups merge into a flat `cfg` (so the experiment YAMLs can override
e.g. `data.bin_size` directly).

## Cowley V4 image experiments

All Cowley experiments share `data: cowley_v4`. Pick a backbone
(`hiera_small_112` from scratch or the `hiera_pretrained_224` frozen ImageNet
weights) and a loss mode.

| Recipe | Backbone | Loss / head | Tokenizer | Notes |
|---|---|---|---|---|
| `cowley_v4_base` | hiera_small_112 (scratch) | Poisson NLL | none | Default smoke baseline. |
| `cowley_v4_conv` | (none — uses ConvCoreReadout) | MSE on rates | none | Cowley et al. 2026 paper-matched conv baseline (stacked 2D core + Gaussian readout). |
| `cowley_v4_cowley_matched` | hiera_pretrained_224 (frozen) | MSE on z-scored rates | none | Hiera body, Cowley-paper data protocol (50–150 ms scalar rates, trial-averaged). |
| `cowley_v4_diffusion` | hiera_pretrained_224 (frozen) | DDPM diffusion | cs-* (`compressed_sensing`, config-built, frozen) | DiffLoss head over cs-* tokenizer latents. |
| `cowley_v4_full_grid_flow` | hiera_pretrained_224 (frozen) | Flow matching | cs-* | Joint flow over `(B, S=N, T=1, D=48)` via `FullGridFlowMatchingHead`. |
| `cowley_v4_energy` | hiera_pretrained_224 (frozen) | EAR energy score | cs-* | Score-based loss; per-channel z-score required. |
| `cowley_v4_mar` | hiera_small_112 (scratch) | MAR + flow matching | cs-* | Masked autoregressive generation with reveal schedule. |
| `cowley_v4_mar_energy` | hiera_pretrained_224 (frozen) | MAR + EAR energy | conv-* VAE checkpoint | MAR with energy-score generator. |
| `cowley_v4_perceiver_mse` | hiera_pretrained_224 (frozen) | MSE on Perceiver latents | Perceiver K9_M16_ld16 (checkpoint) | Restricted to the Pepe sessions, the only ones the NuCLR embeddings cover. |
| `cowley_v4_perceiver_mse_mflat` | hiera_pretrained_224 | MSE on Perceiver latents | Perceiver, m_flat layout | `(B, 1, T, M·C=256)` instead of `(B, M, T, C)`. |
| `cowley_v4_perceiver_full_grid_flow` | hiera_pretrained_224 | Flow matching | Perceiver | Joint denoiser over the full Perceiver latent field. |
| `cowley_v4_wanneural_vq` | hiera_pretrained_224 | Cross-entropy on residual codebooks | wanneural VQ checkpoint (discrete) | `ce_vocab_size=1024` and `num_codebooks=4` are derived from the tokenizer at setup, not set in the recipe. |

### Cowley V4 — data knobs (`data:`)

| Key | Default | Notes |
|---|---|---|
| `dataset_path` | `./data/cowley_v4_compact_2024` | |
| `response_window` | 0.200 (s) | Length of the response window. Cowley-matched recipes set this to 0.100 with `response_offset=0.050`. |
| `bin_size` | 0.001 (s) | 1 ms bins → `T = response_window / bin_size`. |
| `response_offset` | 0.0 | Skip the first N seconds of the response (V4 latency). |
| `train_on_averaged` | false | When true, train on trial-averaged responses (one item per unique image). Used by Cowley-matched recipes. |
| `author_split` | null | `"train"` / `"val"` / `"test"` to follow the dataset's published split, or `null` for all sessions. |
| `session_ids` | null | Allow-list specific sessions. |
| `subjects` | null | `"Pepe"` or `["Raul", "Wakko"]` to filter by animal. |
| `image_size` | null | Default native 112×112; set `[224, 224]` for the pretrained Hiera backbone. |
| `min_snr` | null | SNR threshold (Pospisil & Bair 2021) for unit filtering. Cowley-matched: 0.15. |

### Cowley V4 — backbone knobs

`hiera_small_112` (4.7M params, scratch): `hiera_embed_dim`, `hiera_num_heads`,
`hiera_stages`, `hiera_q_pool`, `hiera_q_stride`, `hiera_mask_unit_size`,
`hiera_patch_*`, `hiera_drop_path_rate`, `hiera_mlp_ratio`. Token grid:
56 → 28 → 14 → 7 through 3 q_pool stages.

`hiera_pretrained_224`: `pretrained_backbone="hiera_base_224"`,
`pretrained_backbone_kwargs={}`, `freeze_backbone=true` (default). The
published recipes without a pretrained backbone use `hiera_small_112` trained
from scratch. The `hiera_pretrained_*` backbones download Meta's Hiera
weights, which are licensed CC BY-NC 4.0.

### Cowley V4 — readout knobs

| Key | Default | Notes |
|---|---|---|
| `dim_head` | 48 | Per-head dim of the cross-attention readout. |
| `num_heads` | 4 | |
| `readout_mlp_ratio` | 4 | FFN expansion in the readout. |
| `readout_use_session_embedding` | true | Adds learned `session_emb[session_id]` to the per-neuron query. |
| `readout_use_subject_embedding` | true | Adds `subject_emb[subject_id]`. |
| `readout_use_gated_ffn` | false | GeGLU-style gating in the readout FFN. |
| `readout_layer_types` | null | `["cross"]` by default (single cross-attn). Pass e.g. `["cross", "cross", "self", "cross"]` for a deep readout with intra-population self-attention. |
| `learnable_softplus` | false | When true, output activation is `ParametrizedSoftplus(w, b)` instead of `nn.Softplus()`. |
| `softplus_beta` | 0.1 | Sharpness; init `a` if learnable. |
| `token_t_latent` | 1 | Number of latent timesteps the prediction head emits. Use 1 for `compress_t` tokenizers whose encoder yields `t_comp=1` at the current `data.response_window`. Set >1 (e.g. 2 for 100 ms tokenizer over 200 ms window) when the widened window forces `t_comp>1`. |
| `deep_token_head` | true | Use a ResMLP head with FiLM modulation instead of a plain 2-layer Linear-GELU-Linear head. |
| `deep_token_head_depth` / `_width_mult` / `_use_film` / `_dropout` | 2 / 2.0 / true / 0.0 | Deep head shape. FiLM is zero-init so the head starts as a plain ResMLP. |

### Cowley V4 — loss-specific knobs

`model.loss_mode` switches between:

| Mode | What it predicts | Required cfg block |
|---|---|---|
| `poisson` | rate per `(B, T, N)` (softplus-activated) | none — no tokenizer |
| `mse_rate` | rate; optionally `response_zscore=true` to z-score targets per neuron | none |
| `mse` | latent `(B, S, T, D)` | tokenizer enabled. Optional `mse_center_weight` and `cosine_sim_weight` shape the loss to focus on image-conditional residuals (subtract per-neuron batch mean). |
| `smooth_l1_latent` | latent | tokenizer enabled |
| `bce` | binary latent (BSQ tokenizers) | tokenizer with `bsq_codebook_scale` set |
| `flow_matching` | latent via flow matching | tokenizer + (`flow_head_*` for pointwise head, **or** `use_full_grid_denoiser=true` + `full_grid_*` for joint denoiser) |
| `diffusion` | latent via DDPM | tokenizer + `model.diffusion: {target_channels, z_channels, depth, width, num_sampling_steps}` |
| `energy` | latent via EAR energy score | tokenizer + `model.energy: {target_channels, z_channels, width, depth, noise_channels, alpha, train_temperature}`. Per-channel z-score is required. |
| `cross_entropy` | discrete codebook indices | discrete-token tokenizer + `ce_vocab_size`, `num_codebooks` (residual VQ), `ce_sample_temperature`, optional `token_t_latent>1` |

`token_prediction_dim` is auto-derived from the tokenizer spec when not set.
For `loss_mode=energy`, `target_channels` and `z_channels` auto-fall-back to
the tokenizer's `latent_dim`. For `loss_mode=flow_matching` with
`use_full_grid_denoiser=true`, the head consumes the unified
`(B, S, T, D)` latent and denoises it jointly.

`diffusion_batch_mul` is a variance-reduction repetition of the
`(target, z)` pair before the loss; used by both `diffusion` and `energy`
heads. The `diffusion`, `energy` and `mar` recipes use 4; `mar_energy` and the
full-grid flow recipes use 2, because MAR pays this cost per reveal step.

#### Full-grid flow head knobs

| Key | Default | Notes |
|---|---|---|
| `use_full_grid_denoiser` | false | Replaces pointwise flow head with a joint denoiser over `(B, S, T, D)`. |
| `full_grid_hidden_channels` | 64 | |
| `full_grid_num_blocks` | 4 | Number of `MSpaceBlock` denoiser layers. |
| `full_grid_num_heads` | 4 | |
| `full_grid_mlp_ratio` | 2 | |
| `full_grid_kernel_size` | 7 | Depthwise temporal kernel; only used when `full_grid_use_s4=false`. |
| `full_grid_use_s4` | false | Switch the temporal mixer to S4. Recommended when `T` is much larger than ~20. |
| `trainer.tokenizer_eval.n_flow_steps` | 5 | Increase to 50 for full-grid inference quality. |
| `trainer.tokenizer_eval.flow_solver` | `euler` | Use `heun` for higher-order ODE solving. |

#### Flow-matching knobs

The flow heads train rectified flow (straight-line interpolant, t=0 is
noise). Four optional upgrades are config-gated and default to off:

| Knob | Where | Default | What it does |
|---|---|---|---|
| `model.flow_time_sampling` | train | `uniform` | `logit_normal` = SD3-style time sampling concentrated mid-trajectory |
| `model.flow_logit_normal_mean` / `_std` | train | `0.0` / `1.0` | logit-space location/scale for the above |
| `model.flow_ot_coupling` | train | `false` | minibatch-OT noise↔target pairing; straightens paths for few-step sampling |
| `model.flow_cond_dropout` | train | `0.0` | conditioning dropout against a learned null embedding; required for CFG |
| `trainer.tokenizer_eval.flow_shift` | eval | `1.0` | shifted ODE grid; >1 concentrates steps near the noise end |
| `trainer.tokenizer_eval.cfg_scale` | eval | `1.0` | classifier-free guidance scale; needs `flow_cond_dropout > 0` |

These apply to all three flow heads (pointwise, full-grid, LDNS). The
Wingert (audio) path reads the training knobs from the `model.flow:` block
instead (`time_sampling`, `logit_normal_mean`, `logit_normal_std`,
`ot_coupling`, `cond_dropout`). The MAR path reads the eval knobs from
`trainer.mar_eval` (`flow_shift`, `cfg_scale`).

The end-of-training `final_eval` pass honors `flow_shift` too: set
`trainer.final_eval.flow_shift` to override it, otherwise it falls back to
`trainer.tokenizer_eval.flow_shift` so the periodic and final passes use the
same inference grid. (final_eval has its own `flow_solver`, default `heun`, and `n_flow_steps`,
and does not apply CFG: `cfg_scale` stays 1.0 there.)

Setting `cfg_scale != 1` on a model trained with `flow_cond_dropout: 0`
raises a `ValueError` — there is no null embedding to guide against. The
`null_cond` parameter is registered only when `flow_cond_dropout > 0`.

#### MAR (masked autoregressive) knobs

| Key | Default | Notes |
|---|---|---|
| `mar_dim` | 128 | MAR transformer width. |
| `mar_depth` | 4 | |
| `mar_num_heads` | 8 | |
| `mar_mlp_ratio` | 4 | |
| `mar_attn_dropout` / `mar_proj_dropout` | 0.1 | |
| `mask_ratio_min` | 0.7 | Minimum mask ratio at training time (`MaskingGenerator`). |
| `mask_ratio_std` | 0.25 | Std of mask ratio sampled per sample. |
| `unmask_schedule` | `cosine` | Schedule for revealing masked tokens at inference. |
| `num_buffer_tokens` | 8 | Extra learnable buffer tokens prepended to the MAR sequence. |
| `use_diffusion_pos_embed` | true | Use diffusion-style pos-embed for the MAR transformer. |
| `flow_head_width` / `flow_head_depth` | 96 / 3 | Pointwise flow generator on the masked positions. |
| `num_sampling_steps` | 100 | MAR flow-step count. |
| `trainer.mar_eval.{n_mar_steps, n_flow_steps, temperature, flow_solver}` | 8 / 5 / 1.0 / `euler` | Eval-time sampler. |

#### Cross-entropy on residual VQ

Required cfg: a discrete-token tokenizer (e.g. wanneural VQ, transformer-vq-*, BSQ). The
following are auto-derived at module setup:
- `ce_vocab_size = codebook_size` per quantizer
- `token_prediction_dim = z_dim` (e.g. 16-dim feature predicted per latent slot)
- `num_codebooks` (residual VQ depth — 4 quantizers ⇒ `4 × 1024` logits per slot)

`ce_sample_temperature` (default 1.0) controls eval-time sampling.

For the wanneural VQ recipe (`cowley_v4_wanneural_vq`), the tokenizer
compresses 200 ms (200 bins at 1 ms) into 2 latent time tokens
(`temporal_compression=100×`); the temporal-tiled readout must produce
`token_t_latent=2` prediction slots per neuron.

## Wingert auditory cortex experiments

All Wingert experiments share `data: wingert_auditory` and
`backbone: causal_conv`.

| Recipe | Loss | Tokenizer | Notes |
|---|---|---|---|
| `wingert_base` | Poisson NLL | none | Default rate baseline. |
| `wingert_mse` | MSE on rates | none | |
| `wingert_diffusion` | DDPM diffusion | cs-* (`compressed_sensing`) | `latent_scale=auto`. |
| `wingert_flow` | Flow matching | cs-* | |
| `wingert_energy` | EAR energy | cs-* | `latent_scale=auto_per_channel`. |
| `wingert_cross_entropy` | Cross-entropy on codebook indices | **discrete** tokenizer (placeholder spec; replace before running) | Requires a real discrete-token tokenizer checkpoint; the bundled spec is a placeholder. |

### Wingert — data knobs

| Key | Default | Notes |
|---|---|---|
| `dataset_path` | `./data/wingert_auditory_cortex_2026` | |
| `bin_size` | 0.01 (s) | 10 ms bins (100 Hz). Spectrogram and model resolution. |
| `response_bin_size` | null | If null, same as `bin_size`. Set e.g. `0.001` for 1 ms tokenizer bins. |
| `response_normalization` | `"none"` | |
| `spectrogram_fs` | 100.0 | Hz; must match `1 / bin_size`. |
| `n_freq_channels` | 32 | |
| `crop_length` | 1000 | Training crop = 1000 spectrogram bins = 10 s. |
| `session_ids` / `subjects` | null | Filters. |

### Wingert — backbone (CausalConv) knobs

| Key | Default | Notes |
|---|---|---|
| `conv_rank` | 48 | Feature dim of the dilated TCN. |
| `conv_kernel_size` | 7 | |
| `conv_dilations` | `[1, 2, 4, 8]` | Dilation schedule across the TCN stack. |

### Wingert — readout knobs

Same query-attention idea as the image branch, plus inter-window self-attention:

| Key | Default | Notes |
|---|---|---|
| `window_size` | 10 | M=10 spectrogram bins per window (100 ms at 100 Hz). When a tokenizer is active, this should match `tokenizer_window_length_seconds`. |
| `n_self_attn_layers` | 2 | Self-attention over windows after the convolutional encoder. |
| `n_self_attn_heads` | 4 | |
| `rope_max_seq_len` | 512 | RoPE positional encoding limit. |
| `dim_head`, `num_heads`, `readout_mlp_ratio` | 48 / 4 / 4 | Cross-attention readout. |
| `readout_use_subject_embedding` | true | (Audio readout doesn't carry session emb by default.) |
| `dropout` | 0.1 | |

### Wingert — loss-specific knobs

`audio_encoding.yaml` is the rate-mode model (Poisson / MSE).
`audio_encoding_tokenizer.yaml` is the tokenizer-target model — supports
`mse_latent`, `smooth_l1_latent`, `bce`, `flow_matching`, `diffusion`,
`energy`, `cross_entropy`. Loss-head sub-blocks live under `model.flow:`,
`model.diffusion:`, `model.energy:` (same shape as the image counterparts).

`tokenizer_aux_loss_weight` (default 0.0) optionally adds a weighted loss
through the tokenizer's auxiliary head; `decoder_recon_loss_weight` /
`decoder_recon_loss_mode` (`"poisson"` or `"mse"`) optionally pass predictions
back through the tokenizer's decoder for a reconstruction loss.

## Tokenizer wrapper (`tokenizer:` block)

`TokenizerWrapper` (in `tokenizer_wrappers/encoding_wrapper.py`) bridges the
encoding pipeline ↔ ntt tokenizers. It supports two compression patterns:

- **`compress_t`** (per-neuron temporal compression, e.g. wanneural,
  cs-*): emits `(B, T', U=N, D)` latents. Variable `N` per
  session is fine.
- **`compress_n`** (population compression, e.g. Perceiver): emits
  `(B, T', M, D)` where `M` is a fixed slot count. For sessions with
  `N > spec.n_neurons`, the wrapper builds a `CompressNSessionPlan` to split
  neurons into balanced chunks, pad each chunk, encode it, and stack — so
  `S = num_chunks` in the unified output.

The wrapper presents both as a unified `(B, S, T, D)` tensor.

| Key | Default | Notes |
|---|---|---|
| `enabled` | false | Master switch. Rate-mode recipes set `false`; latent-mode recipes set `true`. |
| `checkpoint_first` | true | When true and `checkpoint_path` is set, load the tokenizer from the checkpoint (a Lightning ckpt path or `mlflow-artifacts:` URI; `hf://` checkpoints are materialised to a local file by ntt's launcher first), falling back to `spec` if that yields nothing. When false, build from `spec` first and use `checkpoint_path` only as the fallback. |
| `checkpoint_path` | null | Local path or URI. Required (`???`) by the Perceiver, `wanneural_vq` and `mar_energy` recipes. |
| `embeddings_path` | unset | Location of the NuCLR pre-computed embeddings (Perceiver tokenizers). Required (`???`) by the Perceiver recipes. |
| `spec` | (depends) | `_target_` instantiates an ntt `TokenizerSpec` (e.g. `CompressedSensingTokenizerSpec`, `PerceiverTokenizerSpec`). All remaining keys are forwarded as spec kwargs. |
| `latent_scale` | 1.0 | Per-tokenizer normalization. `"auto"` calibrates to unit variance over ~50 train batches; `"auto_per_channel"` calibrates per-channel (required for energy loss). |
| `latent_layout` | `m_as_spatial` | Perceiver only. `m_as_spatial` → unified `(B, S=M=16, T=L, D=C=16)`; `m_flat` → `(B, S=1, T=L, D=M·C=256)`. |
| `tokenizer_window_length_seconds` | null | Audio: when null, the wrapper infers the window from the spec; set explicitly to override. |

`trainer.tokenizer_mode` (default `"frozen"`) controls whether the tokenizer
parameters update during training. `"trainable"` allows gradient through the
tokenizer; `"finetune_decoder"` only unfreezes the decoder. Use
`trainer.decoder_recon_loss_weight > 0` (and `decoder_lr`) to add a
reconstruction loss through the decoder.

## Trainer knobs

`configs/encoding/trainer/default.yaml` is the canonical reference. Highlights:

| Key | Default | Notes |
|---|---|---|
| `lr` | 1e-3 | AdamW base LR. |
| `backbone_lr_multiplier` | 1.0 | Multiplier applied to backbone params (set <1 when fine-tuning a pretrained Hiera). |
| `max_steps` | 100000 | Used directly by Lightning. |
| `n_epochs` / `max_epochs` | null | Set one of these to switch to epoch-based training. |
| `precision` | `bf16-mixed` | |
| `gradient_clipping` | 1.0 | |
| `gradient_accumulation_steps` | 1 | |
| `weight_decay` | 0.01 | |
| `betas`, `eps` | `[0.9, 0.999]`, 1e-8 | |
| `scheduler.T_max` | `${trainer.max_steps}` | Cosine schedule horizon. |
| `scheduler.eta_min` | 1e-5 | Minimum LR. |
| `scheduler.warmup_steps` | 1000 | Linear warmup from ~0 to `lr`. |
| `eval_skip_seconds` | 0.0 | Skip first N seconds in correlation eval (e.g. 0.05 for V4 latency). |
| `eval_end_seconds` | null | Truncate response at this time before eval. |
| `eval_rebin_hz` | 200 | Rebin predictions and targets to this rate before eval (`null` = native). |
| `eval_max_batches_train` / `_val_avg` / `_latent_quality` / `_nc_r2` | 50 / null / 20 / 200 | Budget caps for the corresponding eval pass (`0` disables that eval). |
| `resume_from_ckpt` | null | Path passed straight to `Trainer.fit(ckpt_path=...)`. |
| `save_every_n_train_steps` | 6000 | Defensive periodic checkpoint, independent of validation. Set so a crash before first val still leaves a recoverable last.ckpt. `null` disables. |
| `logging` | true | Master logging toggle (MLflow + TensorBoard). `false` runs with no MLflow at all; see [Running without MLflow](#running-without-mlflow). |
| `experiment_name` | `cowley-v4-encoding` | Wingert recipes set `wingert-auditory-encoding`. |
| `mlflow_tracking_uri` | `${oc.env:MLFLOW_TRACKING_URI,null}` | `null` when the env var is unset; required only when `logging=true`. |
| `tensorboard_logging` | true | TensorBoard runs alongside MLflow when `logging=true`. |
| `logging_frequency_loss` | 50 | |
| `logging_frequency_validation` | 1.0 | int = number of steps; float = fraction of an epoch. |
| `warmup_loss_mode` | null | Staged warmup — use a simpler loss (`"mse"`) for `warmup_steps` before switching to `model.loss_mode`. |
| `tokenizer_eval.{enabled, sample_temperature, flow_solver, n_flow_steps}` | true / 1.0 / `euler` / 5 | Sampler used when reconstructing rates from generated latents at eval. |
| `mar_eval.{n_mar_steps, n_flow_steps, temperature, flow_solver}` | 8 / 5 / 1.0 / `euler` | MAR-specific eval sampler. |
| `final_eval.{enabled, n_samples, flow_solver, n_flow_steps, flow_shift}` | true / 32 / `heun` / null / null | End-of-training eval of the val-best checkpoint (image path only). `null` steps/shift fall back to `tokenizer_eval`. |
| `startup_roundtrip.{enabled, max_batches, strict}` | true / 20 / false | Pre-fit encode-decode roundtrip on a val slice (tokenizer-ceiling diagnostic, MLflow only); `strict=true` aborts on a degenerate tokenizer instead of warning. |
| `final_eval.test` | true | Also score that same checkpoint on the **test** split (single-trial + repeat-averaged). Selection never looks at test. |
| `limit_train_batches` / `limit_val_batches` | null | Smoke-test overrides (forwarded to Lightning). |
| `strategy` | null | DDP strategy override; default is `ddp_find_unused_parameters_true` when `devices > 1`. |

## Validation metrics

`EncodingLitModule` and `AudioEncodingLitModule` log:

| Metric | Meaning |
|---|---|
| `val/corr/<session>` | Per-image temporal Pearson correlation (T > 1 only). |
| `val/rate_corr/<session>` | Cross-image Pearson on time-summed rates. **Monitored** for checkpointing. |
| `val/rate_corr/overall` | Mean across sessions. |
| `val/rate_corr_trial_avg/<session>` | Same metric on repeat-averaged responses. |
| `val/corr_trial_avg/<session>` | Temporal corr on repeat-averaged responses. |
| `val_nc_r2/<session>` | Noise-corrected R² (Pospisil & Bair 2021). |

In generative modes, `latent_quality` metrics (sampled latent vs. target
under the tokenizer's metric of choice) are logged in addition to the rate
metrics — capped at `eval_max_batches_latent_quality` for cost.

## Outputs

Each run writes to:
- The Hydra output dir (resolved from `${now:...}` — DDP children inherit
  rank 0's dir via `NTT_HYDRA_RUN_DIR`).
- `checkpoints/<run_name>/` under the Hydra dir (or `trainer.local_checkpoint_dir`
  if set). Top-3 by `val/rate_corr/overall` plus `last.ckpt` plus an optional
  step-based defensive snapshot.
- MLflow (run name = `<model_tag>_<8-hex>`), with the full `cfg` flattened to
  params (truncated at 250 chars per MLflow's limit), plus `model_params` and
  `model_trainable_params`.
- TensorBoard events under `<run_dir>/tb/<run_name>/`.
- `downstream_encoding_results.json`: the run's results record (schema v2;
  see [results.md](results.md)), written once after final eval and uploaded to
  MLflow / W&B. `val_best` / `val_last` come from the periodic-val snapshots;
  `val_final` / `test` come from final eval (image path only; `test` is the
  authoritative score). If final eval fails or is disabled, the record has no
  `test`, and `metadata.final_eval` says why.
- `final_eval.csv` / `final_eval_test.csv`: human-readable final-eval reports
  (not uploaded). MLflow scalars: `final/...` (val) and `final_test/...` (test).

## Running without MLflow

MLflow is optional. If you have no tracking server, add `trainer.logging=false`
to any encoding command:

```bash
uv run python -m ntt_downstream.encoding.train trainer.logging=false
```

The run then needs no `MLFLOW_*` environment variables. It trains, runs final
eval, and writes everything that does not depend on a logger: checkpoints,
`final_eval.csv` / `final_eval_test.csv`, and the full
`downstream_encoding_results.json` record (with `metadata.mlflow_run_id: null`).
What you lose is the MLflow run itself (flattened params, scalars, figures),
the TensorBoard events, the learning-rate monitor, and the pre-fit
tokenizer-ceiling diagnostic, which only reports to MLflow.

With `trainer.logging=true` (the default) and no URI from either
`MLFLOW_TRACKING_URI` or `trainer.mlflow_tracking_uri`, the run stops before
training with an error that says so.

The record is the run's output, so a run that cannot write it fails rather than
exiting cleanly. The config is also fully resolved at startup (the record stores
the resolved config), so a broken interpolation stops the run before training
instead of after it.

## Running on a cluster

Encoding has no sbatch wrapper. Run the entry point inside your own job
script, with Hydra overrides after the command:

```bash
uv run python -m ntt_downstream.encoding.train \
    --config-name cowley_v4_perceiver_full_grid_flow \
    trainer.max_steps=50000 \
    trainer.devices=1 \
    trainer.lr=5e-4
```

Multi-GPU runs use Lightning DDP (see [environment.md](environment.md#slurm-identity)).
