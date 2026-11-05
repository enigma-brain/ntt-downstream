# Documentation

`ntt-downstream` evaluates `ntt` neural tokenizers on two tasks:

- **Decoding**: predict cursor velocity from spikes via the POYO decoder
  (frozen ntt tokenizer → adapter → POYO bypass path).
- **Encoding**: predict neural responses from a stimulus (Cowley V4 image
  responses; Wingert auditory cortex responses), optionally with a frozen
  tokenizer in the loss.

Each subpackage has its own Hydra entry point and config tree. `ntt`'s own docs
(`downstream_eval.md`, `downstream_maintainer_internals.md`,
`downstream_encoding_advanced.md`) describe the ntt-side launcher that drives
these evals.

## Where to look

| File | Topic |
|---|---|
| [getting-started.md](getting-started.md) | Install (standalone vs. embedded in `ntt`), env vars, first runs (decoding + encoding) |
| [cli.md](cli.md) | The eval CLI contract (the four launcher verbs, promoted flags, ntt integration) |
| [decoding_orchestration.md](decoding_orchestration.md) | The sweep-orchestration engine behind `calibrate` / `benchmark` |
| [repo-organization.md](repo-organization.md) | What lives in each directory, third-party code and licences |
| [results.md](results.md) | The results record every run writes (schema v2), cohort records, local files |
| [environment.md](environment.md) | Cache paths, MLflow / wandb / SLURM env vars |
| [decoding.md](decoding.md) | POYO bypass pipeline, tokenizer kinds, all knobs |
| [encoding.md](encoding.md) | Cowley V4 + Wingert configs, loss modes, all knobs |

## 30-second mental model

**Decoding.** A tokenizer (fake / spike-times / Poisson / frozen ntt) emits a
`TokenizerOutput`; an adapter converts it to POYO's `[B, N, D_model]` input,
optionally adding unit identity and prepending sentinels; the result is fed
into a POYO subclass (`POYOWithEmbeddingBypass`) via the `input_embeddings`
argument. Everything below the input layer is the original POYO. The same
script handles single-session and multi-session+tiered-eval runs.

**Encoding.** A backbone (Hiera image / dilated CausalConv audio) extracts a
stimulus latent. A cross-attention readout queries this latent with per-neuron
queries and predicts either firing rates (Poisson / MSE) or tokenizer-latent
targets through one of several generative heads (flow matching, diffusion,
EAR energy score, MAR, cross-entropy on discrete codebooks). One Hydra entry
point dispatches to image vs. audio based on `model.model_type`.
