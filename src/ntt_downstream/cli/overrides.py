"""Map stable, promoted top-level flags to per-verb Hydra override strings.

This is the only translation the launcher does. Promoted flag NAMES are stable
across releases; the Hydra keys they map to are internal and may differ per verb
(e.g. tokenizer ckpt is tokenizer.ntt_checkpoint= for decode but
tokenizer.checkpoint_path= for encode). Everything else is plain Hydra passthrough.
"""

from __future__ import annotations

from ntt_downstream.cli.errors import ContractError

_VERBS = {"decode", "encode", "calibrate", "benchmark"}


def promoted_to_hydra(verb: str, flags: dict[str, object]) -> list[str]:
    """Return Hydra override strings for the set promoted flags (None/absent skipped)."""
    if verb not in _VERBS:
        raise ContractError(f"unknown verb {verb!r}; expected one of {sorted(_VERBS)}")
    ov: list[str] = []
    g = flags.get  # promoted flag lookup; returns None for unset flags

    # NOTE: `tokenizer_name` is a promoted flag but maps to NO single-run Hydra
    # key — the sweep launcher (sweeps.py) consumes it for run-naming / MLflow
    # tags. For decode/encode it is intentionally a no-op (documented in
    # --help), so a stray --tokenizer-name on a single run is silently ignored.
    if verb == "decode":
        if g("tokenizer_kind") is not None:
            ov.append(f"tokenizer={g('tokenizer_kind')}")  # config-group select
        if g("tokenizer_ckpt") is not None:
            ov.append(f"tokenizer.ntt_checkpoint={g('tokenizer_ckpt')}")
        if g("data_root") is not None:
            ov.append(f"data_root={g('data_root')}")
        if g("output_root") is not None:
            ov.append(f"log_dir={g('output_root')}")
        if g("seed") is not None:
            ov.append(f"seed={g('seed')}")
    elif verb == "encode":
        if g("tokenizer_ckpt") is not None:
            # `++` (set-or-create) because the encode recipe schema is
            # inconsistent: baselines declare only `tokenizer.enabled`, some
            # recipes (full_grid_flow) have no tokenizer block at all, and
            # --config-name/--config-dir are first-class escape hatches onto
            # arbitrary external recipes. Plain overrides would hit Hydra struct
            # mode ("Key 'checkpoint_path' is not in struct"). The train app
            # reads these keys tolerantly via `tokenizer_cfg.get(...)`.
            ov.append(f"++tokenizer.checkpoint_path={g('tokenizer_ckpt')}")
            ov.append("++tokenizer.enabled=true")
        if g("data_root") is not None:
            ov.append(f"data.dataset_path={g('data_root')}")
        if g("output_root") is not None:
            ov.append(f"hydra.run.dir={g('output_root')}")
        if g("seed") is not None:
            ov.append(f"seed={g('seed')}")  # cfg.seed (via trainer/default.yaml @package _global_)
    # calibrate/benchmark map their promoted flags in the sweeps module, not
    # here — the empty return for those verbs is by design.
    return ov
