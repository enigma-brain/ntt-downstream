"""Sweep launcher: drive the orchestration engine from flat Hydra sweep configs.

``app.main`` dispatches the ``calibrate`` / ``benchmark`` verbs here. This module
loads ``configs/cli/<verb>.yaml`` (flat config data, no Hydra ``defaults:`` list,
whose keys ARE the orchestration engine's argparse dest names), overlays the
launcher's promoted CLI flags, and feeds the result to the EXISTING engine via the
config-object seam in ``orchestration`` — ``calibrate_config_from_mapping`` /
``benchmark_from_mapping``. No engine logic is reimplemented here; the engine's
own ``--dry-run`` plan-printing is reused by setting ``dry_run`` in the mapping.

Tokenizer checkpoint URIs (incl. ``mlflow-artifacts:/`` URIs) pass through
UNCHANGED — the per-cell ``train.py`` invocation resolves them.
"""

from __future__ import annotations

import argparse
import logging
import shlex

import rootutils
from omegaconf import OmegaConf

from ntt_downstream.cli.errors import ContractError
from ntt_downstream.decoding import orchestration

_ROOT = rootutils.find_root(search_from=__file__, indicator=".project-root")


def _load_sweep_cfg(verb: str, args: argparse.Namespace) -> dict:
    """Load the flat sweep YAML for ``verb`` as a plain dict.

    Key-level validation happens downstream in the engine seam
    (``orchestration._namespace_from_mapping`` rejects unknown dests), so a
    typo'd or stale sweep key fails loudly instead of being silently ignored.
    """
    config_dir = args.config_dir or f"{_ROOT}/configs/cli"
    config_name = args.config_name or verb
    raw = OmegaConf.load(f"{config_dir}/{config_name}.yaml")
    cfg = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(cfg, dict):
        raise ContractError(f"{config_name}.yaml: expected a mapping of sweep keys, got {type(cfg).__name__}")
    return cfg


def _overlay_common(mapping: dict[str, object], args: argparse.Namespace) -> None:
    """Apply the promoted flags + passthrough shared verbatim by every sweep verb.

    A CLI flag wins only when explicitly set (not None).
    """
    if args.data_root is not None:
        mapping["data_root"] = args.data_root
    if args.hydra_overrides:
        # The engine does shlex.split(args.extra or ""); join the raw passthrough
        # tokens back into a single string it can re-split.
        mapping["extra"] = shlex.join(args.hydra_overrides)
    mapping["dry_run"] = args.dry_run
    # Make the installed CLI cwd-independent: the engine otherwise defaults
    # repo_root to Path.cwd() and would resolve configs/decoding/... against the
    # caller's directory. Anchor it to the discovered project root (_ROOT, found
    # via the .project-root marker) unless the sweep YAML pins an explicit value.
    if not mapping.get("repo_root"):
        mapping["repo_root"] = str(_ROOT)


def _calibrate_mapping(cfg: dict, args: argparse.Namespace) -> dict[str, object]:
    """Overlay promoted flags onto the calibrate YAML (keys are engine dests).

    The engine seam seeds argparse defaults, so ``null`` YAML values that we pass
    through as ``None`` simply match the default (and trigger the engine's
    ``_pick`` YAML-fallback) — nothing special is needed for them.
    """
    mapping: dict[str, object] = dict(cfg)

    if args.tokenizer_kind is not None:
        mapping["tokenizer_kind"] = args.tokenizer_kind
    if args.tokenizer_ckpt is not None:
        # Pass the ckpt URI through verbatim — per-cell train.py resolves it.
        mapping["ckpt"] = args.tokenizer_ckpt
    if args.tokenizer_name is not None:
        mapping["tokenizer_name"] = args.tokenizer_name
    if args.output_root is not None:
        # output_root is the stable promoted name; calibrate's engine dest is calib_root.
        mapping["calib_root"] = args.output_root
    if args.seed is not None:
        # _resolve_calibrate runs _as_list(args.seeds); a one-element list is the
        # natural single-seed override.
        mapping["seeds"] = [args.seed]
    _overlay_common(mapping, args)
    return mapping


def _benchmark_mapping(cfg: dict, args: argparse.Namespace) -> dict[str, object]:
    """Overlay promoted flags onto the benchmark YAML (keys are engine dests)."""
    mapping: dict[str, object] = dict(cfg)

    if args.seed is not None:
        # benchmark's seed dest is an int (run_benchmark does str(args.seed)).
        mapping["seed"] = int(args.seed)
    # Tokenizer promoted flags vs benchmark's COHORT model: benchmark discovers N
    # frozen tokenizers from ckpt_root, so a single --tokenizer-name maps naturally
    # to a one-element `tokenizers` cohort (run_benchmark's --tokenizers filter).
    # --tokenizer-ckpt / --tokenizer-kind do NOT map cleanly onto the cohort model
    # (one ckpt/kind can't parameterize a discovered cohort), so they are
    # intentionally IGNORED for benchmark rather than forced into a wrong shape.
    if args.tokenizer_name is not None:
        mapping["tokenizers"] = [args.tokenizer_name]
    _overlay_common(mapping, args)
    return mapping


def run_sweep(verb: str, args: argparse.Namespace) -> int:
    """Load the sweep config for ``verb``, overlay promoted flags, and run the engine."""
    # The orchestration engine emits its plan (job queue + per-cell train.py
    # commands — the documented GPU-free dry-run contract) via log.info. The CLI
    # launcher otherwise never configures logging, leaving the root logger at
    # WARNING and silently swallowing the plan. Configure INFO here (sweep path
    # only — single-run decode/encode just print() their command). basicConfig is
    # a no-op if logging is already configured, which is fine.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = _load_sweep_cfg(verb, args)
    if verb == "calibrate":
        mapping = _calibrate_mapping(cfg, args)
    elif verb == "benchmark":
        mapping = _benchmark_mapping(cfg, args)
    else:
        # Defensive only: app.main never routes a non-sweep verb here. A programming
        # error, not bad contract input, so ValueError (not ContractError).
        raise ValueError(f"unknown sweep verb {verb!r}; expected calibrate|benchmark")

    if args.print_resolved_config:
        # The sweep analogue of single-run `--cfg job`: print the fully overlaid
        # engine input (sweep YAML + promoted flags + passthrough) and exit
        # WITHOUT touching the engine. The engine's plan preview stays on
        # --dry-run; this flag must never launch cells.
        print(OmegaConf.to_yaml(OmegaConf.create(mapping)))
        return 0
    if verb == "calibrate":
        return orchestration.run_calibrate(orchestration.calibrate_config_from_mapping(mapping))
    return orchestration.benchmark_from_mapping(mapping)
