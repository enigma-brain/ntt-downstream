"""ntt-downstream CLI: a thin Hydra launcher (the eval CLI contract).

decode/encode default to configs/cli/<verb>.yaml, map promoted flags to per-verb
Hydra keys, then subprocess the train app (with raw Hydra passthrough appended
last). calibrate/benchmark load configs/cli/<verb>.yaml and drive the
orchestration engine (see sweeps module). ntt may override config selection with
--config-name / --config-dir.
"""

from __future__ import annotations

import argparse
import sys

import rootutils

from ntt_downstream.cli import overrides, run
from ntt_downstream.cli.errors import ContractError

_ROOT = rootutils.find_root(search_from=__file__, indicator=".project-root")
_CLI_CONFIG_DIR = f"{_ROOT}/configs/cli"

_SINGLE_RUN = {
    "decode": "ntt_downstream.decoding.train",
    "encode": "ntt_downstream.encoding.train",
}


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--config-name", default=None, help="override the entrypoint config name")
    sp.add_argument("--config-dir", default=None, help="extra Hydra config dir (e.g. ntt's own)")
    sp.add_argument(
        "--tokenizer-kind", default=None, help="tokenizer config group to select (decode only), e.g. fake/frozen"
    )
    sp.add_argument("--tokenizer-ckpt", default=None, help="path or mlflow-artifacts:/ URI")
    sp.add_argument(
        "--tokenizer-name",
        default=None,
        help="tokenizer label for sweep run-naming / MLflow tags "
        "(calibrate/benchmark); no effect on single-run decode/encode",
    )
    sp.add_argument("--data-root", default=None, help="dataset root directory")
    sp.add_argument("--output-root", default=None, help="run output directory")
    sp.add_argument("--seed", default=None, help="global RNG seed")
    mx = sp.add_mutually_exclusive_group()
    mx.add_argument(
        "--dry-run", action="store_true", help="print the assembled command and exit; never subprocesses (GPU-free)"
    )
    mx.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="print the resolved config and exit (GPU-free): Hydra --cfg job for "
        "decode/encode, the overlaid sweep config for calibrate/benchmark",
    )
    # Remaining raw Hydra overrides:
    sp.add_argument("hydra_overrides", nargs="*", help="raw key=value Hydra overrides")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ntt-downstream")
    sub = p.add_subparsers(dest="verb", required=True)
    for verb in ("decode", "encode", "calibrate", "benchmark"):
        _add_common(sub.add_parser(verb))
    return p


def _single_run_command(verb: str, args: argparse.Namespace) -> list[str]:
    config_dir = args.config_dir or _CLI_CONFIG_DIR
    config_name = args.config_name or verb
    promoted = overrides.promoted_to_hydra(
        verb,
        {
            "tokenizer_kind": args.tokenizer_kind,
            "tokenizer_ckpt": args.tokenizer_ckpt,
            "tokenizer_name": args.tokenizer_name,
            "data_root": args.data_root,
            "output_root": args.output_root,
            "seed": args.seed,
        },
    )
    return [
        sys.executable,
        "-m",
        _SINGLE_RUN[verb],
        "--config-dir",
        config_dir,
        "--config-name",
        config_name,
        *promoted,
        *args.hydra_overrides,  # passthrough LAST (Hydra last-wins)
    ]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.verb in _SINGLE_RUN:
            cmd = _single_run_command(args.verb, args)
            return run.execute(cmd, dry_run=args.dry_run, print_resolved=args.print_resolved_config)
        # calibrate / benchmark — delegate to the sweep launcher. Imported lazily
        # so single-run verbs never pay for the orchestration-engine import.
        from ntt_downstream.cli import sweeps

        return sweeps.run_sweep(args.verb, args)
    except (ContractError, ValueError, FileNotFoundError) as exc:
        # ContractError is launcher validation; ValueError/FileNotFoundError are the
        # orchestration engine's user-misconfig signals (e.g. benchmark with no
        # discoverable tokenizers + no spike-times baseline). The single-run path
        # only builds a command list and subprocesses — its train app's own errors
        # happen in the subprocess, not here — so broadening the catch is safe and
        # gives the contract surface a clean `error: …`/exit 2 instead of a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


def decode_cli() -> int:
    return main(["decode", *sys.argv[1:]])


def encode_cli() -> int:
    return main(["encode", *sys.argv[1:]])


def calibrate_cli() -> int:
    return main(["calibrate", *sys.argv[1:]])


def benchmark_cli() -> int:
    return main(["benchmark", *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
