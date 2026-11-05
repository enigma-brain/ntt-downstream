"""MLflow parent-run helper for the decoding orchestrator.

Invoked as a subprocess (``python -m ntt_downstream.decoding.utils.mlflow_parent
<subcommand>``) by :mod:`ntt_downstream.decoding.orchestration`. Three subcommands:

  * ``create``    — open a new parent MLflow run, attach orchestrator-supplied
                    tags, end the run, and emit ``RUN_ID=`` / ``RUN_URI=`` lines.
  * ``aggregate`` — query every child run and log mean / median of every numeric
                    metric back onto the parent, plus child counts.
  * ``log-artifacts`` — archive each subrun's metric deliverables onto the
                    parent run (``downstream_decoding_results.json`` +
                    ``metrics.csv`` + ``calibration_*.csv``), path-keyed by the
                    on-disk base_run(seed)/ft_mode/session/trials tree; plus
                    ``--file`` paths (the benchmark cohort record) at the
                    parent's artifact root. ``run_result.json`` is the sweep's
                    local bookkeeping and is never uploaded: consumers (ntt's
                    aggregation) read only the v2 records.

The *pure* helpers (URI synthesis, tag parsing, metric filtering, mean/median)
live in :mod:`ntt_downstream.decoding.mlflow_utils` and are imported below, so
they're importable in-process by the orchestration engine and unit-testable.
The *stateful* subcommands live here and are invoked as a subprocess, so a flaky
tracking server can't take down the orchestrator loop.

The helper is a soft dependency: when ``MLFLOW_TRACKING_URI`` is unset or
``--disabled`` is passed, ``create`` exits 0 with empty ``RUN_ID=`` /
``RUN_URI=`` so callers degrade gracefully. ``aggregate`` is a no-op when
its ``--run-id`` is empty.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from ntt_downstream.decoding.mlflow_utils import (
    _aggregate_values,
    _build_run_uri,
    _is_aggregable,
    _metrics_table,
    _parse_tag_args,
)

__all__ = [
    "_aggregate_values",
    "_build_run_uri",
    "_is_aggregable",
    "_metrics_table",
    "_parse_tag_args",
    "cmd_aggregate",
    "cmd_create",
    "cmd_export",
    "cmd_log_artifacts",
    "main",
]


def _safe_import_mlflow():
    """Import mlflow only when actually invoked (keeps disabled-path startup fast)."""
    import mlflow  # noqa: PLC0415

    return mlflow


def _tracking_uri() -> str | None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    return uri or None


def cmd_create(args: argparse.Namespace) -> int:
    """Create the parent run; print RUN_ID / RUN_URI; end the run."""
    tracking_uri = _tracking_uri()
    if args.disabled or tracking_uri is None:
        # Soft fail — caller decides whether to proceed without a parent.
        print("RUN_ID=")
        print("RUN_URI=")
        return 0

    try:
        mlflow = _safe_import_mlflow()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        tags = _parse_tag_args(args.tag)
        tags.setdefault("orchestrator.role", "parent")
        run = mlflow.start_run(run_name=args.run_name, tags=tags)
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        mlflow.end_run()
    except Exception as exc:  # noqa: BLE001 - surface to caller, never abort
        print(f"# _mlflow_parent.create failed: {exc}", file=sys.stderr)
        print("RUN_ID=")
        print("RUN_URI=")
        return 0

    uri = _build_run_uri(tracking_uri, experiment_id, run_id)
    print(f"RUN_ID={run_id}")
    print(f"RUN_URI={uri}")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    """Pull child metrics and log mean / median back onto the parent run."""
    if not args.run_id:
        # Soft no-op — matches the create-disabled contract.
        return 0
    tracking_uri = _tracking_uri()
    if tracking_uri is None:
        print(
            "# _mlflow_parent.aggregate: MLFLOW_TRACKING_URI not set; skipping",
            file=sys.stderr,
        )
        return 0

    mlflow = _safe_import_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()  # pyright: ignore[reportPrivateImportUsage]  # mlflow.tracking is a real submodule; stub lacks __all__ entry

    try:
        parent = client.get_run(args.run_id)
    except Exception as exc:  # noqa: BLE001
        print(
            f"# _mlflow_parent.aggregate: parent run {args.run_id} unreachable: {exc}",
            file=sys.stderr,
        )
        return 0

    experiment_id = parent.info.experiment_id

    filter_string = f"tags.`mlflow.parentRunId` = '{args.run_id}'"
    children = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        max_results=5000,
    )

    n_total = len(children)
    n_failed = sum(1 for r in children if r.info.status == "FAILED")
    n_active = sum(1 for r in children if r.info.status == "RUNNING")
    n_finished = sum(1 for r in children if r.info.status == "FINISHED")

    buckets: dict[str, list[float]] = {}
    for r in children:
        for key, val in r.data.metrics.items():
            if not _is_aggregable(key):
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fval):
                continue
            buckets.setdefault(key, []).append(fval)

    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_metric("child_count", float(n_total))
        mlflow.log_metric("child_finished_count", float(n_finished))
        mlflow.log_metric("child_failed_count", float(n_failed))
        mlflow.log_metric("child_active_count", float(n_active))

        for key, values in sorted(buckets.items()):
            if not values:
                continue
            mean, median, n = _aggregate_values(values)
            mlflow.log_metric(f"{key}__mean", mean)
            mlflow.log_metric(f"{key}__median", median)
            mlflow.log_metric(f"{key}__count", float(n))

    print(
        f"# _mlflow_parent.aggregate: parent={args.run_id} "
        f"children={n_total} (finished={n_finished} failed={n_failed} active={n_active}) "
        f"metrics={len(buckets)}",
        file=sys.stderr,
    )
    return 0


def _is_superseded_attempt(path) -> bool:
    """True if ``path`` lives in a ``csv_logs/version_N`` that isn't the newest one.

    Reruns of a cell add a new Lightning version dir; only the latest attempt's
    files are deliverables, an earlier attempt's record would mislead drill-down.
    """
    for parent in path.parents:
        name = parent.name
        if parent.parent.name == "csv_logs" and name.startswith("version_") and name[8:].isdigit():
            siblings = [p.name[8:] for p in parent.parent.iterdir() if p.is_dir() and p.name.startswith("version_")]
            newest = max(int(n) for n in siblings if n.isdigit())
            return int(name[8:]) != newest
    return False


# Per-child files uploaded from each --src tree (plus calibration_*.csv).
_UPLOAD_ALLOWLIST = ("downstream_decoding_results.json", "metrics.csv")


def cmd_log_artifacts(args: argparse.Namespace) -> int:
    """Upload per-child deliverables (``--src`` trees) and root files (``--file``) to the parent run."""
    if not args.run_id:
        # Soft no-op — matches the create-disabled contract.
        return 0
    tracking_uri = _tracking_uri()
    if tracking_uri is None:
        print(
            "# _mlflow_parent.log-artifacts: MLFLOW_TRACKING_URI not set; skipping",
            file=sys.stderr,
        )
        return 0

    import fnmatch
    from pathlib import Path

    mlflow = _safe_import_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()  # pyright: ignore[reportPrivateImportUsage]  # mlflow.tracking is a real submodule; stub lacks __all__ entry

    n_logged = 0
    n_failed = 0
    for src in args.src:
        src_path = Path(src)
        if not src_path.is_dir():
            print(
                f"# log-artifacts: not a directory, skipping: {src}",
                file=sys.stderr,
            )
            continue
        base = src_path.parent
        for f in sorted(src_path.rglob("*")):
            if not f.is_file():
                continue
            # Per-subrun metric deliverables only — never checkpoints/logs, and
            # never run_result.json (local sweep bookkeeping, not a contract):
            #   downstream_decoding_results.json (the v2 per-run record)
            #   metrics.csv       (Lightning's full per-step metrics log)
            #   calibration_*.csv (the per-ft-mode aggregate tables)
            if f.name not in _UPLOAD_ALLOWLIST and not fnmatch.fnmatch(f.name, "calibration_*.csv"):
                continue
            if _is_superseded_attempt(f):
                continue
            # artifact_path mirrors the on-disk tree relative to the src's parent,
            # so the parent's artifact browser reproduces the
            # <base_run(seed)>/<ft_mode>/<session>/ft_trials_NNN/... layout — pull
            # one subrun's metrics, or group the seed-tagged base_run dirs.
            artifact_path = str(f.parent.relative_to(base))
            try:
                client.log_artifact(args.run_id, str(f), artifact_path=artifact_path)
                n_logged += 1
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                if n_failed <= 3:
                    print(f"# log-artifacts: failed for {f}: {exc}", file=sys.stderr)

    # Root-level files (e.g. the benchmark cohort record): artifact_path=None so
    # consumers find them at a fixed location on the parent run.
    for f in getattr(args, "file", None) or []:
        if not Path(f).is_file():
            print(f"# log-artifacts: not a file, skipping: {f}", file=sys.stderr)
            continue
        try:
            client.log_artifact(args.run_id, str(f), artifact_path=None)
            n_logged += 1
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            print(f"# log-artifacts: failed for {f}: {exc}", file=sys.stderr)

    print(
        f"# _mlflow_parent.log-artifacts: parent={args.run_id} logged={n_logged} failed={n_failed}",
        file=sys.stderr,
    )
    return 1 if (n_failed > 0 and n_logged == 0) else 0


def cmd_export(args: argparse.Namespace) -> int:
    """Pull every child run of a parent into a tidy per-subrun metrics table.

    Writes CSV to ``--out`` (or stdout): one row per child with its facet tags
    (seed / ft_mode / calib_session / ft_trials / stage / tokenizer_name / …) and
    last-logged metrics. Group/pivot it (e.g. by ``tag.seed``) to compute any
    aggregate table. Soft no-op when ``--run-id`` is empty or no tracking server
    is configured (matches the create/aggregate contract)."""
    import csv

    if not args.run_id:
        return 0
    tracking_uri = _tracking_uri()
    if tracking_uri is None:
        print("# _mlflow_parent.export-metrics: MLFLOW_TRACKING_URI not set; skipping", file=sys.stderr)
        return 0
    mlflow = _safe_import_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()  # pyright: ignore[reportPrivateImportUsage]  # mlflow.tracking is a real submodule; stub lacks __all__ entry
    try:
        parent = client.get_run(args.run_id)
    except Exception as exc:  # noqa: BLE001
        print(f"# export-metrics: parent run {args.run_id} unreachable: {exc}", file=sys.stderr)
        return 1
    children = client.search_runs(
        experiment_ids=[parent.info.experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{args.run_id}'",
        max_results=50000,
    )
    fieldnames, rows = _metrics_table(children)
    out = open(args.out, "w", newline="") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            out.close()
    print(f"# _mlflow_parent.export-metrics: parent={args.run_id} rows={len(rows)}", file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="open a parent run and end it")
    p_create.add_argument("--experiment-name", required=True)
    p_create.add_argument("--run-name", required=True)
    p_create.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="key=value",
        help="repeatable tag to attach to the parent run",
    )
    p_create.add_argument(
        "--disabled",
        action="store_true",
        help="force the disabled path (RUN_ID=, RUN_URI=)",
    )
    p_create.set_defaults(func=cmd_create)

    p_agg = sub.add_parser("aggregate", help="log mean/median of child metrics onto a parent run")
    p_agg.add_argument(
        "--run-id",
        required=True,
        help="parent run id; pass empty to no-op (matches the disabled path)",
    )
    p_agg.set_defaults(func=cmd_aggregate)

    p_log = sub.add_parser(
        "log-artifacts",
        help="archive per-subrun results records + metrics.csv + calibration_*.csv onto a parent run",
    )
    p_log.add_argument(
        "--run-id",
        required=True,
        help="parent run id; pass empty to no-op (matches the disabled path)",
    )
    p_log.add_argument(
        "--src",
        action="append",
        default=[],
        metavar="DIR",
        help="repeatable experiment-tree root to scan for artifacts",
    )
    p_log.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="PATH",
        help="repeatable file uploaded to the parent's artifact root (e.g. the cohort record)",
    )
    p_log.set_defaults(func=cmd_log_artifacts)

    p_exp = sub.add_parser(
        "export-metrics",
        help="dump a tidy per-subrun (tags + metrics) CSV pulled from a parent run",
    )
    p_exp.add_argument(
        "--run-id",
        required=True,
        help="parent run id; pass empty to no-op (matches the disabled path)",
    )
    p_exp.add_argument("--out", default=None, metavar="CSV", help="output CSV path (default: stdout)")
    p_exp.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
