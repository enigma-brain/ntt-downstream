"""Run-provenance helpers for results files."""

from __future__ import annotations

import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def git_sha(cwd: str | None = None) -> str | None:
    """Best-effort current commit sha; ``None`` if not a git repo / git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip() or None


def now_iso() -> str:
    """Current UTC time as ISO-8601 with trailing ``Z``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def hostname() -> str:
    """Machine hostname, or ``"unknown"`` on failure."""
    try:
        return socket.gethostname() or "unknown"
    except OSError:
        return "unknown"


def _ntt_commit(*, direct_url: dict | None, package_dir: Path) -> str | None:
    """The ntt commit: from a VCS install's ``direct_url.json``, else the checkout's HEAD.

    An editable checkout (ntt's launcher overlay) reports the same package
    version for every commit, so only the commit tells two of them apart.
    """
    commit = ((direct_url or {}).get("vcs_info") or {}).get("commit_id")
    if commit:
        return str(commit)
    if not Path(package_dir).is_dir():
        return None
    return git_sha(cwd=str(package_dir))


def ntt_provenance() -> dict[str, str | None]:
    """Which ``ntt`` this process imports: version, resolved source path and commit.

    Recorded in every results record because a downstream run can load either
    the lock's pinned ntt tag or a parent checkout overlaid by ntt's launcher,
    and the two can disagree on checkpoint formats. ``ntt_path``
    says which kind it was; ``ntt_commit`` says which code. Best-effort:
    ``None`` values when ntt is not importable.
    """
    try:
        import importlib.metadata
        import json

        import ntt

        dist = importlib.metadata.distribution("ntt")
        raw = dist.read_text("direct_url.json")
        package_dir = Path(ntt.__file__).resolve().parent
        return {
            "ntt_version": dist.version,
            "ntt_path": str(Path(ntt.__file__).resolve()),
            "ntt_commit": _ntt_commit(direct_url=json.loads(raw) if raw else None, package_dir=package_dir),
        }
    except Exception:  # noqa: BLE001
        return {"ntt_version": None, "ntt_path": None, "ntt_commit": None}
