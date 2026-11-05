import json
import os
import re
from pathlib import Path

from ntt_downstream.common.results.provenance import _ntt_commit, git_sha, hostname, now_iso, ntt_provenance


def test_now_iso_format():
    ts = now_iso()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts)


def test_hostname_nonempty():
    assert isinstance(hostname(), str)
    assert hostname()


def test_git_sha_in_repo():
    # Running inside the repo worktree: a 40-char sha (or None if git absent).
    sha = git_sha()
    assert sha is None or re.fullmatch(r"[0-9a-f]{7,40}", sha)


def test_git_sha_outside_repo(tmp_path):
    assert git_sha(cwd=str(tmp_path)) is None


def test_ntt_provenance_reports_installed_ntt():
    import importlib.metadata

    import ntt

    prov = ntt_provenance()
    assert prov["ntt_path"] == str(Path(ntt.__file__).resolve())
    assert prov["ntt_version"] == importlib.metadata.version("ntt")


def test_ntt_provenance_reports_commit_of_git_install():
    """The installed ntt comes from a git tag: its commit is in direct_url.json."""
    import importlib.metadata

    direct = json.loads(importlib.metadata.distribution("ntt").read_text("direct_url.json") or "{}")
    expected = (direct.get("vcs_info") or {}).get("commit_id")

    assert ntt_provenance()["ntt_commit"] == expected


def test_ntt_commit_of_editable_checkout_comes_from_git(tmp_path):
    """The embedded case: an editable checkout reports the same version for
    every commit, so the commit must come from the checkout itself."""
    import subprocess

    pkg = tmp_path / "src" / "fakentt"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "x"], check=True, env={**os.environ, **env})
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    assert _ntt_commit(direct_url=None, package_dir=pkg) == head
    assert _ntt_commit(direct_url={"vcs_info": {"commit_id": "abc"}}, package_dir=pkg) == "abc"
    assert _ntt_commit(direct_url=None, package_dir=tmp_path / "nowhere") is None
