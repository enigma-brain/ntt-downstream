"""Runtime loader for ``ntt``'s curated selection YAMLs.

The ``ntt`` main repo ships a family of selection configs under
``<ntt>/config/data/selections/ntt_reaching_strict_*.yaml``. Each
declares which brainsets + sessions belong to a specific role (train, tier1
xtime val, tier2 xsession val/test, tier3 xsubject val/test, tier4 xsource
val/test). They are the single source of truth for split memberships and
are regenerated from ``<ntt>/scripts/generate_reaching_splits.py``.

This module reads those YAMLs at pipeline-compose time and rewrites them
into the ``list[{selection: [...], config?: {...}}]`` shape that
``torch_brain.data.Dataset`` consumes. Two consequences:

* downstream pipelines reference curated splits by name
  (``${ntt_split:ntt_reaching_strict_train}``) instead of copying session
  lists into this repo;
* any regeneration in the ``ntt`` repo flows through to the next downstream
  run with no change here.

The resolver is registered as ``${ntt_split:<name>}``. It returns the raw
torch_brain-shaped selection list; downstream callers are free to filter
brainsets (see ``filter_selection_by_brainsets``) before handing it to the
store.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


_ENV_OVERRIDE = "NTT_SPLITS_DIR"


def ntt_selections_dir() -> Path:
    """Return the filesystem path to ``<ntt>/config/data/selections``.

    Resolution order:

    1. ``NTT_SPLITS_DIR`` env var — explicit override, used when ``ntt``
       ships without its source checkout (e.g. pip-installed from a
       built wheel).
    2. The PEP 610 ``direct_url.json`` that uv/pip writes next to the
       installed ``ntt`` dist-info — points at the source repo when ``ntt``
       is installed via ``tool.uv.sources = { path = … }``, even if the
       install isn't editable.
    3. Walk up from ``ntt.__file__`` looking for a sibling ``config/data/
       selections`` directory — covers true editable installs.
    """

    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        path = Path(override)
        if not path.is_dir():
            raise FileNotFoundError(f"{_ENV_OVERRIDE}={path} is not a directory.")
        return path

    direct_url_root = _ntt_source_root_from_direct_url()
    if direct_url_root is not None:
        candidate = direct_url_root / "config" / "data" / "selections"
        if candidate.is_dir():
            return candidate

    import ntt  # type: ignore[import-untyped]

    ntt_pkg = Path(ntt.__file__).resolve().parent
    for parent in ntt_pkg.parents:
        candidate = parent / "config" / "data" / "selections"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate ntt's config/data/selections directory. Set "
        f"{_ENV_OVERRIDE} to override (should point at that directory)."
    )


def _ntt_source_root_from_direct_url() -> Path | None:
    """Return the source repo path recorded in ``ntt``'s direct_url.json, if any.

    PEP 610 mandates ``direct_url.json`` next to the dist-info when a
    package is installed from a local path, VCS ref, or URL. uv writes it
    for ``tool.uv.sources = { path = … }`` installs even when they aren't
    editable, so we can recover the source checkout from there.
    """

    try:
        from importlib.metadata import Distribution
    except ImportError:
        return None
    try:
        dist = Distribution.from_name("ntt")
    except Exception:
        return None
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        raw = None
    if not raw:
        return None
    import json
    from urllib.parse import unquote, urlparse

    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return None
    url = info.get("url")
    if not isinstance(url, str) or not url.startswith("file://"):
        return None
    parsed = urlparse(url)
    path = Path(unquote(parsed.path))
    return path if path.is_dir() else None


def load_ntt_selection(name: str) -> list[dict[str, Any]]:
    """Load one selection YAML and return its torch_brain-shaped payload.

    Input (ntt YAML shape):

        _target_: ntt.core.data.components.config.SelectionConfig
        selection:
          - brainset: odoherty_sabes_nonhuman_2017
            sessions: [indy_20160407_02, ...]
          - brainset: perich_miller_population_2018
            sessions: [...]
        config:
          ignore_splits: true   # optional

    Output (torch_brain.data.Dataset shape):

        [{
          "selection": [
            {"brainset": "odoherty_sabes_nonhuman_2017",
             "sessions": ["indy_20160407_02", ...]},
            {"brainset": "perich_miller_population_2018", "sessions": [...]},
          ],
          "config": {"ignore_splits": true},   # only when set upstream
        }]
    """

    path = ntt_selections_dir() / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown ntt selection {name!r}: {path} not found.")
    raw = OmegaConf.load(path)
    doc = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(doc, dict) or "selection" not in doc:
        raise ValueError(f"{path}: expected a SelectionConfig YAML with a 'selection' list.")
    entry: dict[str, Any] = {"selection": _clean_selection(doc["selection"], path)}
    cfg = doc.get("config") or {}
    if cfg:
        entry["config"] = dict(cfg)
    return [entry]


def _clean_selection(selection: Any, path: Path) -> list[dict[str, Any]]:
    """Drop the ``_target_`` key from each selection entry and stringify IDs."""

    if not isinstance(selection, list) or not selection:
        raise ValueError(f"{path}: selection must be a non-empty list.")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(selection):
        if not isinstance(item, dict) or "brainset" not in item:
            raise ValueError(f"{path}: selection[{i}] missing 'brainset'. Got {item!r}.")
        cleaned = {k: v for k, v in item.items() if not k.startswith("_")}
        # Stringify session / subject IDs defensively — YAML parses all-digit
        # IDs as ints otherwise.
        for key in ("sessions", "subjects", "exclude_sessions"):
            if key in cleaned and cleaned[key] is not None:
                cleaned[key] = [str(v) for v in cleaned[key]]
        if "session" in cleaned and cleaned["session"] is not None:
            cleaned["session"] = str(cleaned["session"])
        if "subject" in cleaned and cleaned["subject"] is not None:
            cleaned["subject"] = str(cleaned["subject"])
        out.append(cleaned)
    return out


def selection_brainsets(selection_list: list[dict[str, Any]]) -> list[str]:
    """Return the sorted unique brainset IDs in a torch_brain selection list."""

    ids: set[str] = set()
    for entry in selection_list:
        for sub in entry.get("selection", []):
            ids.add(str(sub["brainset"]))
    return sorted(ids)


def selection_sessions_by_brainset(
    selection_list: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Return ``{brainset: [sessions]}`` in stable order."""

    out: dict[str, list[str]] = {}
    for entry in selection_list:
        for sub in entry.get("selection", []):
            brainset = str(sub["brainset"])
            out.setdefault(brainset, []).extend(str(s) for s in sub.get("sessions") or [])
    return out


def filter_selection_by_brainsets(
    selection_list: list[dict[str, Any]],
    allow: list[str] | None,
) -> list[dict[str, Any]]:
    """Drop selection entries whose brainset is not in ``allow``.

    ``allow=None`` or empty ⇒ no filter (returns the input unchanged).
    Raises ``ValueError`` if any allow-listed brainset is missing from the
    input entirely — typo-guard so a broken override doesn't silently
    train on zero data.
    """

    if not allow:
        return selection_list

    allow_set = {str(b) for b in allow}
    present = set(selection_brainsets(selection_list))
    missing = allow_set - present
    if missing:
        raise ValueError(
            "restrict_brainsets references brainsets not present in the "
            f"selection: {sorted(missing)}. Available: {sorted(present)}."
        )

    out: list[dict[str, Any]] = []
    for entry in selection_list:
        kept_sub = [sub for sub in entry.get("selection", []) if str(sub["brainset"]) in allow_set]
        if not kept_sub:
            continue
        new_entry = dict(entry)
        new_entry["selection"] = kept_sub
        out.append(new_entry)
    return out


def register_ntt_split_resolver() -> None:
    """Register the ``${ntt_split:<name>}`` OmegaConf resolver.

    The resolver wraps its list output in an :class:`OmegaConf.ListConfig` so
    the resolved value can replace a scalar node (``dataset: ???``) without
    OmegaConf rejecting the new container type. Idempotent.
    """

    if not OmegaConf.has_resolver("ntt_split"):

        def _resolve(name: str):
            return OmegaConf.create(load_ntt_selection(name))

        OmegaConf.register_new_resolver("ntt_split", _resolve, use_cache=True)


__all__ = [
    "filter_selection_by_brainsets",
    "load_ntt_selection",
    "ntt_selections_dir",
    "register_ntt_split_resolver",
    "selection_brainsets",
    "selection_sessions_by_brainset",
]
