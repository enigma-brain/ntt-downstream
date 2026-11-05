"""Execute or preview a resolved Hydra launch for a single-run verb."""

from __future__ import annotations

import shlex
import subprocess


def execute(command: list[str], *, dry_run: bool, print_resolved: bool) -> int:
    """Run, preview, or compose-only a Hydra launch command.

    ``print_resolved`` asks Hydra to print the composed job config and exit
    (``--cfg job``) without running training. ``dry_run`` only echoes the exact
    command that would run and never subprocesses. Otherwise the command is run
    to completion and its return code is propagated.
    """
    if print_resolved:
        # Hydra prints the composed config and exits without running.
        return subprocess.run([*command, "--cfg", "job"], check=False).returncode
    if dry_run:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, check=False).returncode
