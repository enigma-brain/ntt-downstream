"""Thin Hydra launcher package for the eval CLI (see app.py).

Intentionally empty: the console scripts target ``ntt_downstream.cli.app:*_cli``
directly, and eagerly importing ``app`` here would make
``python -m ntt_downstream.cli.app`` emit a spurious RuntimeWarning
("found in sys.modules after import of package").
"""
