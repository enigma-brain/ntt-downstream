"""Helper utilities for the decoding orchestration engine.

* :mod:`~ntt_downstream.decoding.utils.mlflow_parent` — the *stateful* MLflow
  parent-run lifecycle (create / aggregate / log-artifacts), invokable as
  ``python -m ntt_downstream.decoding.utils.mlflow_parent``. The orchestrator
  runs it as a subprocess so a flaky tracking server can't take down the
  loop; its pure helpers live in :mod:`ntt_downstream.decoding.mlflow_utils`.
"""
