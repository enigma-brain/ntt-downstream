"""Guard tests for the sweep entrypoint configs.

These pin the contract between configs/cli/{calibrate,benchmark}.yaml and the
orchestration engine. The YAMLs are flat config data (no Hydra defaults list)
whose keys are the engine's argparse dests, loaded directly via OmegaConf — a
`null` value still counts as the key being present, which is the intended
semantics.
"""

from omegaconf import OmegaConf

from ntt_downstream.decoding import orchestration as orch


def _parser_defaults(verb: str):
    """The engine's own argparse defaults for ``verb`` — the source of truth."""
    return orch._build_parser().parse_args([verb])


def test_calibrate_config_has_required_fields():
    cfg = OmegaConf.load("configs/cli/calibrate.yaml")
    for k in ("seeds", "tokenizer_kind", "ckpt", "gpus"):
        assert k in cfg, k


def test_benchmark_config_has_required_fields():
    cfg = OmegaConf.load("configs/cli/benchmark.yaml")
    for k in ("tokenizers", "sessions", "seed"):
        assert k in cfg, k


def test_calibrate_engine_defaults_match_orchestration():
    """calibrate.yaml pins the engine knobs that don't live in the decode YAMLs.

    Compare against the LIVE parser defaults rather than hardcoded literals, so
    an engine-side default bump that leaves the YAML stale fails here. The
    engine carries its bool defaults as 'true'/'false' strings and parses them
    with _parse_bool; normalize through the same helper.
    """
    cfg = OmegaConf.load("configs/cli/calibrate.yaml")
    ns = _parser_defaults("calibrate")
    assert cfg.jobs_per_gpu == ns.jobs_per_gpu
    assert cfg.multi_session == ns.multi_session
    assert cfg.hydra_data_config == ns.hydra_data_config
    for knob in ("missing_only", "reuse_base_if_exists", "skip_base_train"):
        assert getattr(cfg, knob) == orch._parse_bool(getattr(ns, knob), knob), knob


def test_sweep_configs_pin_repo_root_null():
    """Both sweep YAMLs must leave repo_root null so the launcher's
    _overlay_common can anchor it to the discovered project root (so runs are
    independent of the working directory). A hardcoded value here would silently
    break it."""
    for verb in ("calibrate", "benchmark"):
        cfg = OmegaConf.load(f"configs/cli/{verb}.yaml")
        assert "repo_root" in cfg, verb
        assert cfg.repo_root is None, verb


def test_benchmark_engine_defaults_match_orchestration():
    """benchmark.yaml mirrors run_benchmark's literal defaults.

    The parser-owned defaults are compared live (see calibrate test above). The
    rest are fallbacks applied INLINE in run_benchmark's body (e.g.
    ``args.epochs if args.epochs is not None else 400``) and are not
    introspectable, so for those this test is the literal cross-check copy.
    """
    cfg = OmegaConf.load("configs/cli/benchmark.yaml")
    ns = _parser_defaults("benchmark")
    # Parser-owned defaults — compared live:
    assert cfg.data_config == ns.data_config
    assert cfg.tokenizers_limit == ns.tokenizers_limit
    assert cfg.include_spike_times == orch._parse_bool(ns.include_spike_times, "include_spike_times")
    assert cfg.spike_times_only == orch._parse_bool(ns.spike_times_only, "spike_times_only")
    # run_benchmark inline fallbacks — literals by necessity:
    assert cfg.seed == 42
    assert cfg.epochs == 400
    assert cfg.batch_size == 128
    assert cfg.num_workers == 8
    assert cfg.unit_embedding_mode == "add"
