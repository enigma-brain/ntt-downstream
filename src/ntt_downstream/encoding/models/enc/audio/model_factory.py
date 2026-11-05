"""Factory for creating audio encoding models from Hydra config."""

from __future__ import annotations

from omegaconf import DictConfig

from ntt_downstream.encoding.models.enc.audio.base_model import AudioEncodingModel


def create_audio_model(
    cfg: DictConfig,
    session_info: dict[str, int],
    subject_info: dict[str, str] | None = None,
    token_prediction_dim: int | None = None,
    n_neurons_override: dict[str, int] | None = None,
    n_tokens_per_neuron: int | None = None,
) -> AudioEncodingModel:
    """Create an AudioEncodingModel from Hydra config."""
    mcfg = cfg.model

    effective_session_info = dict(session_info)
    if n_neurons_override is not None:
        effective_session_info = dict(n_neurons_override)

    if token_prediction_dim is None:
        token_prediction_dim = mcfg.get("token_prediction_dim", None)
        if token_prediction_dim is not None:
            token_prediction_dim = int(token_prediction_dim)

    # conv_dilations determines the layer count; without it, n_conv_layers
    # (default 4) sets the depth with dilations 1, 2, 4, ...
    conv_dilations = mcfg.get("conv_dilations", None)
    if conv_dilations is not None:
        conv_dilations = [int(d) for d in conv_dilations]
    else:
        n_layers = int(mcfg.get("n_conv_layers", 4))
        conv_dilations = [2**i for i in range(n_layers)]

    return AudioEncodingModel(
        session_info=effective_session_info,
        n_freq_channels=int(cfg.data.get("n_freq_channels", 32)),
        conv_rank=int(mcfg.get("conv_rank", 48)),
        conv_kernel_size=int(mcfg.get("conv_kernel_size", 7)),
        conv_dilations=conv_dilations,
        window_size=int(mcfg.get("window_size", 10)),
        dim_q=int(mcfg.get("dim_head", 48)) * int(mcfg.get("num_heads", 4)),
        num_heads=int(mcfg.get("num_heads", 4)),
        n_self_attn_layers=int(mcfg.get("n_self_attn_layers", 2)),
        mlp_ratio=int(mcfg.get("readout_mlp_ratio", 2)),
        loss_mode=str(mcfg.get("loss_mode", "poisson")),
        use_subject_embedding=bool(mcfg.get("readout_use_subject_embedding", False)),
        subject_info=subject_info,
        learnable_softplus=bool(mcfg.get("learnable_softplus", False)),
        softplus_beta=float(mcfg.get("softplus_beta", 1.0)),
        token_prediction_dim=token_prediction_dim,
        n_tokens_per_neuron=n_tokens_per_neuron,
        dropout=float(mcfg.get("dropout", 0.0)),
        rope_max_seq_len=int(mcfg.get("rope_max_seq_len", 512)),
        # Tokenizer-loss configuration
        diffusion_cfg=dict(mcfg.get("diffusion", {})) or None,
        flow_cfg=dict(mcfg.get("flow", {})) or None,
        energy_cfg=dict(mcfg.get("energy", {})) or None,
        ce_vocab_size=mcfg.get("ce_vocab_size", None),
        diffusion_batch_mul=int(mcfg.get("diffusion_batch_mul", 1)),
        tokenizer_aux_loss_weight=float(mcfg.get("tokenizer_aux_loss_weight", 0.0)),
        deep_token_head=bool(mcfg.get("deep_token_head", True)),
        deep_token_head_depth=int(mcfg.get("deep_token_head_depth", 2)),
        deep_token_head_width_mult=float(mcfg.get("deep_token_head_width_mult", 2.0)),
        deep_token_head_use_film=bool(mcfg.get("deep_token_head_use_film", True)),
    )
