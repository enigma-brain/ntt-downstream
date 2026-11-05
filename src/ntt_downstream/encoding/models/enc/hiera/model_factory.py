"""Factory functions for creating image encoding models."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import autocast, nn

from .modded_hiera import Hiera

logger = logging.getLogger(__name__)


def _to_native_list(v: Any) -> Any:
    """Convert OmegaConf ListConfig to a native Python list, or return None/list as-is."""
    if v is None:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(v):
            return OmegaConf.to_container(v, resolve=True)
    except ImportError:
        pass
    return list(v) if not isinstance(v, list) else v


def create_backbone(
    cfg,
    h: int,
    w: int,
    dtype: torch.dtype,
    pretrained: str | None = None,
    pretrained_kwargs: dict | None = None,
    freeze: bool = False,
) -> tuple[nn.Module, int, tuple]:
    """Create and initialize the Hiera backbone model.

    Args:
        cfg: Configuration object.
        h: Height of input images.
        w: Width of input images.
        dtype: Data type for the model.
        pretrained: Name of pre-trained Hiera model (e.g., "hiera_base_224").
        pretrained_kwargs: Additional kwargs for pretrained model.
        freeze: Whether to freeze backbone weights.

    Returns:
        (backbone, backbone_dim, input_size)
    """
    import ntt_downstream.encoding.models.enc.hiera.modded_hiera as modded_hiera_module

    if pretrained is not None:
        if not pretrained.startswith("hiera_"):
            raise ValueError(f"Unknown pretrained model '{pretrained}'. Must start with 'hiera_'.")
        available = [fn for fn in dir(modded_hiera_module) if fn.startswith("hiera_")]
        if pretrained not in available:
            raise ValueError(f"Unknown pretrained model '{pretrained}'. Available: {available}")
        fn = getattr(modded_hiera_module, pretrained)
        pretrained_kwargs = pretrained_kwargs or {}
        backbone = fn(pretrained=True, **pretrained_kwargs)
        input_size = backbone.input_size if hasattr(backbone, "input_size") else (224, 224)
    else:
        # Train Hiera from scratch with native image resolution
        # Convert OmegaConf containers to native Python types for Hiera
        def _to_native(v: Any) -> Any:
            """Convert OmegaConf ListConfig/DictConfig to native Python."""
            try:
                from omegaconf import OmegaConf

                if OmegaConf.is_config(v):
                    return OmegaConf.to_container(v, resolve=True)
            except ImportError:
                pass
            return v

        backbone = Hiera(
            input_size=(h, w),
            num_heads=int(cfg.model.get("hiera_num_heads", 1)),  # initial heads; doubles per stage
            embed_dim=int(cfg.model.get("hiera_embed_dim", 96)),
            stages=_to_native(cfg.model.get("hiera_stages", [2, 3, 16, 3])),
            q_pool=int(cfg.model.get("hiera_q_pool", 3)),
            in_chans=int(cfg.model.get("hiera_in_chans", 3)),
            q_stride=tuple(_to_native(cfg.model.get("hiera_q_stride", [2, 2]))),
            mask_unit_size=_to_native(cfg.model.get("hiera_mask_unit_size", [8, 8])),
            patch_kernel=tuple(_to_native(cfg.model.get("hiera_patch_kernel", [7, 7]))),
            patch_stride=tuple(_to_native(cfg.model.get("hiera_patch_stride", [4, 4]))),
            patch_padding=tuple(_to_native(cfg.model.get("hiera_patch_padding", [3, 3]))),
            sep_pos_embed=bool(cfg.model.get("hiera_sep_pos_embed", False)),
            drop_path_rate=float(cfg.model.get("hiera_drop_path_rate", 0.0)),
            mlp_ratio=float(cfg.model.get("hiera_mlp_ratio", 4.0)),
        )
        input_size = (h, w)

    backbone = backbone.cuda().to(dtype)

    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()

    # Determine backbone output dimension
    example_input = torch.ones(1, cfg.model.get("hiera_in_chans", 3), *input_size).to("cuda", dtype)
    with torch.no_grad():
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.trainer.get("amp", True)):
            out = backbone(example_input)
            backbone_dim = out.shape[-1]

    return backbone, backbone_dim, input_size


def create_model(
    cfg,
    session_info: dict[str, int],
    model_type: str = "image_encoding",
    compile_model: bool | None = None,
    device: str = "cuda",
    pretrained: str | None = None,
    pretrained_kwargs: dict | None = None,
    n_neurons_override: dict[str, int] | None = None,
    subject_info: dict[str, str] | None = None,
    **kwargs,
) -> Any:
    """Factory function to create and configure image encoding models.

    Args:
        cfg: Configuration object.
        session_info: dict mapping session_id → n_neurons.
        model_type: "image_encoding", "image_encoding_mar", or "conv_core_readout".
        compile_model: Whether to compile the model.
        device: Device to place model on.
        pretrained: Pretrained backbone name.
        pretrained_kwargs: Additional kwargs for pretrained backbone.
        n_neurons_override: Override n_neurons per session (for compress_n).

    Returns:
        Configured model.
    """
    from .base_model import ImageEncodingHiera
    from .mar_model import ImageEncodingHieraMAR

    h, w = cfg.model.get("image_size", [112, 112])
    dtype = _get_torch_dtype(cfg.trainer.get("dtype", "bfloat16"))

    # Auto-compute n_time_bins from data config if not explicitly set in model config
    n_time_bins = cfg.model.get("n_time_bins", None)
    if n_time_bins is None and hasattr(cfg, "data"):
        response_window = float(cfg.data.get("response_window", 0.200))
        bin_size = float(cfg.data.get("bin_size", 0.001))
        n_time_bins = int(round(response_window / bin_size))
    n_time_bins = int(n_time_bins) if n_time_bins is not None else 200

    # --- Conv Core + Readout baseline (no Hiera backbone) ---
    if model_type == "conv_core_readout":
        from ..conv.model import ConvCoreReadout

        def _to_list(v: Any) -> Any:
            try:
                from omegaconf import OmegaConf

                if OmegaConf.is_config(v):
                    return OmegaConf.to_container(v, resolve=True)
            except ImportError:
                pass
            return list(v) if not isinstance(v, list) else v

        loss_mode = str(cfg.model.get("loss_mode", "poisson"))
        model = ConvCoreReadout(
            session_info=session_info,
            in_channels=int(cfg.model.get("hiera_in_chans", 3)),
            image_size=(h, w),
            hidden_channels=_to_list(cfg.model.get("hidden_channels", [32])),
            spatial_kernel_sizes=_to_list(cfg.model.get("spatial_kernel_sizes", [13])),
            n_time_bins=n_time_bins,
            loss_mode=loss_mode,
            gamma_input=float(cfg.model.get("gamma_input", 0.5)),
            gamma_hidden=float(cfg.model.get("gamma_hidden", 0.0)),
            input_padding=cfg.model.get("input_padding", 0),
            hidden_padding=cfg.model.get("hidden_padding", "same"),
            core_dropout=float(cfg.model.get("core_dropout", 0.0)),
            readout_scale=bool(cfg.model.get("readout_scale", True)),
            readout_bias=bool(cfg.model.get("readout_bias", True)),
            readout_gaussian_mean_scale=float(cfg.model.get("readout_gaussian_mean_scale", 6.0)),
            readout_gaussian_var_scale=float(cfg.model.get("readout_gaussian_var_scale", 4.0)),
            readout_positive=bool(cfg.model.get("readout_positive", False)),
            readout_mask_l1_reg=float(cfg.model.get("readout_mask_l1_reg", 0.0)),
            readout_feature_l1_reg=float(cfg.model.get("readout_feature_l1_reg", 0.0)),
        )
        model = model.to(dtype).to(device)
        return model

    pretrained = pretrained or cfg.model.get("pretrained_backbone")
    pretrained_kwargs = pretrained_kwargs or cfg.model.get("pretrained_backbone_kwargs", {})
    freeze_backbone = bool(cfg.model.get("freeze_backbone", False))

    backbone, backbone_dim, input_size = create_backbone(
        cfg, h, w, dtype, pretrained=pretrained, pretrained_kwargs=pretrained_kwargs, freeze=freeze_backbone
    )

    loss_mode = str(cfg.model.get("loss_mode", "poisson"))

    from ..losses.flow_utils import FlowOptions

    # Flow-matching training options (only consumed when loss_mode='flow_matching').
    flow_opts = FlowOptions(
        time_sampling=str(cfg.model.get("flow_time_sampling", "uniform")),
        logit_normal_mean=float(cfg.model.get("flow_logit_normal_mean", 0.0)),
        logit_normal_std=float(cfg.model.get("flow_logit_normal_std", 1.0)),
        ot_coupling=bool(cfg.model.get("flow_ot_coupling", False)),
        cond_dropout=float(cfg.model.get("flow_cond_dropout", 0.0)),
    )

    if model_type == "image_encoding":
        tokenizer_wrapper = kwargs.get("tokenizer_wrapper") if loss_mode not in ("poisson", "mse_rate") else None

        # DiffLoss for diffusion mode
        diffloss = None
        diffusion_cfg = cfg.model.get("diffusion")
        if diffusion_cfg is not None and loss_mode == "diffusion":
            from ..losses.diffloss import DiffLoss

            diff_target_ch = int(diffusion_cfg.target_channels)
            diff_z_ch = int(diffusion_cfg.z_channels)
            diffloss = DiffLoss(
                target_channels=diff_target_ch,
                z_channels=diff_z_ch,
                depth=int(diffusion_cfg.depth),
                width=int(diffusion_cfg.width),
                num_sampling_steps=str(diffusion_cfg.num_sampling_steps),
            )

        # EnergyLoss (EAR) for energy-score mode
        energy_loss = None
        energy_cfg = cfg.model.get("energy")
        if loss_mode == "energy":
            if energy_cfg is None:
                raise ValueError(
                    "loss_mode='energy' requires a 'model.energy:' config block "
                    "with (optional) target_channels, z_channels, width, depth, alpha."
                )
            from ..losses.energy_loss import EnergyLoss

            # Auto-derive target_channels / z_channels from the tokenizer if the
            # config leaves them null. Avoids dimension mismatches when the
            # tokenizer checkpoint is swapped and the config is not updated.
            cfg_target_ch = energy_cfg.get("target_channels", None)
            cfg_z_ch = energy_cfg.get("z_channels", None)
            if cfg_target_ch is None or cfg_z_ch is None:
                if tokenizer_wrapper is None:
                    raise ValueError(
                        "loss_mode='energy' with null target_channels/z_channels "
                        "requires a tokenizer_wrapper to auto-derive the latent dim."
                    )
                derived = int(tokenizer_wrapper.latent_dim)
                cfg_target_ch = derived if cfg_target_ch is None else cfg_target_ch
                cfg_z_ch = derived if cfg_z_ch is None else cfg_z_ch
            energy_target_ch = int(cfg_target_ch)
            energy_z_ch = int(cfg_z_ch)
            energy_loss = EnergyLoss(
                target_channels=energy_target_ch,
                z_channels=energy_z_ch,
                width=int(energy_cfg.get("width", 256)),
                depth=int(energy_cfg.get("depth", 3)),
                noise_channels=int(energy_cfg.get("noise_channels", 64)),
                alpha=float(energy_cfg.get("alpha", 1.5)),
                train_temperature=float(energy_cfg.get("train_temperature", 1.0)),
                grad_checkpointing=bool(energy_cfg.get("grad_checkpointing", False)),
            )

        model = ImageEncodingHiera(
            backbone=backbone,
            session_info=session_info,
            n_time_bins=n_time_bins,
            dim=backbone_dim,
            backbone_dim=backbone_dim,
            dim_head=cfg.model.get("dim_head", 32),
            num_heads=cfg.model.get("num_heads", 4),
            mlp_ratio=cfg.model.get("readout_mlp_ratio", 4),
            learnable_softplus=bool(cfg.model.get("learnable_softplus", False)),
            softplus_beta=float(cfg.model.get("softplus_beta", 0.1)),
            use_session_embedding=cfg.model.get("readout_use_session_embedding", False),
            use_subject_embedding=cfg.model.get("readout_use_subject_embedding", False),
            subject_info=subject_info,
            use_gated_ffn=cfg.model.get("readout_use_gated_ffn", False),
            token_prediction_dim=(
                cfg.model.get("token_prediction_dim", None)
                or (diff_target_ch if diffloss is not None else None)
                or (energy_target_ch if energy_loss is not None else None)
            ),
            diffloss=diffloss,
            energy_loss=energy_loss,
            n_neurons_override=n_neurons_override,
            compress_n_latent_n=cfg.model.get("compress_n_latent_n", None),
            compress_n_t_coarse=cfg.model.get("compress_n_t_coarse", None),
            compress_n_t_target=cfg.model.get("compress_n_t_target", None),
            compress_n_temporal_upsample=str(cfg.model.get("compress_n_temporal_upsample", "interpolate")),
            compress_n_target_slice=cfg.model.get("compress_n_target_slice", None),
            use_ldns_denoiser=bool(cfg.model.get("use_ldns_denoiser", False)),
            ldns_hidden_channels=int(cfg.model.get("ldns_hidden_channels", 64)),
            ldns_num_blocks=int(cfg.model.get("ldns_num_blocks", 4)),
            ldns_kernel_size=int(cfg.model.get("ldns_kernel_size", 15)),
            use_full_grid_denoiser=bool(cfg.model.get("use_full_grid_denoiser", False)),
            full_grid_hidden_channels=int(cfg.model.get("full_grid_hidden_channels", 64)),
            full_grid_num_blocks=int(cfg.model.get("full_grid_num_blocks", 4)),
            full_grid_num_heads=int(cfg.model.get("full_grid_num_heads", 4)),
            full_grid_mlp_ratio=int(cfg.model.get("full_grid_mlp_ratio", 2)),
            full_grid_kernel_size=int(cfg.model.get("full_grid_kernel_size", 7)),
            full_grid_use_s4=bool(cfg.model.get("full_grid_use_s4", False)),
            loss_mode=loss_mode,
            tokenizer_wrapper=tokenizer_wrapper,
            diffusion_batch_mul=int(cfg.model.get("diffusion_batch_mul", 4)),
            flow_head_width=int(cfg.model.get("flow_head_width", 256)),
            flow_head_depth=int(cfg.model.get("flow_head_depth", 3)),
            flow_opts=flow_opts,
            ce_vocab_size=cfg.model.get("ce_vocab_size", None),
            ce_sample_temperature=float(cfg.model.get("ce_sample_temperature", 1.0)),
            # Class-imbalance knobs (default OFF; see base_model __init__).
            ce_focal_gamma=float(cfg.model.get("ce_focal_gamma", 0.0)),
            ce_class_weight_beta=float(cfg.model.get("ce_class_weight_beta", 0.5)),
            num_codebooks=int(cfg.model.get("num_codebooks", 1)),
            response_zscore=bool(cfg.model.get("response_zscore", False)),
            tokenizer_aux_loss_weight=float(cfg.trainer.get("tokenizer_aux_loss_weight", 0.0)),
            decoder_recon_loss_weight=float(cfg.trainer.get("decoder_recon_loss_weight", 0.0)),
            decoder_recon_loss_mode=str(cfg.trainer.get("decoder_recon_loss_mode", "poisson")),
            mse_center_weight=float(cfg.model.get("mse_center_weight", 0.0)),
            cosine_sim_weight=float(cfg.model.get("cosine_sim_weight", 0.0)),
            token_t_latent=int(cfg.model.get("token_t_latent", 1)),
            readout_layer_types=_to_native_list(cfg.model.get("readout_layer_types", None)),
            deep_token_head=bool(cfg.model.get("deep_token_head", True)),
            deep_token_head_depth=int(cfg.model.get("deep_token_head_depth", 2)),
            deep_token_head_width_mult=float(cfg.model.get("deep_token_head_width_mult", 2.0)),
            deep_token_head_use_film=bool(cfg.model.get("deep_token_head_use_film", True)),
            deep_token_head_dropout=float(cfg.model.get("deep_token_head_dropout", 0.0)),
        )

    elif model_type == "image_encoding_mar":
        tokenizer_wrapper = kwargs.get("tokenizer_wrapper")
        if tokenizer_wrapper is None and loss_mode not in ("poisson", "mse_rate"):
            raise ValueError("MAR model requires tokenizer_wrapper to be passed via kwargs")

        # Energy loss (EAR) defaults for MAR. Same config path as non-MAR:
        # cfg.model.energy: {...}. MAR uses its own token_dim (derived from
        # tokenizer) and mar_dim for conditioning, so target_channels /
        # z_channels from the energy block are NOT used here.
        _energy_cfg = cfg.model.get("energy") or {}

        model = ImageEncodingHieraMAR(
            backbone=backbone,
            tokenizer_wrapper=tokenizer_wrapper,
            session_info=session_info,
            dim=backbone_dim,
            backbone_dim=backbone_dim,
            mar_dim=int(cfg.model.get("mar_dim", 256)),
            mar_depth=int(cfg.model.get("mar_depth", 3)),
            mar_num_heads=int(cfg.model.get("mar_num_heads", 8)),
            mar_mlp_ratio=float(cfg.model.get("mar_mlp_ratio", 4)),
            mar_attn_dropout=float(cfg.model.get("mar_attn_dropout", 0.0)),
            mar_proj_dropout=float(cfg.model.get("mar_proj_dropout", 0.0)),
            flow_head_width=int(cfg.model.get("flow_head_width", 256)),
            flow_head_depth=int(cfg.model.get("flow_head_depth", 3)),
            flow_opts=flow_opts,
            mask_ratio_min=float(cfg.model.get("mask_ratio_min", 0.7)),
            mask_ratio_std=float(cfg.model.get("mask_ratio_std", 0.25)),
            use_diffusion_pos_embed=bool(cfg.model.get("use_diffusion_pos_embed", True)),
            num_buffer_tokens=int(cfg.model.get("num_buffer_tokens", 8)),
            unmask_schedule=str(cfg.model.get("unmask_schedule", "cosine")),
            diffusion_batch_mul=int(cfg.model.get("diffusion_batch_mul", 4)),
            loss_mode=loss_mode,
            num_sampling_steps=str(cfg.model.get("num_sampling_steps", "100")),
            ce_vocab_size=cfg.model.get("ce_vocab_size", None),
            ce_sample_temperature=cfg.model.get("ce_sample_temperature", 1.0),
            use_subject_embedding=cfg.model.get("readout_use_subject_embedding", False),
            subject_info=subject_info,
            n_neurons_override=n_neurons_override,
            energy_head_width=int(_energy_cfg.get("width", 256)),
            energy_head_depth=int(_energy_cfg.get("depth", 3)),
            energy_noise_channels=int(_energy_cfg.get("noise_channels", 64)),
            energy_alpha=float(_energy_cfg.get("alpha", 1.5)),
            energy_train_temperature=float(_energy_cfg.get("train_temperature", 1.0)),
            energy_grad_checkpointing=bool(_energy_cfg.get("grad_checkpointing", False)),
            n_temporal_tokens=int(cfg.model.get("n_temporal_tokens", 1)),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from ['image_encoding', 'image_encoding_mar']")

    model = model.to(dtype).to(device)

    should_compile = compile_model if compile_model is not None else cfg.trainer.get("compile", False)
    if should_compile:
        if hasattr(model, "flow_head") and model.flow_head is not None:
            logger.info("Compiling FlowMatchingHead (internals only)")
            model.flow_head.input_proj = torch.compile(model.flow_head.input_proj)  # pyright: ignore[reportAttributeAccessIssue] -- torch.compile returns a wrapped callable, not nn.Linear
        # Compile the backbone only — the top-level forward dispatches into
        # nn.ModuleDict by session-key (a Python str), which dynamo specializes
        # per value and recompiles past recompile_limit, falling back to eager.
        # The backbone is ~95% of FLOPs and has no key dispatch, so this
        # captures the speedup without the recompile thrash.
        logger.info("Using torch.compile (backbone only)")
        model.backbone = torch.compile(model.backbone)

    return model


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype_str, torch.bfloat16)
