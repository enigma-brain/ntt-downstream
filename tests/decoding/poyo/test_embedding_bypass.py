"""Core test: POYOWithEmbeddingBypass bypass path matches original path.

Strategy: instantiate with small dims, initialize unit+session vocabs, build
`inputs = unit_emb(idx) + token_type_emb(type)` manually, and verify that
forward(..., input_embeddings=inputs) == forward(..., input_unit_index=idx,
input_token_type=type). Determinism via eval() + manual_seed.
"""

import torch
from torch_brain.registry import MODALITY_REGISTRY

from ntt_downstream.decoding.backends.poyo import POYOWithEmbeddingBypass


def _make_model(dim: int = 16) -> POYOWithEmbeddingBypass:
    """Build a tiny POYOWithEmbeddingBypass suitable for unit tests."""
    readout_spec = MODALITY_REGISTRY["cursor_velocity_2d"]
    model = POYOWithEmbeddingBypass(
        sequence_length=1.0,
        readout_spec=readout_spec,
        latent_step=0.5,
        num_latents_per_step=4,
        dim=dim,
        depth=1,
        dim_head=8,
        cross_heads=1,
        self_heads=2,
        ffn_dropout=0.0,
        lin_dropout=0.0,
        atn_dropout=0.0,
    )
    # Initialize infinite vocabs.
    model.unit_emb.initialize_vocab([f"u{i}" for i in range(5)])
    model.session_emb.initialize_vocab(["sess0"])
    model.eval()
    return model


def test_bypass_matches_original_path():
    """Feeding the explicit sum as input_embeddings reproduces the original path."""
    torch.manual_seed(0)
    dim = 16
    model = _make_model(dim=dim)

    B, N_in, N_lat, N_out = 2, 7, 4, 3
    input_unit_index = torch.randint(0, 5, (B, N_in))
    input_token_type = torch.randint(0, 4, (B, N_in))
    input_timestamps = torch.rand(B, N_in)
    input_mask = torch.ones(B, N_in, dtype=torch.bool)

    latent_index = torch.arange(N_lat).unsqueeze(0).expand(B, -1).contiguous()
    latent_timestamps = torch.linspace(0, 1, N_lat).unsqueeze(0).expand(B, -1).contiguous()

    output_session_index = torch.zeros(B, N_out, dtype=torch.long)
    output_timestamps = torch.rand(B, N_out)

    with torch.no_grad():
        out_original = model(
            input_unit_index=input_unit_index,
            input_timestamps=input_timestamps,
            input_token_type=input_token_type,
            input_mask=input_mask,
            latent_index=latent_index,
            latent_timestamps=latent_timestamps,
            output_session_index=output_session_index,
            output_timestamps=output_timestamps,
        )

        # Build the same content the bypass would skip.
        pre_emb = model.unit_emb(input_unit_index) + model.token_type_emb(input_token_type)

        out_bypass = model(
            input_timestamps=input_timestamps,
            input_mask=input_mask,
            input_embeddings=pre_emb,
            latent_index=latent_index,
            latent_timestamps=latent_timestamps,
            output_session_index=output_session_index,
            output_timestamps=output_timestamps,
        )

    assert out_original.shape == out_bypass.shape == (B, N_out, 2)
    assert torch.allclose(out_original, out_bypass, atol=1e-6), (
        f"max abs diff = {(out_original - out_bypass).abs().max().item():.3e}"
    )


def test_bypass_rejects_wrong_dim():
    """Wrong last-dim in input_embeddings raises ValueError."""
    model = _make_model(dim=16)
    B, N_in = 1, 3
    bad = torch.zeros(B, N_in, 8)  # wrong dim
    try:
        model(
            input_timestamps=torch.zeros(B, N_in),
            input_embeddings=bad,
            latent_index=torch.zeros(B, 2, dtype=torch.long),
            latent_timestamps=torch.zeros(B, 2),
            output_session_index=torch.zeros(B, 1, dtype=torch.long),
            output_timestamps=torch.zeros(B, 1),
        )
    except ValueError as e:
        assert "does not match model dim" in str(e)
    else:
        raise AssertionError("expected ValueError for mismatched input_embeddings dim")
