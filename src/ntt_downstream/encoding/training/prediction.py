"""Shared loss-mode -> rate-prediction dispatch for encoding eval.

This is the single source of truth for turning a model forward pass into an
aligned ``(pred_btn, target_btn)`` pair, used by BOTH the periodic Lightning
validation (``encoding_module._predict_and_align``) and the end-of-training
``final_eval``. Sharing one dispatch keeps the checkpoint-selection metric and
the reported metric on the same time support: the boundary slice is applied
uniformly to every mode, including the CE-Q1 closed-form path (which would
otherwise skip it for LDNS or wavelet tokenizers).

The function covers: poisson / mse_rate (direct rate), image_encoding_mar
(generate), conv_core_readout (scalar rate), CE-Q1 (closed-form marginal), and
the sampled tokenizer modes (CE-Q>1, flow_matching, diffusion, energy, bce,
mse, smooth_l1, continuous). After producing the rate it applies, in order:
the LDNS encode-long/predict-short splice + post-stim slice, OR the wavelet-tokenizer
``cosmos_t_safe`` slice, then ``lit._rebin_and_trim``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ntt_downstream.encoding.training.ce_marginal import compute_ce_q1_marginal_rate


@torch.no_grad()
def predict_and_align(
    lit,
    images: Tensor,
    responses: Tensor,
    key: str,
    *,
    loss_mode: str,
    model_type: str,
    n_samples: int = 1,
    flow_solver: str = "euler",
    n_flow_steps: int = 5,
    sample_temperature: float = 1.0,
    cfg_scale: float = 1.0,
    flow_shift: float = 1.0,
    mar_n_mar_steps: int = 8,
    mar_n_flow_steps: int = 5,
    mar_temperature: float = 1.0,
    mar_flow_solver: str = "euler",
    mar_flow_shift: float = 1.0,
    mar_cfg_scale: float = 1.0,
    ce_cache: dict[str, list[Tensor]] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Return aligned ``(pred_btn, target_btn)`` as ``(B, T', N)``.

    Runs the loss-mode-appropriate rate estimate, applies the LDNS/Wavelet
    boundary slice uniformly, then rebins/trims via ``lit._rebin_and_trim``.

    All sampling knobs are passed explicitly by the caller; the helper never
    reads them from ``cfg.trainer`` itself. This is deliberate: the periodic
    Lightning val and final_eval legitimately resolve these from DIFFERENT
    sources (periodic reads ``cfg.trainer.tokenizer_eval`` / ``mar_eval``;
    final_eval pins its own ``run()`` args — e.g. ``flow_solver`` defaults to
    "heun" there but "euler" in ``tokenizer_eval``). Reading them here would
    silently let one caller's config override the other's, flipping the flow
    solver. Each caller owns its resolution and passes the result in.

    Args:
        lit: object exposing ``model``, ``tokenizer_wrapper``, ``cfg``,
            ``device`` and the eval helpers ``_get_native_hz``,
            ``_rebin_and_trim``, ``_ldns_splice_active``, ``_cosmos_active``,
            ``_get_ldns_bg``, ``_ensure_tokenizer_device``.
        images, responses: batch tensors; ``responses`` is ``(B, N, T)``.
        key: session key.
        loss_mode, model_type: dispatch selectors (read from cfg by the caller).
        n_samples: MC samples for stochastic generative modes. The periodic
            val path passes 1; final_eval passes its configured count. mse /
            smooth_l1 are deterministic and always use 1 regardless.
        flow_solver, n_flow_steps, sample_temperature, cfg_scale, flow_shift:
            arguments for the ``sample_tokens`` path (CE-Q>1,
            flow/diffusion/energy, bce, continuous). Caller-resolved (see
            above). ``flow_shift`` is the inference time-grid shift for flow
            heads (1.0 = uniform; no-op for non-flow modes).
        mar_n_mar_steps, mar_n_flow_steps, mar_temperature, mar_flow_solver,
            mar_flow_shift, mar_cfg_scale: arguments for the MAR ``generate``
            path. Caller-resolved. ``mar_flow_shift``/``mar_cfg_scale`` are the
            flow-grid shift and classifier-free-guidance scale (both 1.0 = off).
        ce_cache: per-session CE-Q1 codebook-rate table cache (mutated in
            place). Defaults to ``lit._ce_codebook_rates_cache``.
        generator: optional RNG threaded into sampling for reproducibility.
            Seeding *behavior* is intentionally out of scope here.

    Returns:
        ``(pred_btn, target_btn)`` both ``(B, T', N)``.
    """
    if ce_cache is None:
        ce_cache = lit._ce_codebook_rates_cache
    # ``lit._ce_codebook_rates_cache`` is always a real dict (initialized in the
    # LitModule ctor), so after the fallback above ``ce_cache`` is never None.
    assert ce_cache is not None

    model = lit.model
    wrapper = lit.tokenizer_wrapper
    native_hz = lit._get_native_hz()

    # Tokenizer must be on the model device before any encode/decode: the
    # sanity-check val runs before the first training step's device move.
    lit._ensure_tokenizer_device()

    # --- MAR generative path: generate() already wraps decode -> (B, T, N) ---
    if model_type == "image_encoding_mar":
        # mse / smooth_l1 generate deterministically, so a single call suffices;
        # other modes are stochastic and get MC-averaged over n_samples.
        deterministic = loss_mode in ("mse", "smooth_l1")
        n_mc = 1 if deterministic else n_samples
        acc: Tensor | None = None
        for _ in range(n_mc):
            pred = model.generate(
                images,
                key,
                n_mar_steps=mar_n_mar_steps,
                n_flow_steps=mar_n_flow_steps,
                temperature=mar_temperature,
                flow_solver=mar_flow_solver,
                flow_shift=mar_flow_shift,
                cfg_scale=mar_cfg_scale,
            )  # (B, T, N)
            acc = pred if acc is None else acc + pred
        assert acc is not None
        pred = acc / n_mc
        target_btn = responses.transpose(1, 2)
        pred_hz = wrapper.reconstruction_rate_hz if wrapper is not None else native_hz
        return lit._rebin_and_trim(pred, target_btn, pred_hz, native_hz)

    # --- Tokenizer-based modes (everything except poisson/mse_rate) ---
    if loss_mode not in ("poisson", "mse_rate") and wrapper is not None:
        model_out = model(images, key, return_token_prediction=True)
        token_pred = model_out["token_prediction"]

        # Q=1 CE marginal: closed-form expected rate over the full categorical
        # posterior, no MC sampling. Avoids the OOM-prone full-vocab softmax
        # that sample_tokens materializes (~B*S*T*V fp32) and is exact.
        #
        # Escape hatch for HUGE single-codebook vocabularies (e.g. FSQ z=8,
        # V=5^8=390625): the exact marginal builds a per-position codebook-rate
        # table + einsum that OOMs a 40GB GPU regardless of val_micro_batch
        # (the table scales with V, not batch). Setting model.ce_eval_use_sampling=true
        # routes Q=1 CE through the shared MC sample_tokens path below instead,
        # which only decodes the *sampled* indices (no per-vocab table). The
        # estimate is then MC (n_samples), not exact — the intended trade for
        # tokenizers whose V is too large to marginalize.
        ce_use_sampling = bool(lit.cfg.model.get("ce_eval_use_sampling", False))
        if loss_mode == "cross_entropy" and int(getattr(model, "num_codebooks", 1)) == 1 and not ce_use_sampling:
            lit._ensure_tokenizer_device()
            target_btn = responses.transpose(1, 2)
            pred_hz = wrapper.reconstruction_rate_hz
            with torch.autocast(device_type="cuda", enabled=False):
                pred = compute_ce_q1_marginal_rate(model, wrapper, token_pred, key, cache=ce_cache)
            # The slice + rebin below is applied uniformly, so this path lives
            # on the same time support as every other loss mode.
            pred, target_btn = _apply_boundary_slice(lit, pred, responses, target_btn, loss_mode)
            return lit._rebin_and_trim(pred, target_btn, pred_hz, native_hz)

        # Sampled paths: CE-Q>1, flow/diffusion/energy, bce, mse/smooth_l1,
        # continuous. mse/smooth_l1 are deterministic (sample_tokens returns z
        # unchanged), so they collapse to a single sample.
        deterministic = loss_mode in ("mse", "smooth_l1")
        n_mc = 1 if deterministic else n_samples

        ldns_splice_t: int | None = None
        if lit._ldns_splice_active() and loss_mode not in ("cross_entropy", "bce"):
            ldns_splice_t = int(lit.cfg.model.compress_n_target_slice)

        # BSQ (bce) eval-only soft decode. Default OFF -> byte-identical hard path.
        # The hard bce decode in sample_tokens binarizes each bit via
        # (sigmoid(z) > 0.5) -> {0,1} -> code = (bit*2 - 1) * codebook_scale, i.e.
        # it snaps every position onto the nearest hypercube vertex (+/-codebook_scale
        # per channel). The BSQ collapse diagnosis showed this threshold throws away
        # most of the image-conditional modulation that *is* present in the logits z,
        # by saturating onto the dominant silence code. With the flag on we instead
        # decode the EXPECTED code -- the per-bit Bernoulli mean (2*sigmoid(z) - 1),
        # scaled by the same codebook_scale -- so the surviving logit modulation is
        # carried through the (continuous, non-requantizing) BSQ decoder instead of
        # discarded. This changes ONLY the eval decode path; the bce training loss
        # is untouched. Mirrors the existing model.ce_eval_use_sampling escape hatch.
        bce_soft_decode = loss_mode == "bce" and bool(lit.cfg.model.get("bce_eval_soft_decode", False))

        acc = None
        for _ in range(n_mc):
            if hasattr(model, "sample_tokens"):
                sampled = model.sample_tokens(
                    token_pred,
                    n_flow_steps=n_flow_steps,
                    temperature=sample_temperature,
                    cfg_scale=cfg_scale,
                    flow_solver=flow_solver,
                    flow_shift=flow_shift,
                )
            else:
                sampled = token_pred

            lit._ensure_tokenizer_device()

            # Decode predicted tokens back to response space. sample_tokens
            # returns (B, S, T, D) in unified format; for CE, D == num_codebooks.
            with torch.autocast(device_type="cuda", enabled=False):
                if loss_mode == "cross_entropy":
                    decoded = wrapper.decode_differentiable(sampled, session_key=key)
                elif loss_mode == "bce":
                    if bce_soft_decode:
                        # Expected code instead of the hard-thresholded bit. token_pred
                        # is the per-bit logit z; E[code] = (2*sigmoid(z) - 1)*scale
                        # lands in the interior of the hypercube (the hard path's
                        # endpoints are the special case sigmoid->{0,1}). The BSQ
                        # decoder is a plain conv stack with no re-quantization, so a
                        # continuous code is a valid, in-distribution input.
                        codebook_scale = wrapper.bsq_codebook_scale
                        soft_code = (2.0 * torch.sigmoid(token_pred.float()) - 1.0) * codebook_scale
                        decoded = wrapper.decode_differentiable(soft_code, session_key=key)
                    else:
                        # BSQ codes from sample_tokens are already raw tokenizer format.
                        decoded = wrapper.decode_differentiable(sampled.float(), session_key=key)
                elif ldns_splice_t is not None:
                    # LDNS-only: splice the model's short prediction into a cached
                    # real-distribution full-length latent trajectory so the
                    # decoder's InstanceNorm1d sees its trained operating length.
                    responses_btn = responses.transpose(1, 2)
                    bg = lit._get_ldns_bg(key, responses_btn)
                    bg = bg.expand(sampled.shape[0], -1, -1, -1).clone()
                    bg[:, :, :ldns_splice_t, :] = sampled.float()
                    raw_latents = wrapper._target_to_tokens(bg)
                    decoded = wrapper.decode_differentiable(raw_latents, session_key=key)
                else:
                    # Continuous latents: model predicts in scaled target space;
                    # unscale back to raw tokenizer latent space before decoding.
                    raw_latents = wrapper._target_to_tokens(sampled.float())
                    decoded = wrapper.decode_differentiable(raw_latents, session_key=key)
            acc = decoded if acc is None else acc + decoded
        assert acc is not None
        pred = acc / n_mc

        # Boundary slice ordering matters: the LDNS splice region and the Wavelet
        # safe horizon must be applied to BOTH pred and target before rebinning,
        # otherwise the rebin would mix contaminated bins into clean ones.
        pred, target_btn = _apply_boundary_slice(lit, pred, responses, None, loss_mode, ldns_splice_t=ldns_splice_t)
        pred_hz = wrapper.reconstruction_rate_hz
        return lit._rebin_and_trim(pred, target_btn, pred_hz, native_hz)

    # --- conv_core_readout: scalar (B, N) mean rate, no temporal dimension ---
    if model_type == "conv_core_readout":
        out = model(images, key)  # (B, N)
        pred = out.unsqueeze(1)  # (B, 1, N) — single "time step"
        # The target is the time-MEAN over the window, matching the model's
        # time-averaged rate prediction (single "time step").
        target_btn = responses.mean(dim=2, keepdim=True).transpose(1, 2)  # (B, 1, N)
        return pred, target_btn

    # --- poisson / mse_rate base model: (B, N, T) at native_hz ---
    out = model(images, key)  # (B, N, T)
    pred = out.transpose(1, 2)  # (B, T, N)
    target_btn = responses.transpose(1, 2)  # (B, T, N)
    return lit._rebin_and_trim(pred, target_btn, native_hz, native_hz)


def _apply_boundary_slice(lit, pred, responses, target_btn, loss_mode, *, ldns_splice_t=None):
    """Apply the LDNS post-stim slice or the wavelet-tokenizer safe-horizon slice.

    Returns ``(pred, target_btn)``. Exactly one of the two slices applies (LDNS
    splice is only set for non-CE/bce continuous modes); when neither applies the
    full window is kept. ``target_btn`` may be passed in pre-transposed (CE-Q1
    path computes it first) or ``None`` to derive it here.
    """
    if ldns_splice_t is not None:
        # Only the post-stim slice is the encoder's actual prediction target —
        # score on that region, not the spliced background.
        pred = pred[:, :ldns_splice_t, :]
        target_btn = responses[:, :, :ldns_splice_t].transpose(1, 2)
    elif lit._cosmos_active():
        # Wavelet tokenizers (wavelet-*): drop the last few output
        # bins affected by the decoder's lookahead at the splice boundary.
        # The encoder is causal in time, so leading latents do not depend on
        # the zero-padded tail; cosmos_t_safe is the empirically chosen
        # horizon that stays clear of the boundary.
        cosmos_t_safe = int(lit.tokenizer_wrapper.cosmos_t_safe)
        pred = pred[:, :cosmos_t_safe, :]
        target_btn = responses[:, :, :cosmos_t_safe].transpose(1, 2)
    elif target_btn is None:
        target_btn = responses.transpose(1, 2)
    return pred, target_btn
