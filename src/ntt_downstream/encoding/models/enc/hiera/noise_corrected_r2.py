"""Noise-corrected R^2 between a model and trial-averaged neural responses.

Single-trial responses are noisy, so the plain R^2 between a model and the
repeat-averaged response underestimates how much of the *explainable*
(stimulus-driven) variance the model captures. Pospisil & Bair (2021) derive
an approximately unbiased estimator of the squared correlation between a
model and the expected response ("ER"); this module implements that estimator
in torch, vectorised over neurons, with support for ragged repeat counts.

Write ``x`` for the mean-centred model prediction of one neuron over ``I``
images, ``y`` for its mean-centred repeat-averaged response, and ``k`` for the
variance of the repeat average (trial variance divided by the repeat count).
The raw squared correlation ``<x, y>^2 / (|x|^2 |y|^2)`` is biased upward in
numerator and denominator by noise in ``y``; subtracting the expected
contribution of that noise from each gives

    numerator   = <x, y>^2 - k |x|^2
    denominator = |x|^2 (|y|^2 - (I - 1) k)

and the estimate is their ratio.

References:
    Pospisil & Bair (2021). "The unbiased estimation of the fraction of
    variance explained by a model."
    Pospisil & Bair (2021). "The unbiased estimation of r^2 between two sets
    of noisy neural responses." bioRxiv. Reference implementation:
    https://github.com/deanpospisil/er_est (see ``er_est.py``).

The Cowley group's V4 compact-model code applies the same estimator to this
dataset and informed the interface (neurons x images x repeats); this
implementation is written independently from the formula above. Its output is
pinned against reference values in ``tests/encoding/test_eval_metrics.py``
because ``nc_r2`` is a published encoding score.
"""

from __future__ import annotations

import torch

# Below this magnitude the denominator carries no usable signal (constant
# model or constant response), so the estimate is reported as NaN.
_DENOMINATOR_FLOOR = 1e-12


def _trial_noise_of_mean(trials: torch.Tensor) -> torch.Tensor:
    """Variance of the repeat-averaged response, per neuron.

    Args:
        trials: ``(N, I, R)`` single-trial responses, NaN where a repeat is
            missing.

    Returns:
        ``(N,)`` tensor ``k``: the unbiased per-image trial variance averaged
        over images, divided by the median number of valid repeats per image.
        Images with fewer than two valid repeats contribute zero variance.
    """
    present = ~torch.isnan(trials)
    repeat_counts = present.sum(dim=2).float()  # (N, I)

    centred = trials - torch.nanmean(trials, dim=2, keepdim=True)
    ss_within = torch.where(present, centred, torch.zeros_like(centred)).pow(2).sum(dim=2)
    trial_var = ss_within / (repeat_counts - 1).clamp(min=1)  # Bessel-corrected

    typical_repeats = repeat_counts.median(dim=1).values
    return trial_var.mean(dim=1) / typical_repeats


def noise_corrected_r2(
    responses_all_trials: torch.Tensor,
    responses_hat: torch.Tensor,
) -> torch.Tensor:
    """Compute the noise-corrected R^2 (ER estimator) per neuron.

    Args:
        responses_all_trials: ``(N_neurons, N_images, N_repeats)`` single-trial
            responses. NaN marks a missing repeat (ragged repeat counts).
        responses_hat: ``(N_neurons, N_images)`` model predictions for the
            repeat-averaged response.

    Returns:
        ``(N_neurons,)`` noise-corrected R^2. The value is a ratio of two
        individually unbiased estimates, so it can be negative or exceed 1
        under estimation noise, and it is sign-agnostic (an anti-correlated
        model scores like a correlated one). NaN where the denominator is
        numerically zero.
    """
    n_images = responses_all_trials.shape[1]
    noise_of_mean = _trial_noise_of_mean(responses_all_trials)  # (N,)

    x = responses_hat - responses_hat.mean(dim=1, keepdim=True)
    mean_response = torch.nanmean(responses_all_trials, dim=2)
    y = mean_response - mean_response.mean(dim=1, keepdim=True)

    x_energy = x.pow(2).sum(dim=1)
    y_energy = y.pow(2).sum(dim=1)
    cross = (x * y).sum(dim=1)

    numerator = cross.pow(2) - noise_of_mean * x_energy
    denominator = x_energy * y_energy - noise_of_mean * (n_images - 1) * x_energy

    usable = denominator.abs() > _DENOMINATOR_FLOOR
    return torch.where(usable, numerator / denominator, torch.full_like(denominator, float("nan")))
