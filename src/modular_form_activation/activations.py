from __future__ import annotations

"""Activation functions inspired by classical modular forms.

Currently provides:
    - DedekindEtaActivation: An activation based on the (log-magnitude of the)
      Dedekind eta function η(τ) evaluated at purely imaginary arguments.

References
----------
Dedekind eta function:
    η(τ) = q^{1/24} ∏_{n≥1} (1 - q^n),
    where q = e^{2πiτ} and Im(τ) > 0.

For τ = i·y,   y > 0,  q = e^{-2πy} ∈ (0, 1).
The infinite product converges rapidly; we truncate after `truncation` terms.

Numerical form used here (real-valued):
    log η(i·y) = (1/24) · log q + Σ_{n=1}^N log(1 - q^n).

We return  −log η(i·softplus(s·x)) so that the activation is positive and
well-behaved for all real inputs `x`.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class _EtaConfig:
    """Configuration for η-based activations."""

    truncation: int = 20  # Number of product terms N
    init_scale: float = 1.0  # Initial multiplicative scale applied to input
    learnable_scale: bool = False  # Whether `scale` is a learnable parameter


class DedekindEtaActivation(nn.Module):
    """Dedekind-eta based activation.

    The activation is defined as

        f(x) = − log η( i · softplus(scale · x) )

    where the η function is approximated by truncating its q-product after
    `truncation` terms. The resulting function is smooth, strictly positive,
    and exhibits the characteristic fractal-like behaviour of modular forms
    when composed with periodic input features.

    Parameters
    ----------
    truncation: int, default ``20``
        Number of factors kept in the infinite product definition of η.
    init_scale: float, default ``1.0``
        Initial value for the multiplicative scale parameter.
    learnable_scale: bool, default ``False``
        If ``True`` the scale parameter is registered as a learnable
        ``nn.Parameter`` – useful when the network should adapt the imaginary
        part of τ dynamically.
    """

    def __init__(
        self,
        truncation: int = 20,
        init_scale: float = 1.0,
        learnable_scale: bool = False,
    ) -> None:
        super().__init__()
        if truncation < 1:
            raise ValueError("truncation must be a positive integer")
        self.truncation = truncation

        scale_tensor = torch.tensor(float(init_scale))
        if learnable_scale:
            self.scale = nn.Parameter(scale_tensor)
        else:
            # Register as buffer so that it is moved with .to(), .cuda(), …
            self.register_buffer("scale", scale_tensor)

    @staticmethod
    def _eta_log(q: Tensor, truncation: int) -> Tensor:
        """Compute log η for real *q* (|q| < 1) via truncated product."""
        # Prevent underflow: clamp q ≥ tiny positive value so log(q) finite.
        tiny = torch.finfo(q.dtype).tiny
        q = torch.clamp(q, min=tiny, max=1.0 - 1e-7)

        # Build n = 1 … truncation on the correct device / dtype
        n = torch.arange(1, truncation + 1, device=q.device, dtype=q.dtype)
        # Compute q^n with broadcasting:  shape [..., truncation]
        q_pow_n = q.unsqueeze(-1) ** n  # noqa: NPY002 – exponentiation is fine
        # log(1 - q^n)
        log_terms = torch.log1p(-q_pow_n)
        # Σ log(1 - q^n)
        sum_log_terms = torch.sum(log_terms, dim=-1)
        # (1/24) log q
        const = torch.tensor(1.0 / 24.0, device=q.device, dtype=q.dtype)
        return const * torch.log(q) + sum_log_terms

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Apply the η-based activation to *x*.

        Notes
        -----
        We restrict the argument τ of η to the imaginary axis: τ = i·y with
        y > 0 to ensure convergence. We set y = softplus(scale · x) so that y
        is strictly positive for all real x.
        """
        # Ensure `scale` has same dtype as `x` before multiplication
        scale = self.scale.to(dtype=x.dtype)
        y = torch.nn.functional.softplus(scale * x)
        # q = e^{−2π y}
        q = torch.exp(-2.0 * torch.pi * y)
        log_eta = self._eta_log(q, self.truncation)
        # Return −log η which is positive-valued
        return -log_eta

    # Convenience alias so `DedekindEta()` can be used like a function
    def __call__(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return super().__call__(x)


__all__ = [
    "DedekindEtaActivation",
] 