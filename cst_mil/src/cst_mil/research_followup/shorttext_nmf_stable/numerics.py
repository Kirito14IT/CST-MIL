"""Class-local arithmetic repair for the frozen batch Frobenius MU objective.

The original update F *= numerator/denominator has a removable intermediate
overflow: a locked zero factor can become 0*inf, contaminating every topic.
Zero factors/numerators retain zero. Finite ratios follow the original arithmetic;
only positive factors with overflowing ratios evaluate the same product in logs.
True out-of-range products and invalid operands still fail. No library globals
are changed, no positive denominator floor/regularizer is added, and the outer
solver/initialization/stopping logic is the installed, fingerprinted sklearn code.
"""

from __future__ import annotations

from types import FunctionType

import numpy as np
from sklearn.decomposition import NMF, _nmf
from sklearn.utils.extmath import safe_sparse_dot


def stable_product(factor, numerator, denominator, gamma=1.0):
    """Evaluate the same multiplicative update without zero-times-inf poisoning."""
    if any(not np.isfinite(v).all() or np.any(v < 0) for v in (factor, numerator, denominator)):
        raise FloatingPointError("MU operands must be finite and nonnegative")
    if np.any(denominator <= 0):
        raise FloatingPointError("MU denominator must be positive after original zero guard")
    output = np.zeros_like(factor)
    active = (factor > 0) & (numerator > 0)
    ratio = np.zeros_like(numerator)
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        np.divide(numerator, denominator, out=ratio, where=active)
    unsafe = active & ~np.isfinite(ratio)
    ordinary = active & ~unsafe
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        ordinary_ratio = ratio[ordinary]
        if gamma != 1:
            ordinary_ratio **= gamma
        output[ordinary] = factor[ordinary] * ordinary_ratio
        if unsafe.any():
            logs = np.log(factor[unsafe]) + gamma * (
                np.log(numerator[unsafe]) - np.log(denominator[unsafe])
            )
            output[unsafe] = np.exp(logs)
    if not np.isfinite(output).all():
        raise FloatingPointError("True MU product exceeded floating-point range")
    return output


def stable_w(
    X, W, H, beta_loss, l1_reg_W, l2_reg_W, gamma, H_sum=None, HHt=None, XHt=None, update_H=True
):
    if beta_loss != 2 or gamma != 1:
        raise ValueError("Recovery only supports the frozen Frobenius MU protocol")
    if XHt is None:
        XHt = safe_sparse_dot(X, H.T)
    if HHt is None:
        HHt = np.dot(H, H.T)
    denominator = np.dot(W, HHt)
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator = denominator + l2_reg_W * W
    denominator[denominator == 0] = _nmf.EPSILON
    W[:] = stable_product(W, XHt, denominator, gamma)
    return W, H_sum, HHt, XHt


def stable_h(X, W, H, beta_loss, l1_reg_H, l2_reg_H, gamma, A=None, B=None, rho=None):
    if beta_loss != 2 or gamma != 1 or A is not None or B is not None:
        raise ValueError("Recovery only supports batch Frobenius MU")
    numerator = safe_sparse_dot(W.T, X)
    denominator = np.linalg.multi_dot([W.T, W, H])
    if l1_reg_H > 0:
        denominator += l1_reg_H
    if l2_reg_H > 0:
        denominator = denominator + l2_reg_H * H
    denominator[denominator == 0] = _nmf.EPSILON
    H[:] = stable_product(H, numerator, denominator, gamma)
    return H


def _clone_function(function, replacements):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    clone = FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


_local_fit_mu = _clone_function(
    _nmf._fit_multiplicative_update,
    {
        "_multiplicative_update_w": stable_w,
        "_multiplicative_update_h": stable_h,
    },
)


class StableMU(NMF):
    """Original sklearn NMF with isolated stable MU arithmetic functions."""

    _fit_transform = _clone_function(
        NMF._fit_transform,
        {
            "_fit_multiplicative_update": _local_fit_mu,
        },
    )
