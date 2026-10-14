"""MR sequence signal equations used by the physiology simulator."""

from __future__ import annotations

import numpy as np


def _gradient_echo(t1: float, t2: float, rho: float, *, te: float, tr: float, alpha: float) -> float:
    alpha_rad = np.deg2rad(alpha)
    return float(
        rho
        * np.sin(alpha_rad)
        * (1 - np.exp(-tr / t1))
        * np.exp(-te / t2)
        / (1 - np.cos(alpha_rad) * np.exp(-tr / t1))
    )


def vibe(t1: float, t2: float, rho: float) -> float:
    return _gradient_echo(t1, t2, rho, te=4.54, tr=7.25, alpha=10)


def vibe_opposed(t1: float, t2: float, rho: float) -> float:
    return _gradient_echo(t1, t2, rho, te=2.30, tr=7.25, alpha=10)


def _dixon(t1: float, t2: float, rho: float, *, te: float) -> float:
    base = _gradient_echo(t1, t2, rho, te=te, tr=4.6, alpha=10)
    phase_shift = np.exp(1j * 2 * np.pi * 220.0 * te / 1000)
    return float(base * np.abs(0.9 + 0.1 * phase_shift))


def vibe_dixon_in(t1: float, t2: float, rho: float) -> float:
    return _dixon(t1, t2, rho, te=2.3)


def vibe_dixon_opposed(t1: float, t2: float, rho: float) -> float:
    return _dixon(t1, t2, rho, te=1.15)


def gre_t1(t1: float, t2_star: float, rho: float) -> float:
    return _gradient_echo(t1, t2_star, rho, te=4.0, tr=7.0, alpha=10)


def t2_space(t1: float, t2: float, rho: float = 1.0) -> float:
    return float(rho * (1 - np.exp(-2000.0 / t1)) * np.exp(-150.0 / t2))
