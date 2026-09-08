"""
Predictive distributions, line probabilities, confidence intervals, and
market-edge math.

Distribution choice by market:
  passing_yards    Normal. Sample sizes are large (30+ attempts), CLT holds.
  rushing_yards    Gamma. Right-skewed; a 70-yard run is possible, a -20 yard
                   game is not.
  receiving_yards  Gamma with a zero-inflation component for low-target players,
                   since a WR3 has real probability mass at exactly 0.
  anytime_td       Bernoulli from a Poisson intensity.

Dispersion is heteroskedastic: cv = cv_base * (mu_ref/mu)^0.16, then inflated
by role uncertainty. A 300-yard passer is proportionally more predictable than
a 150-yard passer.
"""

from __future__ import annotations

import math
import numpy as np
from scipy import stats

from .config import (
    DISPERSION_CV, CV_VOLUME_EXP, CV_REFERENCE_MU,
    CV_UNCERTAINTY_PENALTY, CI_LEVELS, KELLY_FRACTION,
)


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------
def predictive_sd(market: str, mu: float, role_uncertainty: float = 0.0) -> float:
    if mu <= 0:
        return 0.0
    cv0 = DISPERSION_CV.get(market, 0.5)
    ref = CV_REFERENCE_MU.get(market, max(mu, 1.0))
    cv = cv0 * (ref / max(mu, 1e-6)) ** CV_VOLUME_EXP
    cv *= (1.0 + CV_UNCERTAINTY_PENALTY * float(np.clip(role_uncertainty, 0, 1)))
    return float(cv * mu)


# ---------------------------------------------------------------------------
# Distribution objects
# ---------------------------------------------------------------------------
class YardageDistribution:
    """Wraps a Normal or Gamma predictive distribution with optional zero inflation."""

    def __init__(self, market: str, mu: float, sd: float, p_zero: float = 0.0):
        self.market = market
        self.mu = max(float(mu), 0.0)
        self.sd = max(float(sd), 1e-6)
        self.p_zero = float(np.clip(p_zero, 0.0, 0.85))
        self.family = "normal" if market == "passing_yards" else "gamma"

        if self.family == "gamma":
            # mean of the non-zero component so the mixture mean equals mu
            m = self.mu / max(1e-6, (1 - self.p_zero))
            v = self.sd ** 2
            self.shape = max(m * m / max(v, 1e-6), 0.35)
            self.scale = max(v / max(m, 1e-6), 1e-6)
            self._d = stats.gamma(a=self.shape, scale=self.scale)
        else:
            self._d = stats.norm(loc=self.mu, scale=self.sd)

    def sf(self, x: float) -> float:
        """P(X > x). Half-point lines need no continuity correction."""
        if self.family == "gamma":
            return float((1 - self.p_zero) * self._d.sf(max(x, 0.0)))
        return float(self._d.sf(x))

    def cdf(self, x: float) -> float:
        return 1.0 - self.sf(x)

    def quantile(self, q: float) -> float:
        q = float(np.clip(q, 1e-6, 1 - 1e-6))
        if self.family == "gamma":
            if q <= self.p_zero:
                return 0.0
            qq = (q - self.p_zero) / (1 - self.p_zero)
            return float(self._d.ppf(qq))
        return float(self._d.ppf(q))

    def interval(self, level: float = 0.80) -> tuple[float, float]:
        a = (1 - level) / 2
        return (round(self.quantile(a), 1), round(self.quantile(1 - a), 1))

    def median(self) -> float:
        return round(self.quantile(0.5), 1)


def zero_inflation_prob(market: str, expected_opportunities: float) -> float:
    """
    P(exactly zero yards). Driven by the chance of zero productive touches.
    Poisson-ish on opportunities, damped because a target can still gain 0.
    """
    if market == "receiving_yards":
        return float(np.clip(math.exp(-0.72 * max(expected_opportunities, 0.0)), 0.0, 0.55))
    if market == "rushing_yards":
        return float(np.clip(math.exp(-1.05 * max(expected_opportunities, 0.0)), 0.0, 0.45))
    return 0.0


# ---------------------------------------------------------------------------
# Line probability
# ---------------------------------------------------------------------------
def line_probabilities(dist: YardageDistribution, line: float) -> dict:
    p_over = dist.sf(line)
    return {
        "line": float(line),
        "p_over": round(float(p_over), 4),
        "p_under": round(float(1 - p_over), 4),
        "fair_over_odds": american_odds(p_over),
        "fair_under_odds": american_odds(1 - p_over),
    }


def anytime_td_probability(lam: float, kappa: float = 0.965) -> float:
    """
    P(at least one TD) from a Poisson intensity, with a calibration constant.
    Raw Poisson slightly overstates the multi-TD tail because TD events within
    a game are positively correlated with blowout scripts we already priced in.
    """
    lam = max(float(lam), 0.0)
    return float(np.clip(1.0 - math.exp(-lam * kappa), 0.0, 0.97))


# ---------------------------------------------------------------------------
# Odds utilities
# ---------------------------------------------------------------------------
def american_odds(p: float) -> int:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def implied_prob(american: float) -> float:
    """Vig-inclusive break-even probability of an American price."""
    a = float(american)
    if a < 0:
        return abs(a) / (abs(a) + 100.0)
    return 100.0 / (a + 100.0)


def devig_two_way(over_odds: float, under_odds: float) -> tuple[float, float]:
    """Multiplicative (proportional) de-vig."""
    po, pu = implied_prob(over_odds), implied_prob(under_odds)
    s = po + pu
    if s <= 0:
        return (0.5, 0.5)
    return (po / s, pu / s)


def edge_and_stake(p_model: float, american: float,
                   p_market_devig: float | None = None) -> dict:
    p_book = implied_prob(american)
    ref = p_market_devig if p_market_devig is not None else p_book
    b = (american / 100.0) if american > 0 else (100.0 / abs(american))
    ev = p_model * b - (1 - p_model)
    kelly = (p_model * (b + 1) - 1) / b if b > 0 else 0.0
    return {
        "book_odds": int(american),
        "book_implied_prob": round(p_book, 4),
        "market_devig_prob": round(ref, 4) if ref is not None else None,
        "model_prob": round(float(p_model), 4),
        "edge_pct_points": round(float(p_model - ref), 4),
        "expected_value_per_unit": round(float(ev), 4),
        "kelly_full": round(float(max(kelly, 0.0)), 4),
        "kelly_recommended": round(float(max(kelly, 0.0) * KELLY_FRACTION), 4),
    }


def intervals(dist: YardageDistribution, levels=CI_LEVELS) -> dict:
    return {f"ci_{int(l*100)}": list(dist.interval(l)) for l in levels}
