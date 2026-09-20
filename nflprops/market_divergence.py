"""
Ladder-level market divergence.

Comparing a model against a single market strike is misleading. A player's
Kalshi ladder carries several strikes at once, and they answer different
questions. Running the naive per-strike comparison on James Cook produced
three contradictory verdicts from one market:

    15+ yards   market 58.5%  model 51.1%   -> "unsupported"
    25+ yards   market 35.5%  model 24.1%   -> "aligned"
    40+ yards   market 16.5%  model  6.1%   -> "supported"

The contradiction is the signal. The medians broadly agree (both near 17
yards), but the market's TAIL is far fatter than the model's. That is a
distribution-shape disagreement, not a projection disagreement, and the two
have completely different implications:

    LEVEL divergence   the model and market disagree about the median. This
                       is a genuine opinion difference about usage, matchup
                       or game script, and it is where median-prop value
                       lives.

    SHAPE divergence   the model and market agree on the median but disagree
                       about the tails. This is where ALT-LINE value lives --
                       and it is also the classic signature of a model whose
                       variance assumptions are wrong, so it must be treated
                       with suspicion before it is treated as edge.

This module decomposes the two and says which is dominant.
"""

from __future__ import annotations

import numpy as np


def _market_implied_quantiles(contracts: list[dict]) -> list[tuple[float, float]]:
    """[(strike, market P(>= strike))], liquid contracts only, sorted."""
    out = []
    for c in contracts:
        if not c.get("liquid"):
            continue
        s = c.get("market_strike")
        p = c.get("market_implied_probability")
        if s is None or p is None:
            continue
        if 0.02 < float(p) < 0.98:
            out.append((float(s), float(p)))
    return sorted(out)


def _interp_median(points: list[tuple[float, float]]) -> float | None:
    """
    Estimate the market's median from its ladder: the strike where
    P(>= strike) crosses 0.50, linearly interpolated.
    """
    if len(points) < 2:
        return None
    pts = sorted(points, key=lambda x: -x[1])   # descending probability
    for (s1, p1), (s2, p2) in zip(pts, pts[1:]):
        if (p1 >= 0.5 >= p2) and p1 != p2:
            w = (p1 - 0.5) / (p1 - p2)
            return s1 + w * (s2 - s1)
    # 0.5 not bracketed: extrapolate from the two closest strikes
    (s1, p1), (s2, p2) = pts[0], pts[1]
    if p1 == p2:
        return None
    slope = (s2 - s1) / (p2 - p1)
    return s1 + slope * (0.5 - p1)


def ladder_divergence(contracts: list[dict], model_projection: float,
                      model_dist=None) -> dict:
    """
    Decompose model-vs-market disagreement across a full ladder.

    contracts: the per-strike rows produced by fetch_kalshi (each carrying
               market_strike, market_implied_probability, model_probability,
               liquid).
    """
    pts = _market_implied_quantiles(contracts)
    if len(pts) < 2:
        return {"status": "insufficient_ladder",
                "n_liquid_strikes": len(pts)}

    market_median = _interp_median(pts)
    if market_median is None:
        return {"status": "median_not_bracketed", "n_liquid_strikes": len(pts)}

    level_gap = float(model_projection) - float(market_median)
    level_pct = level_gap / max(float(market_median), 1e-6)

    # Shape: compare model vs market probability at each strike AFTER
    # removing the level difference, by measuring each strike in units of
    # distance above the respective median.
    tail_gaps = []
    for c in contracts:
        if not c.get("liquid"):
            continue
        s, pm, pmod = (c.get("market_strike"), c.get("market_implied_probability"),
                       c.get("model_probability"))
        if s is None or pm is None or pmod is None:
            continue
        if float(s) > market_median:          # tail strikes only
            tail_gaps.append(float(pmod) - float(pm))

    mean_tail_gap = float(np.mean(tail_gaps)) if tail_gaps else 0.0

    # which effect dominates?
    level_signal = abs(level_pct)
    shape_signal = abs(mean_tail_gap)
    if level_signal < 0.08 and shape_signal < 0.04:
        verdict = "model_and_market_agree"
    elif shape_signal >= 0.04 and level_signal < 0.12:
        verdict = ("market_tail_fatter_than_model" if mean_tail_gap < 0
                   else "model_tail_fatter_than_market")
    elif level_signal >= 0.12:
        verdict = ("model_projects_higher" if level_gap > 0
                   else "model_projects_lower")
    else:
        verdict = "mixed"

    interpretation = {
        "model_and_market_agree":
            "No material disagreement. No edge here.",
        "market_tail_fatter_than_model":
            ("Medians agree but the market prices ceiling outcomes far higher "
             "than the model. Either the model's variance is too narrow (most "
             "likely, and a reason to distrust any alt-line 'edge' here), or "
             "the market is overpaying for the tail. Check the model's "
             "dispersion before treating this as value."),
        "model_tail_fatter_than_market":
            ("Medians agree but the model sees more ceiling than the market "
             "prices. If the dispersion assumptions hold up, this is exactly "
             "where alt-line/ladder value lives."),
        "model_projects_higher":
            ("Genuine level disagreement: the model projects materially above "
             "the market's implied median. Check the attribution breakdown to "
             "see which driver is responsible and whether it is defensible."),
        "model_projects_lower":
            ("Genuine level disagreement: the model projects materially below "
             "the market. Historically in this build that has more often meant "
             "understated opportunity in the model than a real market error."),
        "mixed": "Both level and shape differ. Treat with caution.",
    }

    return {
        "status": "ok",
        "n_liquid_strikes": len(pts),
        "model_projection": round(float(model_projection), 1),
        "market_implied_median": round(float(market_median), 1),
        "level_gap": round(level_gap, 1),
        "level_gap_pct": round(level_pct, 4),
        "mean_tail_prob_gap": round(mean_tail_gap, 4),
        "verdict": verdict,
        "interpretation": interpretation[verdict],
        "strikes": [
            {"strike": c.get("market_strike"),
             "market_prob": c.get("market_implied_probability"),
             "model_prob": c.get("model_probability"),
             "gap": (None if c.get("model_probability") is None
                     or c.get("market_implied_probability") is None
                     else round(c["model_probability"] - c["market_implied_probability"], 4)),
             "liquid": c.get("liquid")}
            for c in sorted(contracts, key=lambda x: (x.get("market_strike") or 0))
        ],
    }
