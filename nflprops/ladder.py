"""
Alt-line ladder engine.

Most prop value is not in the median. A player can project dead-on the market
number and still be badly mispriced at 100+ yards, because books price the
tail off a generic distribution shape rather than that player's actual
outcome profile. This module turns each projection's predictive distribution
into a full ladder and scores how attractive the ceiling is.

LADDER SCORE (0-100)
--------------------
Deliberately NOT a function of the median. A player projecting at market can
still carry an elite ceiling. Inputs, each normalized 0-1 then weighted:

    ceiling_ratio   P(reach a ceiling rung) vs what a league-typical player
                    at the same median would produce. This is the core term:
                    it isolates distribution SHAPE from level.
    volume          projected opportunities (more touches = more tail)
    explosiveness   aDOT for receivers, YPC for backs, air yards for QBs
    team_total      implied points, a proxy for scoring environment
    matchup         the model's own matchup multiplier
    role_stability  inverse of role uncertainty; a volatile role has a fat
                    tail but an untrustworthy one, so this DAMPS the score
                    rather than boosting it

A high ladder score means: the tail is fatter than the median implies, and
the role is stable enough to trust it.
"""

from __future__ import annotations

import numpy as np

LADDER_RUNGS = {
    "passing_yards": [200, 225, 250, 275, 300, 325, 350],
    "rushing_yards": [50, 75, 100, 125, 150],
    "receiving_yards": [50, 75, 100, 125, 150],
    "receptions": [3, 5, 7, 9, 11],
}

# the rung used as the "ceiling" reference when scoring
CEILING_RUNG = {
    "passing_yards": 300,
    "rushing_yards": 100,
    "receiving_yards": 100,
    "receptions": 7,
}

# reference volume for normalizing the volume term
VOLUME_REF = {"passing_yards": 35.0, "rushing_yards": 18.0,
              "receiving_yards": 9.0, "receptions": 9.0}


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def build_ladder(dist, market: str) -> list[dict]:
    """
    P(X >= rung) for every rung. Uses the integer continuity correction, so
    these line up directly with Kalshi "X or more" contracts.
    """
    rungs = LADDER_RUNGS.get(market, [])
    out = []
    for r in rungs:
        p = float(dist.sf(r - 0.5))
        out.append({
            "rung": r,
            "probability": round(p, 4),
            "fair_odds_american": _american(p),
            "fair_price_cents": round(p * 100, 1),
        })
    return out


def _american(p: float) -> int:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def _baseline_ceiling_prob(market: str, mu: float, dist_cls) -> float:
    """
    What P(ceiling) would be for a league-TYPICAL player with this same
    median. Comparing against this is what separates distribution shape from
    projection level -- otherwise the ladder score just re-ranks by median.
    """
    from .probability import YardageDistribution, predictive_sd
    if mu <= 0:
        return 1e-6
    sd = predictive_sd(market, mu, 0.0)   # no role-uncertainty inflation
    ref = YardageDistribution(market, mu, sd, 0.0)
    return max(float(ref.sf(CEILING_RUNG.get(market, 100) - 0.5)), 1e-6)


def ladder_score(dist, market: str, features: dict, opportunities: float,
                 projection: float) -> dict:
    """Score 0-100 plus the component breakdown, so it is explainable."""
    from .probability import YardageDistribution
    ceiling = CEILING_RUNG.get(market, 100)
    p_ceiling = float(dist.sf(ceiling - 0.5))
    p_base = _baseline_ceiling_prob(market, max(projection, 1.0), YardageDistribution)

    # shape term: how much fatter is this tail than a typical player's at the
    # same median? ratio > 1 means genuine ceiling equity beyond the median.
    shape = _clamp01(np.log1p(max(p_ceiling / p_base, 0.0)) / np.log(3.0))

    vol = _clamp01(opportunities / VOLUME_REF.get(market, 10.0))

    if market == "passing_yards":
        explosive = _clamp01((features.get("ypa") or 7.0) / 9.0)
    elif market == "rushing_yards":
        explosive = _clamp01((features.get("ypc") or 4.3) / 5.5)
    else:
        explosive = _clamp01((features.get("adot") or 8.0) / 14.0)

    team_total = _clamp01(((features.get("team_total_implied") or 22.0) - 14.0) / 16.0)

    if market == "rushing_yards":
        matchup = _clamp01(((features.get("rush_matchup_mult") or 1.0) - 0.84) / 0.34)
    else:
        matchup = _clamp01(((features.get("pass_matchup_mult") or 1.0) - 0.86) / 0.30)

    stability = _clamp01(1.0 - (features.get("role_uncertainty") or 0.1))

    components = {
        "ceiling_shape": round(shape, 3),
        "volume": round(vol, 3),
        "explosiveness": round(explosive, 3),
        "team_total": round(team_total, 3),
        "matchup": round(matchup, 3),
        "role_stability": round(stability, 3),
    }
    weights = {"ceiling_shape": 0.34, "volume": 0.22, "explosiveness": 0.14,
               "team_total": 0.12, "matchup": 0.10, "role_stability": 0.08}
    score = sum(components[k] * w for k, w in weights.items()) * 100.0

    return {
        "ladder_score": round(score, 1),
        "p_ceiling": round(p_ceiling, 4),
        "ceiling_rung": ceiling,
        "p_ceiling_vs_typical": round(p_ceiling / p_base, 2),
        "components": components,
    }


def percentiles(dist) -> dict:
    """Floor / quartiles / ceiling, for outcome-range display."""
    return {
        "floor_p10": round(dist.quantile(0.10), 1),
        "p25": round(dist.quantile(0.25), 1),
        "median": round(dist.quantile(0.50), 1),
        "p75": round(dist.quantile(0.75), 1),
        "ceiling_p90": round(dist.quantile(0.90), 1),
    }


# ---------------------------------------------------------------------------
# Market divergence diagnostic
# ---------------------------------------------------------------------------
def divergence_report(projection: float, market_line: float, features: dict,
                      opportunities: float, efficiency: float,
                      market: str) -> dict:
    """
    When the model and the market disagree materially, say WHY and whether the
    disagreement is supported. A large divergence with weak support is far
    more likely to be a model error than an edge -- that is the lesson from
    the audit, where a backup QB projected above every elite starter and the
    math was "correct" the whole way down.
    """
    if not market_line or market_line <= 0:
        return {"divergence_pct": None, "verdict": "no_market"}

    diff = projection - market_line
    pct = diff / market_line

    drivers = []
    vm = features.get("team_total_volume_mult")
    if vm is not None and abs(vm - 1.0) > 0.04:
        drivers.append({"driver": "scoring environment",
                        "direction": "up" if vm > 1 else "down",
                        "detail": f"implied team total {features.get('team_total_implied')}"})
    mm = (features.get("rush_matchup_mult") if market == "rushing_yards"
          else features.get("pass_matchup_mult"))
    if mm is not None and abs(mm - 1.0) > 0.03:
        drivers.append({"driver": "opponent matchup",
                        "direction": "up" if mm > 1 else "down",
                        "detail": f"multiplier {round(mm, 3)}"})
    sa = features.get("pass_rate_script_adj")
    if sa and abs(sa) > 0.02:
        drivers.append({"driver": "game script",
                        "direction": "up" if sa > 0 else "down",
                        "detail": f"spread {features.get('spread')}"})
    wp = features.get("weather_penalty")
    if wp is not None and wp < 0.97:
        drivers.append({"driver": "weather", "direction": "down",
                        "detail": f"penalty {round(wp, 3)}"})
    ru = features.get("role_uncertainty") or 0.0
    if ru > 0.30:
        drivers.append({"driver": "role uncertainty", "direction": "caution",
                        "detail": f"uncertainty {round(ru, 2)}"})

    # A divergence is "supported" only if identifiable drivers point the same
    # way as the divergence itself.
    up = sum(1 for d in drivers if d["direction"] == "up")
    down = sum(1 for d in drivers if d["direction"] == "down")
    if abs(pct) < 0.15:
        verdict = "aligned"
    elif (pct > 0 and up > down) or (pct < 0 and down > up):
        verdict = "supported"
    elif ru > 0.30:
        verdict = "low_confidence_role"
    else:
        verdict = "unsupported_check_model"

    return {
        "model_projection": round(projection, 1),
        "market_line": market_line,
        "divergence_yards": round(diff, 1),
        "divergence_pct": round(pct, 4),
        "drivers": drivers,
        "verdict": verdict,
    }
