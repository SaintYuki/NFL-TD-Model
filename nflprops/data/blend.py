"""
Weighted-decay blending of prior-season, current-season, and league-baseline
values.

Core identity for any metric x:

    w_cur    = n_games_current / (n_games_current + K_metric)
    w_prior  = (1 - w_cur) * rho
    w_league = (1 - w_cur) * (1 - rho)
    x_blend  = w_cur * x_cur + w_prior * x_prior + w_league * x_league

n_games_current  number of current-season games with usable snaps
K_metric         stabilization constant (config.STABILIZATION_K)
rho              roster/scheme continuity factor (0..1) from roster.py

Week 1 has n_cur = 0, so w_cur = 0 and the split is entirely prior-vs-league,
governed by rho. A returning starter on the same team with the same OC gets
rho ~= 0.92, i.e. 92/8 prior/league. A player who changed teams gets
0.92 * 0.62 = 0.57, i.e. 57/43. A rookie gets rho = 0 and sits on the
positional baseline until real snaps arrive.

By Week 6 a target share (K=4) is 60% current-season weighted; a yards-per-carry
(K=12) is only 33% current-season weighted. That asymmetry is the point:
opportunity stabilizes long before efficiency does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    ModelConfig,
    DEFAULT_CONFIG,
    WEEK_WEIGHT_SCHEDULE,
)


def current_weight(metric: str, n_games_current: float, week: int | None = None,
                   cfg: ModelConfig = DEFAULT_CONFIG) -> float:
    """Weight placed on current-season observations for `metric`."""
    if cfg.use_schedule_override and week is not None:
        return float(WEEK_WEIGHT_SCHEDULE.get(int(week), 0.96))
    k = cfg.k(metric)
    n = max(0.0, float(n_games_current))
    return n / (n + k)


def blend_value(metric: str,
                x_current: float | None,
                x_prior: float | None,
                x_league: float,
                n_games_current: float,
                rho: float,
                week: int | None = None,
                cfg: ModelConfig = DEFAULT_CONFIG) -> float:
    """Three-way blend. Missing inputs collapse their weight into the others."""
    w_cur = current_weight(metric, n_games_current, week, cfg)
    if x_current is None or (isinstance(x_current, float) and np.isnan(x_current)):
        w_cur = 0.0
        x_current = 0.0
    rho = float(np.clip(rho, 0.0, 1.0))
    w_prior = (1.0 - w_cur) * rho
    w_league = (1.0 - w_cur) * (1.0 - rho)

    if x_prior is None or (isinstance(x_prior, float) and np.isnan(x_prior)):
        w_league += w_prior
        w_prior = 0.0
        x_prior = 0.0

    # No league baseline supplied: the "regress to average" leg has no target,
    # so return its weight to the prior season rather than silently pulling
    # the estimate toward zero.
    if x_league is None or (isinstance(x_league, float) and np.isnan(x_league)):
        w_prior += w_league
        w_league = 0.0
        x_league = 0.0

    total = w_cur + w_prior + w_league
    if total <= 0:
        return float(x_league)
    return float((w_cur * x_current + w_prior * x_prior + w_league * x_league) / total)


def blend_weights_report(metric: str, n_games_current: float, rho: float,
                         week: int | None = None,
                         cfg: ModelConfig = DEFAULT_CONFIG) -> dict:
    """Expose the weights so they can be attached to a projection as a driver."""
    w_cur = current_weight(metric, n_games_current, week, cfg)
    return {
        "metric": metric,
        "w_current_season": round(w_cur, 4),
        "w_prior_season": round((1 - w_cur) * rho, 4),
        "w_league_baseline": round((1 - w_cur) * (1 - rho), 4),
        "stabilization_k": cfg.k(metric),
        "continuity_rho": round(rho, 4),
    }


# ---------------------------------------------------------------------------
# Within-season recency decay
# ---------------------------------------------------------------------------
def decayed_mean(values: pd.Series, weeks: pd.Series, current_week: int,
                 decay: float = None, cfg: ModelConfig = DEFAULT_CONFIG) -> float:
    """
    Exponentially weighted mean of a current-season game log.
    weight_g = decay ** (current_week - week_g)
    """
    decay = cfg.gamelog_decay if decay is None else decay
    if len(values) == 0:
        return np.nan
    v = pd.to_numeric(values, errors="coerce").astype(float)
    w = np.power(decay, np.maximum(0, current_week - weeks.astype(float)))
    mask = ~v.isna()
    if mask.sum() == 0 or w[mask].sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def decayed_rate(numerator: pd.Series, denominator: pd.Series, weeks: pd.Series,
                 current_week: int, decay: float = None,
                 cfg: ModelConfig = DEFAULT_CONFIG) -> float:
    """
    Recency-weighted RATE (e.g. targets/team targets). Weighting the ratio of
    sums rather than the mean of ratios avoids blowing up on small denominators.
    """
    decay = cfg.gamelog_decay if decay is None else decay
    num = pd.to_numeric(numerator, errors="coerce").fillna(0).astype(float)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0).astype(float)
    w = np.power(decay, np.maximum(0, current_week - weeks.astype(float)))
    dsum = float((w * den).sum())
    if dsum <= 0:
        return np.nan
    return float((w * num).sum() / dsum)


def blend_frame(df_current: pd.DataFrame,
                df_prior: pd.DataFrame,
                metrics: list[str],
                key: str,
                league: dict,
                n_games_col: str = "n_games_current",
                rho_col: str = "rho",
                week: int | None = None,
                cfg: ModelConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Vectorized blend of many metrics for many entities at once.
    df_current must carry `key`, `n_games_col`, `rho_col`.
    df_prior must carry `key` and the prior-season metric columns.
    """
    cur = df_current.set_index(key)
    pri = df_prior.set_index(key)
    out = pd.DataFrame(index=cur.index)
    n = cur[n_games_col].astype(float).fillna(0.0)
    rho = cur[rho_col].astype(float).fillna(0.0).clip(0, 1)

    for m in metrics:
        w_cur = np.array([current_weight(m, ni, week, cfg) for ni in n])
        x_cur = pd.to_numeric(cur.get(m, pd.Series(index=cur.index, dtype=float)),
                              errors="coerce")
        x_pri = pd.to_numeric(pri.reindex(cur.index).get(
            m, pd.Series(index=cur.index, dtype=float)), errors="coerce")
        x_lg = float(league.get(m, np.nan))

        w_c = np.where(x_cur.isna(), 0.0, w_cur)
        w_p = (1 - w_c) * rho.values
        w_l = (1 - w_c) * (1 - rho.values)
        # prior missing -> its weight moves to league
        p_missing = x_pri.isna().values
        w_l = np.where(p_missing, w_l + w_p, w_l)
        w_p = np.where(p_missing, 0.0, w_p)

        num = (w_c * x_cur.fillna(0).values
               + w_p * x_pri.fillna(0).values
               + w_l * (0.0 if np.isnan(x_lg) else x_lg))
        den = w_c + w_p + w_l
        out[m] = np.where(den > 0, num / np.where(den == 0, 1, den), np.nan)
        out[f"__wcur_{m}"] = w_c
    return out.reset_index()
