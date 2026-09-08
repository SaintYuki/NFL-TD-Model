"""
Passing / Rushing / Receiving yardage models.

All three share the same skeleton: project OPPORTUNITY, project EFFICIENCY,
multiply, then attach a predictive distribution. Keeping the two halves
separate is what makes roster changes tractable: a trade moves opportunity, a
new offensive line moves efficiency, and they are rarely the same size.

PASSING YARDS
    attempts = team_dropbacks * qb_dropback_share * (1 - sack_rate)
    ypa      = blended_ypa
               * pass_matchup_mult
               * weather_penalty
               + ypa_unit_delta                  (OL pass block + WR corps)
    yards    = attempts * ypa
    Distribution: Normal, cv ~ 0.265 at 240 yards.

RUSHING YARDS
    carries  = team_carries * rush_share_adj
               where rush_share_adj = blended_rush_share
                                      + rush_share_script_adj (spread-driven)
    ypc      = blended_ypc * rush_matchup_mult + ypc_unit_delta
    yards    = carries * ypc
    Distribution: Gamma, cv ~ 0.48, zero-inflated for low-carry backs.

RECEIVING YARDS
    targets  = team_pass_attempts * target_share_adj
               target_share_adj = blended_target_share
                                  * matchup_share_mult * roster_usage_mult
    ypt      = blended_yptarget
               * pass_matchup_mult * matchup_eff_mult (WR/CB) * weather
               + yptarget_unit_delta (QB quality)
    yards    = targets * ypt
    Air-yards share is used as a second opinion on ypt: a receiver whose air
    yards share far exceeds his target share gets an upward ypt correction.
    Distribution: Gamma, cv ~ 0.585, zero-inflated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ModelConfig, DEFAULT_CONFIG
from ..probability import (
    YardageDistribution, predictive_sd, zero_inflation_prob,
    line_probabilities, intervals,
)


def _g(f: dict, key: str, default: float) -> float:
    v = f.get(key, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return float(default)
    return float(v)


class BaseYardageModel:
    market = "yards"

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.residual_model = None   # optional sklearn model on residuals

    def opportunity(self, f: dict) -> dict:
        raise NotImplementedError

    def efficiency(self, f: dict) -> dict:
        raise NotImplementedError

    def predict(self, f: dict, lines: list[float] | None = None) -> dict:
        opp = self.opportunity(f)
        eff = self.efficiency(f)
        mu = max(opp["expected_opportunities"] * eff["expected_yards_per_opp"], 0.0)

        if self.residual_model is not None:
            mu = max(mu + self._residual_correction(f), 0.0)

        unc = _g(f, "role_uncertainty", 0.10)
        sd = predictive_sd(self.market, mu, unc)
        pz = zero_inflation_prob(self.market, opp["expected_opportunities"])
        dist = YardageDistribution(self.market, mu, sd, pz)

        out = {
            "market": self.market,
            "projection": round(mu, 1),
            "median": dist.median(),
            "sd": round(sd, 1),
            "distribution": dist.family,
            "p_zero": round(pz, 4),
            **intervals(dist),
            "opportunity": {k: round(v, 4) if isinstance(v, (int, float)) else v
                            for k, v in opp.items()},
            "efficiency": {k: round(v, 4) if isinstance(v, (int, float)) else v
                           for k, v in eff.items()},
        }
        if lines:
            out["lines"] = [line_probabilities(dist, ln) for ln in lines]
        out["_dist"] = dist
        return out

    def _residual_correction(self, f: dict) -> float:
        from ..features import feature_vector
        return float(self.residual_model.predict(feature_vector(f).reshape(1, -1))[0])

    def fit_residuals(self, X: np.ndarray, y_residual: np.ndarray, alpha: float = 3.0):
        """
        Optional second stage. Fit a ridge model on (actual - structural
        projection). Keeps the interpretable structural model as the backbone
        and lets the learner pick up whatever it systematically misses.
        """
        from sklearn.linear_model import Ridge
        self.residual_model = Ridge(alpha=alpha).fit(X, y_residual)
        return {"n": int(len(y_residual)),
                "r2": float(self.residual_model.score(X, y_residual))}


class PassingYardsModel(BaseYardageModel):
    market = "passing_yards"

    def opportunity(self, f: dict) -> dict:
        dropbacks = _g(f, "proj_team_dropbacks", 36.0)
        share = _g(f, "pass_att_share", 0.92)
        sack_rate = _g(f, "sack_rate", 0.065)
        # a shakier line means more sacks, which removes attempts
        sack_rate = float(np.clip(sack_rate - 0.012 * _g(f, "d_ol_passblock_z", 0.0),
                                  0.02, 0.13))
        attempts = dropbacks * share * (1 - sack_rate)
        return {"expected_opportunities": max(attempts, 0.0),
                "team_dropbacks": dropbacks,
                "dropback_share": share,
                "sack_rate": sack_rate}

    def efficiency(self, f: dict) -> dict:
        ypa = _g(f, "ypa", self.cfg.league["ypa"])
        ypa = ypa * _g(f, "pass_matchup_mult", 1.0) * _g(f, "weather_penalty", 1.0)
        ypa += _g(f, "ypa_unit_delta", 0.0)
        ypa = float(np.clip(ypa, 4.6, 10.2))
        return {"expected_yards_per_opp": ypa,
                "matchup_mult": _g(f, "pass_matchup_mult", 1.0),
                "unit_delta": _g(f, "ypa_unit_delta", 0.0),
                "weather_penalty": _g(f, "weather_penalty", 1.0)}


class RushingYardsModel(BaseYardageModel):
    market = "rushing_yards"

    def opportunity(self, f: dict) -> dict:
        team_carries = _g(f, "proj_team_carries", 25.0)
        share = _g(f, "rush_share", 0.35)
        share = float(np.clip(share + _g(f, "rush_share_script_adj", 0.0), 0.0, 0.92))
        carries = team_carries * share
        if f.get("position") == "QB":
            carries = team_carries * min(share, 0.22)
        return {"expected_opportunities": max(carries, 0.0),
                "team_carries": team_carries,
                "rush_share": share,
                "script_adj": _g(f, "rush_share_script_adj", 0.0)}

    def efficiency(self, f: dict) -> dict:
        ypc = _g(f, "ypc", self.cfg.league["ypc"])
        ypc = ypc * _g(f, "rush_matchup_mult", 1.0) + _g(f, "ypc_unit_delta", 0.0)
        ypc = float(np.clip(ypc, 2.9, 6.4))
        return {"expected_yards_per_opp": ypc,
                "matchup_mult": _g(f, "rush_matchup_mult", 1.0),
                "unit_delta": _g(f, "ypc_unit_delta", 0.0)}


class ReceivingYardsModel(BaseYardageModel):
    market = "receiving_yards"

    def opportunity(self, f: dict) -> dict:
        team_att = _g(f, "proj_team_pass_attempts", 34.0)
        share = _g(f, "target_share", 0.14)
        share *= _g(f, "matchup_share_mult", 1.0)
        share = float(np.clip(share, 0.0, 0.42))
        targets = team_att * share
        return {"expected_opportunities": max(targets, 0.0),
                "team_pass_attempts": team_att,
                "target_share": share,
                "route_participation": _g(f, "route_participation", np.nan)}

    def efficiency(self, f: dict) -> dict:
        ypt = _g(f, "yptarget", self.cfg.league["yptarget"])

        # air-yards cross-check: heavy air-yards share relative to target share
        # implies a deeper role than raw ypt history captures
        ays = f.get("air_yards_share")
        ts = f.get("target_share")
        ay_corr = 1.0
        if ays is not None and ts and not pd.isna(ays) and ts > 0.02:
            ratio = float(ays) / float(ts)
            ay_corr = float(np.clip(1.0 + 0.16 * (ratio - 1.0), 0.88, 1.16))

        ypt = ypt * _g(f, "pass_matchup_mult", 1.0) \
                  * _g(f, "matchup_eff_mult", 1.0) \
                  * _g(f, "weather_penalty", 1.0) * ay_corr
        ypt += _g(f, "yptarget_unit_delta", 0.0)
        ypt = float(np.clip(ypt, 3.6, 14.5))
        return {"expected_yards_per_opp": ypt,
                "matchup_mult": _g(f, "pass_matchup_mult", 1.0),
                "wr_cb_mult": _g(f, "matchup_eff_mult", 1.0),
                "air_yards_correction": ay_corr,
                "qb_unit_delta": _g(f, "yptarget_unit_delta", 0.0)}


MODEL_REGISTRY = {
    "passing_yards": PassingYardsModel,
    "rushing_yards": RushingYardsModel,
    "receiving_yards": ReceivingYardsModel,
}
