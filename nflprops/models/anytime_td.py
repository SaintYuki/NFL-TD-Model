"""
Anytime Touchdown model.

Structure: a Poisson intensity built top-down from the market's implied team
total, then allocated to the player by red zone usage. This beats a raw
player-level logistic regression because the scarce, noisy quantity (player TD
rate) is decomposed into a stable quantity (team scoring) times a
moderately stable quantity (goal-line share).

    E[team_off_TD]  = 0.70*(a + b*implied_total) + 0.30*(hist_td_pg * def_adj)
    E[team_rush_TD] = E[team_off_TD] * rush_share_of_td
    E[team_rec_TD]  = E[team_off_TD] * (1 - rush_share_of_td)

    lambda_rush = E[team_rush_TD] * (0.62*inside5_rush_share
                                     + 0.28*rz_rush_share
                                     + 0.10*rush_share)
    lambda_rec  = E[team_rec_TD]  * (0.45*inside10_target_share
                                     + 0.33*rz_target_share
                                     + 0.22*air_yards_share)
    lambda_qb_rush = E[team_rush_TD] * qb_designed_goalline_share

    lambda = (lambda_rush + lambda_rec) * roster_mult * matchup_mult
    P(ATD) = 1 - exp(-lambda * kappa)

The share weights above reflect how TDs actually distribute: inside-5 carries
dominate rushing scores, while receiving TDs come from a mix of goal-line
targets and deep-shot volume (which is why air yards share carries weight even
though it is not a red zone stat).

An optional logistic recalibration layer maps log(lambda) to observed outcomes
once you have labeled results, correcting any systematic bias in the Poisson
link without rebuilding the structural model.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from ..util import safe_num
from ..config import ModelConfig, DEFAULT_CONFIG, TD_POISSON_KAPPA
from ..probability import anytime_td_probability, american_odds

RUSH_TD_WEIGHTS = {"inside5_rush_share": 0.62, "rz_rush_share": 0.28, "rush_share": 0.10}
REC_TD_WEIGHTS = {"inside10_target_share": 0.45, "rz_target_share": 0.33, "air_yards_share": 0.22}


class AnytimeTDModel:
    """Structural Poisson model with an optional learned calibration layer."""

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.kappa = cfg.td_kappa
        self.calibrator = None      # sklearn LogisticRegression on log(lambda)

    # ------------------------------------------------------------------
    def rush_td_lambda(self, f: dict) -> float:
        share = sum(w * safe_num(f.get(k), 0.0) for k, w in RUSH_TD_WEIGHTS.items())
        # fall back to overall rush share when goal-line data is missing
        if share <= 0:
            share = 0.55 * safe_num(f.get("rush_share"), 0.0)
        return float(f.get("e_team_rush_td", 0.9)) * share

    def rec_td_lambda(self, f: dict) -> float:
        weights = dict(REC_TD_WEIGHTS)
        present = {k: float(f.get(k, np.nan)) for k in weights}
        if all(pd.isna(v) for v in present.values()):
            share = 0.85 * safe_num(f.get("target_share"), 0.0)
        else:
            tot_w = sum(w for k, w in weights.items() if not pd.isna(present[k]))
            share = sum(w * present[k] for k, w in weights.items()
                        if not pd.isna(present[k])) / max(tot_w, 1e-6)
        return float(f.get("e_team_rec_td", 1.35)) * share

    # ------------------------------------------------------------------
    def intensity(self, f: dict) -> dict:
        pos = f.get("position", "WR")
        lam_rush = self.rush_td_lambda(f)
        lam_rec = self.rec_td_lambda(f)

        if pos == "QB":
            # QBs only score on designed runs / scrambles; no receiving component
            lam_rec = 0.0
            lam_rush = float(f.get("e_team_rush_td", 0.9)) * \
                safe_num(f.get("qb_goalline_rush_share", f.get("rush_share")), 0.06)
        elif pos == "RB":
            lam_rec *= 0.85          # RB receiving TDs are rarer than share implies
        elif pos in ("WR", "TE"):
            lam_rush *= 0.35         # jet sweeps only

        roster_mult = float(f.get("roster_usage_mult", 1.0))
        # matchup: rushing TDs care about the run defense, receiving about pass D
        match_mult_rush = 0.5 + 0.5 * float(f.get("rush_matchup_mult", 1.0))
        match_mult_rec = 0.5 + 0.5 * float(f.get("pass_matchup_mult", 1.0))
        rz_def = float(f.get("def_rz_td_adj", 1.0))

        lam = (lam_rush * match_mult_rush + lam_rec * match_mult_rec) * rz_def
        lam = max(lam * min(roster_mult, 2.0), 0.0)
        return {
            "lambda_total": float(lam),
            "lambda_rush": float(lam_rush),
            "lambda_rec": float(lam_rec),
            "goal_line_share": safe_num(f.get("inside5_rush_share"), 0.0),
            "rz_target_share": safe_num(f.get("rz_target_share"), 0.0),
        }

    # ------------------------------------------------------------------
    def predict(self, f: dict) -> dict:
        parts = self.intensity(f)
        lam = parts["lambda_total"]
        if self.calibrator is not None:
            x = np.array([[math.log(max(lam, 1e-4))]])
            p = float(self.calibrator.predict_proba(x)[0, 1])
        else:
            p = anytime_td_probability(lam, self.kappa)

        # multi-TD probabilities straight from the Poisson
        p2 = float(1 - math.exp(-lam) * (1 + lam))
        return {
            "expected_touchdowns": round(lam, 4),
            "p_anytime_td": round(p, 4),
            "p_two_plus_td": round(max(p2, 0.0), 4),
            "fair_odds": american_odds(p),
            "components": {k: round(v, 4) for k, v in parts.items()},
        }

    # ------------------------------------------------------------------
    def fit_calibration(self, lambdas: np.ndarray, outcomes: np.ndarray):
        """
        Optional. Fit logit(p) = a + b*log(lambda) on scored weeks.
        A fitted b well below 1.0 means the structural lambdas are too spread
        out; b above 1.0 means they are too compressed.
        """
        from sklearn.linear_model import LogisticRegression
        x = np.log(np.clip(np.asarray(lambdas, dtype=float), 1e-4, None)).reshape(-1, 1)
        y = np.asarray(outcomes, dtype=int)
        self.calibrator = LogisticRegression(C=10.0, max_iter=1000).fit(x, y)
        return {
            "intercept": float(self.calibrator.intercept_[0]),
            "slope_log_lambda": float(self.calibrator.coef_[0][0]),
            "n": int(len(y)),
        }
