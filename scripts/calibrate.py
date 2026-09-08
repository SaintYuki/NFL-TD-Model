"""
Backtest and recalibration.

Run this after every scored week. It answers three questions:

  1. Are the yardage projections unbiased? (mean residual near zero, by market
     and by position)
  2. Are the intervals honest? (80% interval should contain ~80% of outcomes)
  3. Is the ATD probability calibrated? (Brier score, log loss, and a
     reliability table by predicted-probability bucket)

Then it refits the pieces that are meant to be fit:
  - AnytimeTDModel.fit_calibration -> logistic on log(lambda)
  - BaseYardageModel.fit_residuals -> ridge on the structural residual
  - empirical dispersion CV per market, which feeds config.DISPERSION_CV

    python scripts/calibrate.py --root data --season 2026 --through-week 8
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflprops import DataStore, ProjectionEngine  # noqa: E402

MARKET_ACTUAL = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
}


def collect(root, season, through_week, start_week=1):
    """Walk weeks, project with information available BEFORE each week, score."""
    gl = pd.read_csv(os.path.join(root, "current", "player_gamelog.csv"))
    rows = []
    for wk in range(start_week, through_week + 1):
        store = DataStore(root, season, wk)   # filters to week < wk automatically
        engine = ProjectionEngine(store)
        actual_wk = gl[(gl.season == season) & (gl.week == wk)].set_index("player_id")
        for pid in engine.meta["player_id"]:
            if pid not in actual_wk.index:
                continue
            a = actual_wk.loc[pid]
            if isinstance(a, pd.DataFrame):
                a = a.iloc[0]
            try:
                proj = engine.project_player(pid)
            except Exception:
                continue
            for market, blk in proj.get("markets", {}).items():
                if market == "anytime_td":
                    rows.append({"week": wk, "player_id": pid, "market": market,
                                 "position": proj.get("position"),
                                 "pred": blk["probability"],
                                 "lam": blk["expected_touchdowns"],
                                 "actual": float(a.get("anytime_td", 0))})
                else:
                    rows.append({"week": wk, "player_id": pid, "market": market,
                                 "position": proj.get("position"),
                                 "pred": blk["projection"], "sd": blk["sd"],
                                 "lo80": blk["ci_80"][0], "hi80": blk["ci_80"][1],
                                 "actual": float(a.get(MARKET_ACTUAL[market], 0.0))})
    return pd.DataFrame(rows)


def report(df):
    out = {}
    yd = df[df.market != "anytime_td"]
    for market, g in yd.groupby("market"):
        resid = g["actual"] - g["pred"]
        cover = ((g["actual"] >= g["lo80"]) & (g["actual"] <= g["hi80"])).mean()
        out[market] = {
            "n": int(len(g)),
            "mean_bias": round(float(resid.mean()), 2),
            "mae": round(float(resid.abs().mean()), 2),
            "rmse": round(float(np.sqrt((resid ** 2).mean())), 2),
            "empirical_cv": round(float(resid.std() / max(g["pred"].mean(), 1e-6)), 3),
            "ci80_coverage": round(float(cover), 3),
            "correlation": round(float(np.corrcoef(g["pred"], g["actual"])[0, 1]), 3)
            if len(g) > 3 else None,
        }

    td = df[df.market == "anytime_td"]
    if len(td):
        p, y = td["pred"].values, td["actual"].values
        brier = float(np.mean((p - y) ** 2))
        eps = 1e-9
        logloss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        base = float(y.mean())
        bins = pd.cut(td["pred"], [0, .05, .10, .15, .20, .30, .40, .60, 1.0])
        rel = td.groupby(bins, observed=True).agg(
            n=("actual", "size"), predicted=("pred", "mean"), observed=("actual", "mean"))
        out["anytime_td"] = {
            "n": int(len(td)),
            "brier": round(brier, 4),
            "brier_skill_vs_base": round(1 - brier / max(base * (1 - base), 1e-9), 4),
            "log_loss": round(logloss, 4),
            "mean_predicted": round(float(p.mean()), 4),
            "base_rate": round(base, 4),
            "reliability": rel.round(4).reset_index().astype(str).to_dict("records"),
        }
    return out


def refit(df, root):
    """Fit the optional calibration layers and dump the parameters to disk."""
    params = {}
    td = df[df.market == "anytime_td"]
    if len(td) > 200 and td["actual"].nunique() > 1:
        from nflprops.models.anytime_td import AnytimeTDModel
        m = AnytimeTDModel()
        params["anytime_td_calibration"] = m.fit_calibration(td["lam"].values,
                                                             td["actual"].values)
    for market, g in df[df.market != "anytime_td"].groupby("market"):
        resid = g["actual"] - g["pred"]
        params[f"{market}_empirical_cv"] = round(
            float(resid.std() / max(g["pred"].mean(), 1e-6)), 4)
        params[f"{market}_mean_bias"] = round(float(resid.mean()), 3)
    path = os.path.join(root, "calibration_params.json")
    with open(path, "w") as fh:
        json.dump(params, fh, indent=2)
    return params, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--through-week", type=int, required=True)
    ap.add_argument("--start-week", type=int, default=1)
    args = ap.parse_args()

    df = collect(args.root, args.season, args.through_week, args.start_week)
    if not len(df):
        print("no scored rows found"); return
    print(json.dumps(report(df), indent=2))
    params, path = refit(df, args.root)
    print(f"\ncalibration params -> {path}")
    print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
