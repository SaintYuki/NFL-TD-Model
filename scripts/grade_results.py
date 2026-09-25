"""
Grade past recommendations against actual results.

    python scripts/grade_results.py --season 2026 --weeks 1,2,3 --root data

WHY THIS IS THE MOST VALUABLE THING TO BUILD NEXT
-------------------------------------------------
Everything else in this project measures whether projections look reasonable.
This measures whether they MAKE MONEY, which is a different question and the
only one that settles anything.

The specific thing it answers: does edge size predict profit? A model can be
well calibrated on average and still lose, because the contracts where it
disagrees most with the market are exactly the ones where it is most likely
to be wrong rather than right. Bucketing realized ROI by edge size tells you
where -- if anywhere -- the model's disagreement is worth money.

The expected finding, based on how this model has behaved so far, is that
small edges (2-5pp) are noise and lose to the spread, while only large,
driver-supported edges clear. If that holds, the practical output is a
minimum edge threshold, which is worth more than another feature.

HOW SETTLEMENT WORKS
--------------------
Kalshi player props are "X or more". A contract settles YES when the actual
stat is >= the strike. Entry price is the ASK for a YES buy or (1 - bid) for
a NO buy, matching what evaluate_quote assumed, so realized ROI reflects
prices actually transactable at the time rather than the midpoint.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MARKET_STAT = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
}


def load_actuals(root: str, season: int) -> pd.DataFrame:
    path = os.path.join(root, "current", "player_gamelog.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    gl = pd.read_csv(path)
    return gl[gl["season"] == season] if "season" in gl.columns else gl


def settle(contract: dict, actual_row: pd.Series) -> int | None:
    """1 if the YES contract settled true, 0 if false, None if ungradeable."""
    market = contract.get("market")
    strike = contract.get("market_strike")
    if strike is None:
        return None
    if market == "anytime_td":
        tds = float(actual_row.get("rush_td", 0) or 0) + float(actual_row.get("rec_td", 0) or 0)
        return int(tds >= float(strike))
    stat = MARKET_STAT.get(market)
    if stat is None or stat not in actual_row.index:
        return None
    val = actual_row.get(stat)
    if pd.isna(val):
        return None
    return int(float(val) >= float(strike))


def grade(contracts: list[dict], actuals: pd.DataFrame, week: int) -> list[dict]:
    wk = actuals[actuals["week"] == week]
    by_pid = {r["player_id"]: r for _, r in wk.iterrows()}
    out = []
    for c in contracts:
        side = c.get("recommended_side")
        if side not in ("yes", "no"):
            continue
        if not c.get("liquid"):
            continue
        row = by_pid.get(c.get("player_id"))
        if row is None:
            continue          # did not play / no stat line
        settled_yes = settle(c, row)
        if settled_yes is None:
            continue

        entry = c.get("entry_price")
        if entry is None or entry <= 0 or entry >= 1:
            continue
        won = (settled_yes == 1) if side == "yes" else (settled_yes == 0)
        # binary contract: stake `entry`, return 1.0 on a win
        profit = (1.0 - entry) if won else -entry

        out.append({
            "week": week,
            "player_name": c.get("player_name"),
            "market": c.get("market"),
            "strike": c.get("market_strike"),
            "side": side,
            "edge": c.get("edge"),
            "entry_price": entry,
            "model_prob": c.get("model_probability"),
            "market_prob": c.get("market_implied_probability"),
            "settled_yes": settled_yes,
            "won": int(won),
            "profit_per_unit_stake": profit / entry,
            "profit_abs": profit,
            "stake": entry,
        })
    return out


def report(graded: list[dict]):
    if not graded:
        print("nothing gradeable yet -- need archived kalshi snapshots AND "
              "played games for those weeks")
        return
    df = pd.DataFrame(graded)
    print(f"graded contracts: {len(df)}")
    print(f"  win rate          : {100*df['won'].mean():.1f}%")
    roi = df["profit_abs"].sum() / df["stake"].sum()
    print(f"  ROI (stake-weighted): {100*roi:+.1f}%")
    print(f"  total staked      : {df['stake'].sum():.2f} units")
    print(f"  net               : {df['profit_abs'].sum():+.2f} units")
    print()

    print("ROI BY EDGE BUCKET  (does edge size predict profit?)")
    bins = [0, 0.05, 0.08, 0.12, 0.20, 1.0]
    labels = ["3.5-5pp", "5-8pp", "8-12pp", "12-20pp", "20pp+"]
    df["bucket"] = pd.cut(df["edge"].abs(), bins=bins, labels=labels)
    for b, g in df.groupby("bucket", observed=True):
        if not len(g):
            continue
        r = g["profit_abs"].sum() / g["stake"].sum()
        print(f"  {str(b):10} n={len(g):4}  win={100*g['won'].mean():5.1f}%  "
              f"ROI={100*r:+6.1f}%")
    print()

    print("BY MARKET")
    for m, g in df.groupby("market"):
        r = g["profit_abs"].sum() / g["stake"].sum()
        print(f"  {m:18} n={len(g):4}  win={100*g['won'].mean():5.1f}%  ROI={100*r:+6.1f}%")
    print()

    print("BY SIDE")
    for sdx, g in df.groupby("side"):
        r = g["profit_abs"].sum() / g["stake"].sum()
        print(f"  {sdx:4} n={len(g):4}  win={100*g['won'].mean():5.1f}%  ROI={100*r:+6.1f}%")
    print()

    # calibration: did things the model called 60% happen 60% of the time?
    print("CALIBRATION (model probability vs realized)")
    df["pbin"] = pd.cut(df["model_prob"], [0, .2, .35, .5, .65, .8, 1.0])
    for b, g in df.groupby("pbin", observed=True):
        if len(g) < 5:
            continue
        print(f"  model {str(b):12} n={len(g):4}  predicted={g['model_prob'].mean():.3f}  "
              f"actual={g['settled_yes'].mean():.3f}")
    print()

    n = len(df)
    if n < 100:
        print(f"NOTE: {n} graded contracts is too few to conclude anything. "
              "Treat these numbers as directional until several hundred "
              "accumulate -- a 55% win rate on 40 bets is indistinguishable "
              "from luck.")
    else:
        best = df.groupby("bucket", observed=True).apply(
            lambda g: g["profit_abs"].sum() / g["stake"].sum(), include_groups=False)
        pos = [str(b) for b, v in best.items() if v > 0]
        print("READ: profitable edge buckets so far: "
              + (", ".join(pos) if pos else "none"))
        print("If small buckets are negative and large ones positive, raise "
              "MIN_EDGE_TO_FLAG in config.py to the first profitable bucket.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", default="", help="comma list, e.g. 1,2,3")
    ap.add_argument("--root", default="data")
    ap.add_argument("--hist", default="docs/data/history")
    ap.add_argument("--out", default=None, help="optional CSV of graded bets")
    args = ap.parse_args()

    actuals = load_actuals(args.root, args.season)
    if not len(actuals):
        print("no current-season gamelog found; run fetch_nflverse_data current first")
        sys.exit(1)

    weeks = ([int(w) for w in args.weeks.split(",") if w.strip()]
             if args.weeks else sorted(actuals["week"].unique()))

    graded = []
    for wk in weeks:
        path = os.path.join(args.hist, f"kalshi_{args.season}_wk{wk}.json")
        if not os.path.exists(path):
            print(f"  [skip] no archived market snapshot for week {wk}", file=sys.stderr)
            continue
        try:
            contracts = json.load(open(path))
        except Exception as e:
            print(f"  [skip] week {wk}: {e}", file=sys.stderr)
            continue
        g = grade(contracts, actuals, wk)
        print(f"  week {wk}: {len(g)} gradeable contracts", file=sys.stderr)
        graded.extend(g)

    print()
    report(graded)

    if args.out and graded:
        pd.DataFrame(graded).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
