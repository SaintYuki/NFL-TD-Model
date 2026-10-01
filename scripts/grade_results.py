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


def grade(contracts: list[dict], actuals: pd.DataFrame, week: int,
          sides=("yes", "no"), min_prob: float = 0.0,
          min_edge: float = 0.0, max_edge: float = 1.0,
          tiers=None) -> list[dict]:
    """
    Grade archived contracts, optionally restricted to the population a
    given strategy would actually bet.

    Every filter here keys on information known BEFORE the bet -- side,
    model probability, edge size, playability tier. None of them touch the
    outcome, so restricting on them does not bias the result. (Filtering on
    anything outcome-adjacent would.)

    The default of grading both sides answers "does the model have edge".
    Restricting to yes-only answers "does MY strategy have edge". Those are
    different questions and were previously conflated: 64% of the graded
    set was NO plays that never get placed.
    """
    wk = actuals[actuals["week"] == week]
    by_pid = {r["player_id"]: r for _, r in wk.iterrows()}
    out = []
    for c in contracts:
        side = c.get("recommended_side")
        if side not in sides:
            continue
        if not c.get("liquid"):
            continue
        mp = c.get("model_probability")
        if mp is None or float(mp) < min_prob:
            continue          # excludes longshots when min_prob is set
        e = abs(float(c.get("edge") or 0.0))
        if e < min_edge or e > max_edge:
            continue
        if tiers and c.get("bettable") not in tiers:
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
            "player_id": c.get("player_id"),
            "cluster": f"{week}|{c.get('player_id')}",
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


def clustered_roi(df):
    """
    ROI with a standard error that respects clustering.

    Contracts on the same player in the same week settle together -- if a
    receiver gains 8 yards, every "no" on him wins at once. Treating those
    as independent observations overstates precision by roughly the square
    root of the strikes-per-player, which is what made a one-week sample
    look like 1,371 data points when it held ~300 independent ones.

    So: aggregate profit and stake per (week, player), then compute the
    standard error across CLUSTERS.
    """
    import numpy as np
    if not len(df):
        return None
    g = df.groupby("cluster").agg(profit=("profit_abs", "sum"),
                                  stake=("stake", "sum"))
    n_cl = len(g)
    stake_tot = g["stake"].sum()
    if stake_tot <= 0 or n_cl < 2:
        return None
    roi = g["profit"].sum() / stake_tot
    # per-cluster ROI contribution, weighted by that cluster's stake
    contrib = (g["profit"] - roi * g["stake"]).values
    se = float(np.sqrt((contrib ** 2).sum())) / stake_tot
    return {"roi": float(roi), "se": se, "n_contracts": int(len(df)),
            "n_clusters": int(n_cl),
            "ci_lo": float(roi - 1.96 * se), "ci_hi": float(roi + 1.96 * se)}


def report(graded: list[dict]):
    if not graded:
        print("nothing gradeable yet -- need archived kalshi snapshots AND "
              "played games for those weeks")
        return
    df = pd.DataFrame(graded)
    c = clustered_roi(df)
    print(f"graded contracts: {len(df)}"
          + (f"   independent clusters: {c['n_clusters']}" if c else ""))
    print(f"  win rate          : {100*df['won'].mean():.1f}%")
    roi = df["profit_abs"].sum() / df["stake"].sum()
    if c:
        print(f"  ROI (clustered)   : {100*c['roi']:+.1f}%  "
              f"95% CI [{100*c['ci_lo']:+.1f}%, {100*c['ci_hi']:+.1f}%]")
        verdict = ("POSITIVE, CI excludes zero" if c["ci_lo"] > 0 else
                   "NEGATIVE, CI excludes zero" if c["ci_hi"] < 0 else
                   "INCONCLUSIVE -- CI contains zero")
        print(f"  verdict           : {verdict}")
    else:
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
        cc = clustered_roi(g)
        if cc:
            print(f"  {str(b):10} n={len(g):4} cl={cc['n_clusters']:4}  "
                  f"win={100*g['won'].mean():5.1f}%  ROI={100*cc['roi']:+6.1f}% "
                  f"[{100*cc['ci_lo']:+6.1f}%,{100*cc['ci_hi']:+6.1f}%]")
        else:
            r = g["profit_abs"].sum() / g["stake"].sum()
            print(f"  {str(b):10} n={len(g):4}  win={100*g['won'].mean():5.1f}%  ROI={100*r:+6.1f}%")
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
    ap.add_argument("--side", default="both", choices=["yes", "no", "both"],
                    help="grade only the side you actually bet")
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="drop longshots below this model probability")
    ap.add_argument("--min-edge", type=float, default=0.0)
    ap.add_argument("--max-edge", type=float, default=1.0)
    ap.add_argument("--tiers", default=None,
                    help="comma list, e.g. STRONG,PLAYABLE")
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
        sides = ("yes", "no") if args.side == "both" else (args.side,)
        tiers = [t.strip() for t in args.tiers.split(",")] if args.tiers else None
        g = grade(contracts, actuals, wk, sides=sides,
                  min_prob=args.min_prob, min_edge=args.min_edge,
                  max_edge=args.max_edge, tiers=tiers)
        print(f"  week {wk}: {len(g)} gradeable contracts", file=sys.stderr)
        graded.extend(g)

    print()
    filt = [f"side={args.side}"]
    if args.min_prob > 0: filt.append(f"model_prob>={args.min_prob}")
    if args.min_edge > 0: filt.append(f"|edge|>={args.min_edge}")
    if args.max_edge < 1: filt.append(f"|edge|<={args.max_edge}")
    if args.tiers: filt.append(f"tiers={args.tiers}")
    print("POPULATION: " + ", ".join(filt))
    print()
    report(graded)

    if args.out and graded:
        pd.DataFrame(graded).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
