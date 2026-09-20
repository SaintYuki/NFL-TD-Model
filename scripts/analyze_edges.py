"""
Analyze the Kalshi edge distribution to tell systematic model bias apart from
a tail of hard-to-project bench players.

    python scripts/analyze_edges.py --file docs/data/kalshi.json

WHY THIS EXISTS
---------------
The fetch script prints the top 15 plays sorted by absolute edge. That view
shows extremes BY CONSTRUCTION, so it always looks alarming and tells you
almost nothing about calibration. A model can be perfectly calibrated and
still have fifteen large-edge outliers.

What actually matters:
  - Is the MEDIAN edge near zero? (systematic bias vs noise)
  - Is the yes/no split balanced?
  - Does bias concentrate in a subgroup (low strikes, bench players, one
    market) rather than spreading across the board?

A median edge near zero with a fat negative tail means the model is fine and
a subgroup is broken. A median edge well below zero means the model is
systematically under-projecting and NO edge on the board is trustworthy.
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np


def pct(x):
    return f"{100*x:+.1f}pp"


def summarize(rows):
    liquid = [r for r in rows if r.get("liquid") and r.get("edge") is not None]
    if not liquid:
        print("no liquid contracts with an edge")
        return

    edges = np.array([r["edge"] for r in liquid], dtype=float)
    sides = [r.get("recommended_side") for r in liquid]

    print(f"liquid contracts with an edge : {len(liquid)}")
    print(f"  median edge                 : {pct(np.median(edges))}")
    print(f"  mean edge                   : {pct(edges.mean())}")
    print(f"  share with model ABOVE market: {100*(edges > 0).mean():.1f}%")
    print(f"  actionable yes / no         : {sides.count('yes')} / {sides.count('no')}")
    print()

    print("BY MARKET")
    by = defaultdict(list)
    for r in liquid:
        by[r.get("market")].append(r["edge"])
    for m, e in sorted(by.items()):
        e = np.array(e)
        print(f"  {m:18} n={len(e):5}  median={pct(np.median(e))}  "
              f"model_above={100*(e>0).mean():5.1f}%")
    print()

    # The key cut: does bias concentrate at low strikes? Yardage strikes near
    # the bottom of the ladder are mostly bench players, where the model's
    # replacement-level floor bites hardest.
    print("YARDAGE, BY STRIKE BAND")
    bands = defaultdict(list)
    for r in liquid:
        if r.get("market") == "anytime_td":
            continue
        s = r.get("market_strike")
        if s is None:
            continue
        band = ("1) <=20" if s <= 20 else "2) 21-40" if s <= 40
                else "3) 41-75" if s <= 75 else "4) >75")
        bands[band].append(r["edge"])
    for b, e in sorted(bands.items()):
        e = np.array(e)
        print(f"  {b:12} n={len(e):5}  median={pct(np.median(e))}  "
              f"model_above={100*(e>0).mean():5.1f}%")
    print()

    print("ANYTIME TD, BY STRIKE")
    tdb = defaultdict(list)
    for r in liquid:
        if r.get("market") != "anytime_td":
            continue
        tdb[int(r.get("market_strike") or 1)].append(r["edge"])
    for k, e in sorted(tdb.items()):
        e = np.array(e)
        print(f"  {k}+ TD        n={len(e):5}  median={pct(np.median(e))}  "
              f"model_above={100*(e>0).mean():5.1f}%")
    print()

    print("WORST 10 (model far BELOW market)")
    for r in sorted(liquid, key=lambda x: x["edge"])[:10]:
        print(f"  {str(r.get('player_name'))[:22]:24} {r.get('market'):16} "
              f"strike={str(r.get('market_strike')):>6}  "
              f"model={r.get('model_probability'):.3f} mkt={r.get('market_implied_probability'):.3f} "
              f"edge={r['edge']:+.3f}")
    print()
    print("BEST 10 (model far ABOVE market)")
    for r in sorted(liquid, key=lambda x: -x["edge"])[:10]:
        print(f"  {str(r.get('player_name'))[:22]:24} {r.get('market'):16} "
              f"strike={str(r.get('market_strike')):>6}  "
              f"model={r.get('model_probability'):.3f} mkt={r.get('market_implied_probability'):.3f} "
              f"edge={r['edge']:+.3f}")
    print()

    med = float(np.median(edges))
    if abs(med) < 0.02:
        verdict = ("Median edge is near zero -- the model is broadly calibrated "
                   "against the market. Large negative outliers are a SUBGROUP "
                   "problem, not a global bias. Find the subgroup above and fix "
                   "or filter it; the rest of the board is usable.")
    elif med < -0.05:
        verdict = ("Median edge is well below zero -- the model systematically "
                   "under-projects versus the market. Do NOT trust any edge on "
                   "the board until this is closed; almost every 'no' will be "
                   "the model being wrong rather than value.")
    elif med > 0.05:
        verdict = ("Median edge is well above zero -- the model systematically "
                   "over-projects. Same warning in reverse.")
    else:
        verdict = ("Mild bias. Usable, but treat small edges as noise and only "
                   "act on large ones with a supported divergence verdict.")
    print("VERDICT:", verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="docs/data/kalshi.json")
    args = ap.parse_args()
    try:
        rows = json.load(open(args.file))
    except Exception as e:
        print(f"could not read {args.file}: {e}")
        sys.exit(1)
    summarize(rows)


if __name__ == "__main__":
    main()
