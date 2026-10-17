"""
Closing-line value and bet-timing analysis.

    python scripts/clv_analysis.py --season 2026 --week 4

WHY CLV MATTERS MORE THAN ROI RIGHT NOW
---------------------------------------
ROI on prop bets needs something like 8-10 weeks before the confidence
interval excludes zero. Closing-line value converges in 2-3, because it
measures whether the market moved TOWARD your position rather than whether
your bets happened to win.

If you consistently beat the close, the edge is real even while ROI is still
noise. If you consistently lose to the close, no amount of patience will
rescue the ROI -- you are simply paying worse prices than the market's final
opinion, and the losses are not bad luck.

WHAT THIS MEASURES
------------------
For every contract the model flagged, it compares the price at each snapshot
against the price at the LAST snapshot before kickoff (the proxy for close).

    CLV (yes side) = closing_prob - entry_prob
    CLV (no side)  = entry_prob - closing_prob

Positive means the market moved your way after you would have bet.

THE TIMING QUESTION
-------------------
Because snapshots are stamped, the same contract can be evaluated at several
moments in the week. Bucketing CLV by how many hours before kickoff the
snapshot was taken answers the practical question directly: is it better to
bet Wednesday, Friday, or Sunday morning?

Two honest cautions. Early-week prices on player props are often thin and
wide, so apparent early CLV can be an artifact of a bad quote rather than a
real edge. And this measures the MODEL's flagged plays, not your actual
bets, unless you log those separately.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


def load_snapshots(hist_dir: str, season: int, week: int) -> list[tuple[datetime, list]]:
    pat = os.path.join(hist_dir, "snapshots", f"kalshi_{season}_wk{week}_*.json")
    out = []
    for path in sorted(glob.glob(pat)):
        try:
            rows = json.load(open(path))
        except Exception:
            continue
        ts = None
        for r in rows:
            if r.get("fetched_at"):
                try:
                    ts = datetime.fromisoformat(r["fetched_at"].replace("Z", "+00:00"))
                except Exception:
                    pass
                break
        if ts is None:
            stamp = os.path.basename(path).rsplit("_", 1)[-1].replace(".json", "")
            try:
                ts = datetime.strptime(stamp, "%Y%m%dT%H%M%S")
            except Exception:
                continue
        out.append((ts, rows))
    return sorted(out, key=lambda x: x[0])


def load_kickoffs(root: str, season: int, week: int) -> dict:
    """team -> kickoff datetime (UTC) for this week."""
    path = os.path.join(root, "game_environment.csv")
    if not os.path.exists(path):
        return {}
    try:
        e = pd.read_csv(path)
    except Exception:
        return {}
    e = e[(e.get("season") == season) & (e.get("week") == week)]
    if "kickoff_utc" not in e.columns:
        return {}
    out = {}
    for _, r in e.iterrows():
        k = r.get("kickoff_utc")
        if isinstance(k, str) and k:
            try:
                out[r["team"]] = pd.Timestamp(k).to_pydatetime()
            except Exception:
                pass
    return out


def build_clv(snaps, kickoffs: dict | None = None) -> pd.DataFrame:
    """
    Compute CLV using a PER-TEAM close.

    A single global "last snapshot" is wrong whenever snapshots straddle a
    kickoff. The Sunday-morning fetch ran while the London game was underway,
    so for those contracts the final snapshot was a LIVE price converging on
    a known scoreline -- which showed up as -73pp of fake CLV on Mariota and
    McGowan.

    So each contract's close is the last snapshot strictly BEFORE that team's
    kickoff, and any contract with no pre-kickoff close is dropped rather
    than measured against a live price. Timing is measured as hours before
    that team's own kickoff, which also makes the timing bands comparable
    across Thursday, London and Sunday games.
    """
    kickoffs = kickoffs or {}
    if len(snaps) < 2:
        return pd.DataFrame()

    def team_kick(r):
        return kickoffs.get(r.get("team")) or kickoffs.get(r.get("opponent"))

    # index every snapshot by ticker
    indexed = []
    for ts, rows in snaps:
        indexed.append((ts, {r.get("market_ticker"): r for r in rows if r.get("market_ticker")}))

    # per-ticker: the last snapshot strictly before its kickoff is the close
    closes = {}
    for ts, by_tkr in indexed:
        for tkr, r in by_tkr.items():
            ko = team_kick(r)
            ts_cmp = ts if ts.tzinfo else (ts.replace(tzinfo=ko.tzinfo) if ko else ts)
            if ko is not None and ts_cmp >= ko:
                continue                      # live price, never a valid close
            prev = closes.get(tkr)
            if prev is None or ts_cmp > prev[0]:
                closes[tkr] = (ts_cmp, r.get("market_implied_probability"))

    recs, skipped_live, skipped_noclose = [], 0, 0
    for ts, by_tkr in indexed:
        for tkr, r in by_tkr.items():
            side = r.get("recommended_side")
            if side not in ("yes", "no") or not r.get("liquid"):
                continue
            ko = team_kick(r)
            ts_cmp = ts if ts.tzinfo else (ts.replace(tzinfo=ko.tzinfo) if ko else ts)
            if ko is not None and ts_cmp >= ko:
                skipped_live += 1
                continue
            c = closes.get(tkr)
            if c is None or c[1] is None:
                skipped_noclose += 1
                continue
            close_ts, closing = c
            if ts_cmp >= close_ts:
                continue                      # this IS the close
            entry = r.get("market_implied_probability")
            if entry is None:
                continue
            clv = (closing - entry) if side == "yes" else (entry - closing)
            hrs = ((ko - ts_cmp).total_seconds() / 3600.0 if ko is not None
                   else (close_ts - ts_cmp).total_seconds() / 3600.0)
            recs.append({
                "ticker": tkr, "player": r.get("player_name"),
                "team": r.get("team"), "market": r.get("market"), "side": side,
                "hours_before_close": hrs,
                "entry_prob": entry, "closing_prob": closing,
                "clv": clv, "edge": r.get("edge"),
                "cluster": f"{r.get('player_id')}|{r.get('market')}",
            })

    if skipped_live:
        print(f"excluded {skipped_live} contract-snapshots priced after kickoff")
    if skipped_noclose:
        print(f"excluded {skipped_noclose} with no valid pre-kickoff close")
    return pd.DataFrame(recs)


def clustered_mean(df, col="clv"):
    """Mean with a cluster-aware SE: strikes on one player move together."""
    if not len(df):
        return None
    g = df.groupby("cluster")[col].mean()
    n = len(g)
    if n < 2:
        return None
    m = float(g.mean())
    se = float(g.std(ddof=1) / np.sqrt(n))
    return {"mean": m, "se": se, "n_clusters": n, "n": len(df),
            "lo": m - 1.96 * se, "hi": m + 1.96 * se}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--hist", default="docs/data/history")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    snaps = load_snapshots(args.hist, args.season, args.week)
    print(f"snapshots found: {len(snaps)}")
    for ts, rows in snaps:
        print(f"  {ts:%Y-%m-%d %H:%M}  {len(rows)} contracts")
    if len(snaps) < 2:
        print("\nNeed at least 2 snapshots to measure CLV.")
        print("Run fetch_kalshi several times through the week -- every run now")
        print("saves a stamped copy, so just fetching Wed / Fri / Sun builds this.")
        return

    kickoffs = load_kickoffs("data", args.season, args.week)
    if kickoffs:
        print(f"kickoff times loaded for {len(kickoffs)} teams")
    else:
        print("no kickoff times found -- live-game contamination NOT filtered")
    df = build_clv(snaps, kickoffs)
    if not len(df):
        print("\nno comparable contracts across snapshots")
        return

    print()
    overall = clustered_mean(df)
    print("OVERALL CLV (positive = market moved toward the model)")
    if overall is None:
        print("  not enough independent clusters to compute a mean "
              f"(rows={len(df)}). Fetch more snapshots across the week.")
        return
    print(f"  mean CLV {100*overall['mean']:+.2f}pp   95% CI "
          f"[{100*overall['lo']:+.2f}, {100*overall['hi']:+.2f}]pp")
    print(f"  n={overall['n']} contracts across {overall['n_clusters']} clusters")
    verdict = ("BEATING the close -- edge is real even if ROI is still noisy"
               if overall["lo"] > 0 else
               "LOSING to the close -- prices are worse than the market's final"
               " opinion, and ROI will not rescue that"
               if overall["hi"] < 0 else
               "INCONCLUSIVE -- CI contains zero, need more snapshots/weeks")
    print(f"  verdict: {verdict}")
    print()

    print("CLV BY TIMING (the practical question: when should you bet?)")
    bands = [(0, 6, "0-6h before close"), (6, 24, "6-24h"),
             (24, 72, "1-3 days"), (72, 999, "3+ days out")]
    for lo, hi, lab in bands:
        sub = df[(df.hours_before_close >= lo) & (df.hours_before_close < hi)]
        c = clustered_mean(sub)
        if c:
            print(f"  {lab:18} n={c['n']:5} cl={c['n_clusters']:4}  "
                  f"CLV {100*c['mean']:+6.2f}pp  "
                  f"[{100*c['lo']:+6.2f},{100*c['hi']:+6.2f}]")
    print()

    print("CLV BY MARKET")
    for m, g in df.groupby("market"):
        c = clustered_mean(g)
        if c:
            print(f"  {m:18} n={c['n']:5} cl={c['n_clusters']:4}  "
                  f"CLV {100*c['mean']:+6.2f}pp")
    print()

    print("CLV BY SIDE")
    for sd, g in df.groupby("side"):
        c = clustered_mean(g)
        if c:
            print(f"  {sd:4} n={c['n']:5} cl={c['n_clusters']:4}  "
                  f"CLV {100*c['mean']:+6.2f}pp")
    print()

    big = df.reindex(df.clv.abs().sort_values(ascending=False).index).head(8)
    print("LARGEST LINE MOVES (where the market disagreed most after the fact)")
    for _, r in big.iterrows():
        print(f"  {str(r['player'])[:20]:22} {r['market']:16} {r['side']:4} "
              f"{r['entry_prob']:.3f} -> {r['closing_prob']:.3f}  CLV {100*r['clv']:+6.1f}pp")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
