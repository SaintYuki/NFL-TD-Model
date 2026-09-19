"""
Fetch advanced inputs (NGS, FTN, PFR advstats, Sleeper) into data/advanced/.

    python scripts/fetch_advanced.py --season 2026 --prior-season 2025 \
        --through-week 2 --root data

    # verify Sleeper reachability separately (it is the one untested source)
    python scripts/fetch_advanced.py --selftest-sleeper

Outputs, all under data/advanced/:
    ngs_players.csv         per-player NGS (separation, CPOE, RYOE, ...)
    pfr_player_eff.csv      per-player yards before/after contact, broken tkls
    pfr_def_units.csv       team pressure + coverage allowed
    pfr_coverage_cbs.csv    per-team top corners with allowed numbers
    team_scheme.csv         FTN play-action / no-huddle / blitz / box rates
    depth_injury.csv        Sleeper depth chart order + injury status
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflprops.advanced import (  # noqa: E402
    ngs_player_summary, pfr_player_efficiency, pfr_defense_units,
    pfr_top_coverage_defenders, ftn_team_tendencies, sleeper_players,
)


def selftest_sleeper():
    try:
        df = sleeper_players()
    except Exception as e:
        print(f"FAILED to reach Sleeper: {type(e).__name__}: {e}")
        print("Sleeper needs no API key; if this fails it is network, not auth.")
        return 1
    print(f"OK -- {len(df)} players with a gsis id")
    have_depth = df["depth_chart_order"].notna().sum()
    have_inj = df["injury_status"].notna().sum()
    print(f"  depth_chart_order present: {have_depth}")
    print(f"  injury_status present:     {have_inj}")
    print(df[df.depth_chart_order == 1][["player_name", "team", "position",
                                         "depth_chart_order"]].head(8).to_string(index=False))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-sleeper", action="store_true")
    ap.add_argument("--season", type=int)
    ap.add_argument("--prior-season", type=int)
    ap.add_argument("--through-week", type=int, default=None)
    ap.add_argument("--root", default="data")
    args = ap.parse_args()

    if args.selftest_sleeper:
        sys.exit(selftest_sleeper())
    if not (args.season and args.prior_season):
        ap.error("--season and --prior-season required")

    out = os.path.join(args.root, "advanced")
    os.makedirs(out, exist_ok=True)

    def write(df, name):
        if df is None or not len(df):
            print(f"[warn] {name}: empty, skipped", file=sys.stderr)
            return
        df.to_csv(os.path.join(out, name), index=False)
        print(f"wrote {name}: {df.shape}", file=sys.stderr)

    # NGS: prefer current season once it has data, else prior
    ngs_cur = ngs_player_summary(args.season, args.through_week)
    ngs_use = ngs_cur if len(ngs_cur) > 20 else ngs_player_summary(args.prior_season)
    write(ngs_use, "ngs_players.csv")

    # id crosswalk for PFR joins
    from nflprops.advanced import _get
    import nflprops.advanced as adv
    ids = pd.read_csv("https://raw.githubusercontent.com/dynastyprocess/data/"
                      "master/files/db_playerids.csv")
    write(pfr_player_efficiency(args.prior_season, ids), "pfr_player_eff.csv")
    write(pfr_defense_units(args.prior_season), "pfr_def_units.csv")
    write(pfr_top_coverage_defenders(args.prior_season), "pfr_coverage_cbs.csv")

    # FTN needs pbp for the team join
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fnd", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "fetch_nflverse_data.py"))
        fnd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fnd)
        pbp = fnd._pbp(args.prior_season)
        write(ftn_team_tendencies(args.prior_season, pbp), "team_scheme.csv")
    except Exception as e:
        print(f"[warn] FTN scheme table failed: {type(e).__name__}: {e}", file=sys.stderr)

    # Sleeper depth chart + injuries
    try:
        sl = sleeper_players()
        write(sl, "depth_injury.csv")
    except Exception as e:
        print(f"[warn] Sleeper failed ({type(e).__name__}: {e}) -- depth chart and "
              f"injury adjustments will be skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
