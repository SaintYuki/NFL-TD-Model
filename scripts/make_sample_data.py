"""
Generates a synthetic but structurally correct dataset so the pipeline can be
run end to end before real feeds are wired up. Replace with nfl_data_py /
nflverse pulls in production; the column names already match schema.py.

    python scripts/make_sample_data.py --root data --season 2025
"""

import argparse
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

TEAMS = ["BUF", "KC", "SF", "PHI", "DAL", "MIA", "BAL", "DET",
         "CIN", "GB", "HOU", "NYJ", "LAC", "SEA", "MIN", "TB"]


def team_offense(season):
    rows = []
    for t in TEAMS:
        rows.append({
            "team": t, "season": season, "games": 17,
            "plays_per_game": rng.normal(63, 3.2),
            "pace_sec_per_play": rng.normal(27.6, 1.6),
            "pass_rate": np.clip(rng.normal(0.575, 0.045), 0.45, 0.70),
            "proe": rng.normal(0, 0.035),
            "rz_pass_rate": np.clip(rng.normal(0.545, 0.06), 0.40, 0.68),
            "rz_trips_per_game": rng.normal(3.4, 0.6),
            "inside5_plays_pg": rng.normal(1.5, 0.4),
            "points_per_game": rng.normal(22.4, 4.4),
            "off_td_per_game": rng.normal(2.28, 0.5),
            "rush_td_share": np.clip(rng.normal(0.395, 0.08), 0.2, 0.6),
            "off_epa_per_play": rng.normal(0.0, 0.08),
            "team_pass_att_pg": rng.normal(34, 3.5),
            "team_rush_att_pg": rng.normal(26, 3.0),
            "team_pass_yds_pg": rng.normal(228, 30),
            "team_rush_yds_pg": rng.normal(115, 22),
            "ol_passblock_grade": rng.normal(65, 8),
            "ol_runblock_grade": rng.normal(65, 8),
            "ol_snaps_returning": np.clip(rng.normal(0.70, 0.18), 0.2, 1.0),
            "wr_corps_grade": rng.normal(65, 8),
            "qb_grade": rng.normal(65, 9),
            "offensive_coordinator": f"OC_{t}",
            "head_coach": f"HC_{t}",
            "play_caller": f"OC_{t}",
            "scheme_run": rng.choice(["zone", "gap", "mixed"]),
            "scheme_pass": rng.choice(["westcoast", "playaction", "spread", "mixed"]),
        })
    return pd.DataFrame(rows)


def team_defense(season):
    rows = []
    for t in TEAMS:
        epa_p = rng.normal(0.06, 0.075)
        epa_r = rng.normal(-0.08, 0.055)
        rows.append({
            "team": t, "season": season, "games": 17,
            "def_epa_per_play_allowed": 0.55 * epa_p + 0.45 * epa_r,
            "def_epa_per_pass_allowed": epa_p,
            "def_epa_per_rush_allowed": epa_r,
            "def_ypa_allowed": 7.05 + 8 * (epa_p - 0.06) + rng.normal(0, 0.25),
            "def_ypc_allowed": 4.35 + 5 * (epa_r + 0.08) + rng.normal(0, 0.20),
            "def_pass_yds_pg_allowed": rng.normal(228, 26),
            "def_rush_yds_pg_allowed": rng.normal(115, 20),
            "def_plays_pg_faced": rng.normal(63, 3.0),
            "def_pace_sec_allowed": rng.normal(27.6, 1.3),
            "def_rz_td_rate_allowed": np.clip(rng.normal(0.56, 0.07), 0.38, 0.75),
            "def_rush_td_pg_allowed": np.clip(rng.normal(0.90, 0.28), 0.2, 1.9),
            "def_pass_td_pg_allowed": np.clip(rng.normal(1.38, 0.35), 0.4, 2.6),
            "def_pressure_rate": np.clip(rng.normal(0.34, 0.045), 0.2, 0.48),
            "def_light_box_rate": np.clip(rng.normal(0.32, 0.07), 0.15, 0.55),
            "def_man_rate": np.clip(rng.normal(0.30, 0.09), 0.1, 0.6),
            "def_zone_rate": None,
            "def_vs_wr_out": rng.normal(0, 0.06),
            "def_vs_wr_slot": rng.normal(0, 0.06),
            "def_vs_te": rng.normal(0, 0.06),
            "def_vs_rb_pass": rng.normal(0, 0.06),
            "cb1_grade": rng.normal(68, 8),
            "cb2_grade": rng.normal(62, 8),
            "slot_cb_grade": rng.normal(63, 8),
            "shadow_cb": bool(rng.random() < 0.25),
            "defensive_coordinator": f"DC_{t}",
        })
    df = pd.DataFrame(rows)
    df["def_zone_rate"] = 1 - df["def_man_rate"]
    return df


def players(season):
    rows = []
    for t in TEAMS:
        # QB
        rows.append(dict(player_id=f"{t}_QB1", player_name=f"{t} QB1", season=season,
                         team=t, position="QB", age=27, games=17,
                         snap_share=0.97, pass_att_share=np.clip(rng.normal(0.93, 0.05), 0.6, 1.0),
                         ypa=rng.normal(7.05, 0.7), sack_rate=np.clip(rng.normal(0.065, 0.018), 0.02, 0.12),
                         rush_share=np.clip(rng.normal(0.09, 0.05), 0.01, 0.25),
                         ypc=rng.normal(4.6, 0.9), pass_td=int(rng.normal(24, 7)),
                         rush_td=int(max(0, rng.normal(2.5, 2))), rec_td=0,
                         passing_yards=rng.normal(3900, 700), rushing_yards=rng.normal(280, 220),
                         receiving_yards=0,
                         rz_rush_share=np.clip(rng.normal(0.10, 0.06), 0, 0.35),
                         inside5_rush_share=np.clip(rng.normal(0.12, 0.08), 0, 0.4)))
        # RBs
        for i, base in enumerate([0.55, 0.28, 0.12]):
            rows.append(dict(player_id=f"{t}_RB{i+1}", player_name=f"{t} RB{i+1}", season=season,
                             team=t, position="RB", age=25, games=17,
                             snap_share=np.clip(base + rng.normal(0, 0.06), 0.05, 0.9),
                             rush_share=np.clip(base + rng.normal(0, 0.07), 0.03, 0.85),
                             carries_per_game=base * 26,
                             ypc=rng.normal(4.35, 0.5),
                             target_share=np.clip(rng.normal(0.10 - 0.03 * i, 0.03), 0.01, 0.22),
                             yptarget=rng.normal(6.2, 1.0), adot=rng.normal(0.4, 1.2),
                             catch_rate=np.clip(rng.normal(0.74, 0.06), 0.5, 0.9),
                             route_participation=np.clip(rng.normal(0.45 - 0.12 * i, 0.1), 0.05, 0.8),
                             air_yards_share=np.clip(rng.normal(0.04, 0.02), 0.005, 0.12),
                             rz_rush_share=np.clip(base + rng.normal(0, 0.10), 0.02, 0.9),
                             inside5_rush_share=np.clip(base + rng.normal(0.05, 0.12), 0.02, 0.95),
                             rz_target_share=np.clip(rng.normal(0.07, 0.03), 0.01, 0.2),
                             inside10_target_share=np.clip(rng.normal(0.06, 0.03), 0.005, 0.2),
                             rush_td=int(max(0, rng.normal(8 * base, 2.5))),
                             rec_td=int(max(0, rng.normal(1.5 * base, 1))),
                             rushing_yards=base * 26 * 17 * 4.3, receiving_yards=rng.normal(280, 150),
                             passing_yards=0))
        # WRs
        for i, base in enumerate([0.245, 0.175, 0.115]):
            rows.append(dict(player_id=f"{t}_WR{i+1}", player_name=f"{t} WR{i+1}", season=season,
                             team=t, position="WR", age=26, games=17,
                             snap_share=np.clip(0.85 - 0.15 * i + rng.normal(0, 0.06), 0.3, 0.98),
                             route_participation=np.clip(0.86 - 0.16 * i + rng.normal(0, 0.06), 0.25, 0.97),
                             target_share=np.clip(base + rng.normal(0, 0.03), 0.04, 0.36),
                             air_yards_share=np.clip(base * 1.15 + rng.normal(0, 0.04), 0.03, 0.45),
                             adot=rng.normal(10.6, 2.4), catch_rate=np.clip(rng.normal(0.645, 0.06), 0.45, 0.82),
                             yptarget=rng.normal(8.1, 1.2), yac_per_rec=rng.normal(4.6, 1.2),
                             rush_share=0.005,
                             rz_target_share=np.clip(base + rng.normal(0.01, 0.05), 0.02, 0.42),
                             inside10_target_share=np.clip(base + rng.normal(0.02, 0.06), 0.02, 0.45),
                             rz_rush_share=0.0, inside5_rush_share=0.0,
                             rec_td=int(max(0, rng.normal(7 - 2 * i, 2.5))), rush_td=0,
                             receiving_yards=base * 34 * 17 * 8.0, rushing_yards=rng.normal(15, 15),
                             passing_yards=0,
                             player_grade=rng.normal(68, 8),
                             slot_rate=np.clip(rng.normal(0.3 + 0.25 * i, 0.2), 0.05, 0.95)))
        # TE
        rows.append(dict(player_id=f"{t}_TE1", player_name=f"{t} TE1", season=season,
                         team=t, position="TE", age=27, games=17,
                         snap_share=np.clip(rng.normal(0.75, 0.1), 0.3, 0.95),
                         route_participation=np.clip(rng.normal(0.68, 0.12), 0.2, 0.9),
                         target_share=np.clip(rng.normal(0.135, 0.035), 0.04, 0.27),
                         air_yards_share=np.clip(rng.normal(0.11, 0.035), 0.02, 0.25),
                         adot=rng.normal(7.9, 1.8), catch_rate=np.clip(rng.normal(0.70, 0.06), 0.5, 0.86),
                         yptarget=rng.normal(7.3, 1.1), rush_share=0.0,
                         rz_target_share=np.clip(rng.normal(0.16, 0.05), 0.03, 0.35),
                         inside10_target_share=np.clip(rng.normal(0.18, 0.06), 0.03, 0.4),
                         rz_rush_share=0.0, inside5_rush_share=0.0,
                         rec_td=int(max(0, rng.normal(5, 2))), rush_td=0,
                         receiving_yards=rng.normal(620, 200), rushing_yards=0, passing_yards=0,
                         player_grade=rng.normal(66, 7), slot_rate=rng.uniform(0.4, 0.8)))
    return pd.DataFrame(rows)


def roster_changes(pl, season):
    rows = []
    for _, p in pl.iterrows():
        changed = rng.random() < 0.16
        new_team = rng.choice([t for t in TEAMS if t != p["team"]]) if changed else p["team"]
        rank = {"QB": 1, "TE": 1}.get(p["position"], int(p["player_id"][-1]))
        promo = rng.random() < 0.10
        rows.append({
            "player_id": p["player_id"], "season": season,
            "prior_team": p["team"], "new_team": new_team,
            "changed_team": bool(changed), "rookie": False,
            "new_oc": bool(rng.random() < 0.22),
            "new_hc_offense": bool(rng.random() < 0.10),
            "new_dc": bool(rng.random() < 0.20),
            "scheme_change_major": bool(rng.random() < 0.12),
            "new_qb": bool(rng.random() < 0.18),
            "depth_chart_rank_prior": rank,
            "depth_chart_rank_new": max(1, rank - 1) if promo else rank,
            "depth_chart_volatility": float(np.clip(rng.normal(0.12, 0.12), 0, 0.8)),
            "returning_from_ir": bool(rng.random() < 0.08),
            "usage_shift_tags": "moved_to_starter" if promo and rank > 1 else "",
            "manual_share_override": np.nan,
            "notes": "",
        })
    return pd.DataFrame(rows)


def game_environment(season, week):
    matchups, pool = [], list(TEAMS)
    rng.shuffle(pool)
    for i in range(0, len(pool), 2):
        matchups.append((pool[i], pool[i + 1]))
    rows = []
    for home, away in matchups:
        total = float(np.clip(rng.normal(44.5, 4.5), 34, 56))
        spread_home = float(np.round(rng.normal(-1.5, 5.5) * 2) / 2)
        dome = bool(rng.random() < 0.25)
        for team, opp, sp, at_home in ((home, away, spread_home, True),
                                       (away, home, -spread_home, False)):
            rows.append({"season": season, "week": week, "team": team, "opponent": opp,
                         "home": at_home, "spread": sp, "total": total,
                         "implied_total": total / 2 - sp / 2, "dome": dome,
                         "wind_mph": 0.0 if dome else float(np.clip(rng.normal(7, 5), 0, 25)),
                         "precip_prob": 0.0 if dome else float(np.clip(rng.normal(0.15, 0.2), 0, 0.9))})
    return pd.DataFrame(rows)


def prop_lines(pl, env, season, week):
    teams_now = set(env["team"])
    rows = []
    for _, p in pl.iterrows():
        if p["team"] not in teams_now:
            continue
        pos = p["position"]
        markets = []
        if pos == "QB":
            markets = [("passing_yards", float(np.round(rng.normal(232, 35) / 5) * 5 + 0.5)),
                       ("rushing_yards", float(np.round(rng.normal(24, 12)) + 0.5)),
                       ("anytime_td", np.nan)]
        elif pos == "RB":
            base = p["rush_share"] * 26 * 4.3
            markets = [("rushing_yards", float(np.round(base / 5) * 5 + 0.5)),
                       ("receiving_yards", float(np.round(p["target_share"] * 34 * 6.2) + 0.5)),
                       ("anytime_td", np.nan)]
        else:
            markets = [("receiving_yards", float(np.round(p["target_share"] * 34 * 8.0 / 5) * 5 + 0.5)),
                       ("anytime_td", np.nan)]
        for m, line in markets:
            odds = int(rng.choice([-125, -115, -110, -105, 100, 120, 160, 240, 320]))
            rows.append({"season": season, "week": week, "player_id": p["player_id"],
                         "market": m, "line": line,
                         "over_odds": odds if m == "anytime_td" else int(rng.choice([-120, -115, -110, -105])),
                         "under_odds": int(-odds if m == "anytime_td" and odds > 0 else rng.choice([-120, -115, -110, -105])),
                         "book": "DemoBook", "timestamp": "2026-09-04T12:00:00Z"})
    return pd.DataFrame(rows)


def gamelogs(pl, season, weeks):
    """Simulate current-season game logs for weeks 1..weeks-1."""
    prows, torows, tdrows = [], [], []
    for wk in range(1, weeks):
        for t in TEAMS:
            plays = int(np.clip(rng.normal(63, 5), 48, 80))
            dropbacks = int(plays * np.clip(rng.normal(0.575, 0.06), 0.4, 0.72))
            carries = plays - dropbacks
            sacks = int(dropbacks * 0.065)
            pass_att = dropbacks - sacks
            tp = pl[pl["team"] == t]
            team_targets = pass_att
            team_air = pass_att * 8.4
            rz_t = int(np.clip(rng.normal(3.4, 1.2), 0, 8))
            rz_c = int(np.clip(rng.normal(2.8, 1.2), 0, 8))
            i5 = int(np.clip(rng.normal(1.5, 1.0), 0, 5))
            off_td = int(np.clip(rng.normal(2.3, 1.3), 0, 7))
            rush_td = int(np.clip(rng.binomial(off_td, 0.40), 0, off_td))
            torows.append(dict(team=t, season=season, week=wk, plays=plays,
                               dropbacks=dropbacks, pass_attempts=pass_att, carries=carries,
                               pace_sec_per_play=float(rng.normal(27.6, 2)),
                               proe=float(rng.normal(0, 0.05)),
                               rz_dropbacks=int(rz_t * 0.55), rz_plays=rz_t + rz_c,
                               off_td=off_td, rush_td=rush_td, points=off_td * 7 + 3 * rng.integers(0, 3),
                               epa_per_play=float(rng.normal(0, 0.12))))
            tdrows.append(dict(team=t, season=season, week=wk,
                               epa_per_play_allowed=float(rng.normal(0, 0.12)),
                               epa_per_pass_allowed=float(rng.normal(0.06, 0.14)),
                               epa_per_rush_allowed=float(rng.normal(-0.08, 0.11)),
                               pass_yds_allowed=float(rng.normal(228, 60)), pass_att_faced=int(rng.normal(34, 5)),
                               rush_yds_allowed=float(rng.normal(115, 40)), carries_faced=int(rng.normal(26, 5)),
                               rz_td_allowed=int(rng.integers(0, 4)), rz_trips_faced=int(rng.integers(1, 6)),
                               pass_td_allowed=int(rng.integers(0, 4)), rush_td_allowed=int(rng.integers(0, 3)),
                               plays_faced=int(rng.normal(63, 5)),
                               epa_vs_wr_out=float(rng.normal(0, 0.12)), epa_vs_wr_slot=float(rng.normal(0, 0.12)),
                               epa_vs_te=float(rng.normal(0, 0.12)), epa_vs_rb_pass=float(rng.normal(0, 0.12))))

            for _, p in tp.iterrows():
                def sv(key, dflt=0.0):
                    v = p.get(key, dflt)
                    return dflt if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
                tgt = int(rng.poisson(max(sv("target_share") * team_targets, 0.01)))
                car = int(rng.poisson(max(sv("rush_share") * carries, 0.01)))
                if p["position"] == "QB":
                    pa, py = pass_att, float(rng.normal(sv("ypa", 7.0) * pass_att, 55))
                    tgt = 0
                else:
                    pa, py = 0, 0.0
                rec = int(rng.binomial(tgt, np.clip(sv("catch_rate", 0.65), 0.3, 0.9))) if tgt else 0
                ry = float(max(0, rng.normal(rec * max(sv("yptarget", 8.0) / max(np.clip(sv("catch_rate", .65), .3, .9), .3), 6), 22))) if rec else 0.0
                rush_y = float(max(-5, rng.normal(car * sv("ypc", 4.3), 18))) if car else 0.0
                rtd = int(rng.random() < 0.10 * sv("inside5_rush_share") * 5)
                ctd = int(rng.random() < 0.09 * sv("rz_target_share") * 5)
                prows.append(dict(player_id=p["player_id"], season=season, week=wk, team=t,
                                  opponent=rng.choice([x for x in TEAMS if x != t]),
                                  position=p["position"],
                                  snaps=int(sv("snap_share", 0.5) * plays), team_snaps=plays,
                                  routes=int(sv("route_participation", 0.5) * dropbacks),
                                  team_dropbacks=dropbacks,
                                  targets=tgt, team_targets=team_targets, receptions=rec,
                                  receiving_yards=ry, air_yards=float(tgt * sv("adot", 8.0)),
                                  team_air_yards=team_air,
                                  carries=car, team_carries=carries, rushing_yards=rush_y,
                                  pass_attempts=pa, passing_yards=py,
                                  rz_targets=int(rng.binomial(rz_t, np.clip(sv("rz_target_share"), 0, 1))),
                                  team_rz_targets=rz_t,
                                  rz_carries=int(rng.binomial(rz_c, np.clip(sv("rz_rush_share"), 0, 1))),
                                  team_rz_carries=rz_c,
                                  inside5_carries=int(rng.binomial(i5, np.clip(sv("inside5_rush_share"), 0, 1))),
                                  team_inside5_carries=i5,
                                  rush_td=rtd, rec_td=ctd, pass_td=int(rng.integers(0, 3)) if pa else 0,
                                  anytime_td=int(rtd or ctd)))
    return pd.DataFrame(prows), pd.DataFrame(torows), pd.DataFrame(tdrows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=1)
    args = ap.parse_args()

    prior = args.season - 1
    os.makedirs(os.path.join(args.root, "prior"), exist_ok=True)
    os.makedirs(os.path.join(args.root, "current"), exist_ok=True)

    pl = players(prior)
    pl["targets_per_game"] = pl["target_share"].fillna(0) * 34.0
    pl["carries_per_game"] = pl["carries_per_game"].fillna(pl["rush_share"].fillna(0) * 26.0)
    pl.to_csv(f"{args.root}/prior/player_season.csv", index=False)
    team_offense(prior).to_csv(f"{args.root}/prior/team_season_offense.csv", index=False)
    team_defense(prior).to_csv(f"{args.root}/prior/team_season_defense.csv", index=False)
    rc = roster_changes(pl, args.season)
    rc.to_csv(f"{args.root}/roster_changes.csv", index=False)

    env_path = f"{args.root}/game_environment.csv"
    env = game_environment(args.season, args.week)
    if os.path.exists(env_path):
        old = pd.read_csv(env_path)
        old = old[~((old.season == args.season) & (old.week == args.week))]
        env = pd.concat([old, env], ignore_index=True)
    env.to_csv(env_path, index=False)
    env = env[(env.season == args.season) & (env.week == args.week)]

    pl_now = pl.copy()
    pl_now["team"] = pl_now["player_id"].map(rc.set_index("player_id")["new_team"]).fillna(pl_now["team"])
    lines_path = f"{args.root}/prop_lines.csv"
    lines = prop_lines(pl_now, env, args.season, args.week)
    if os.path.exists(lines_path):
        old = pd.read_csv(lines_path)
        old = old[~((old.season == args.season) & (old.week == args.week))]
        lines = pd.concat([old, lines], ignore_index=True)
    lines.to_csv(lines_path, index=False)

    if args.week > 1:
        g, to, td = gamelogs(pl_now, args.season, args.week)
        g.to_csv(f"{args.root}/current/player_gamelog.csv", index=False)
        to.to_csv(f"{args.root}/current/team_gamelog_offense.csv", index=False)
        td.to_csv(f"{args.root}/current/team_gamelog_defense.csv", index=False)
    else:
        for f in ("player_gamelog", "team_gamelog_offense", "team_gamelog_defense"):
            p = f"{args.root}/current/{f}.csv"
            if os.path.exists(p):
                os.remove(p)
    print(f"sample data written to {args.root}/ for season {args.season} week {args.week}")


if __name__ == "__main__":
    main()
