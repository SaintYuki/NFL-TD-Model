"""
Canonical dataset structure. Every table the pipeline consumes or produces is
declared here as {column_name: (dtype, description)} so the loaders can
validate incoming files and the docs stay in sync with the code.

Run python -m nflprops.schema to print a markdown data dictionary.
"""

from typing import Dict, Tuple

Col = Tuple[str, str]  # (dtype, description)


# ---------------------------------------------------------------------------
# PLAYER SEASON TABLE  (one row per player-season; source: prior season)
# ---------------------------------------------------------------------------
PLAYER_SEASON: Dict[str, Col] = {
    "player_id":            ("str",   "Stable ID (gsis_id / pfr_id). Primary key with season."),
    "player_name":          ("str",   "Display name."),
    "season":               ("int",   "Season year, e.g. 2024."),
    "team":                 ("str",   "Team abbreviation the player finished the season with."),
    "position":             ("str",   "QB / RB / WR / TE."),
    "age":                  ("float", "Age on Sept 1 of the season."),
    "games":                ("int",   "Games played."),
    "snap_share":           ("float", "Offensive snaps / team offensive snaps."),
    "route_participation":  ("float", "Routes run / team dropbacks. Best single usage stat for pass catchers."),
    "target_share":         ("float", "Targets / team targets."),
    "targets_per_game":     ("float", "Raw targets per game played."),
    "air_yards_share":      ("float", "Player air yards / team air yards. Drives yards more than target share alone."),
    "adot":                 ("float", "Average depth of target."),
    "catch_rate":           ("float", "Receptions / targets."),
    "yptarget":             ("float", "Receiving yards / target."),
    "yac_per_rec":          ("float", "Yards after catch per reception."),
    "rush_share":           ("float", "Carries / team carries."),
    "carries_per_game":     ("float", "Raw carries per game."),
    "ypc":                  ("float", "Rushing yards / carry."),
    "rz_target_share":      ("float", "Targets inside the 20 / team RZ targets."),
    "rz_rush_share":        ("float", "Carries inside the 20 / team RZ carries."),
    "inside5_rush_share":   ("float", "Carries inside the 5 / team inside-5 carries. Dominant ATD driver for RBs."),
    "inside10_target_share":("float", "Targets inside the 10 / team inside-10 targets."),
    "rush_td":              ("int",   "Rushing touchdowns."),
    "rec_td":               ("int",   "Receiving touchdowns."),
    "pass_att_share":       ("float", "QB only. Dropbacks / team dropbacks."),
    "ypa":                  ("float", "QB only. Passing yards / attempt."),
    "sack_rate":            ("float", "QB only. Sacks / dropbacks."),
    "pass_td":              ("int",   "QB only. Passing touchdowns."),
    "receiving_yards":      ("float", "Season receiving yards."),
    "rushing_yards":        ("float", "Season rushing yards."),
    "passing_yards":        ("float", "Season passing yards."),
}

# ---------------------------------------------------------------------------
# TEAM SEASON TABLE (offense) - one row per team-season
# ---------------------------------------------------------------------------
TEAM_SEASON_OFFENSE: Dict[str, Col] = {
    "team":                 ("str",   "Team abbreviation."),
    "season":               ("int",   "Season year."),
    "games":                ("int",   "Games played."),
    "plays_per_game":       ("float", "Offensive plays per game (excl. kneels/spikes)."),
    "pace_sec_per_play":    ("float", "Seconds per play, neutral game script (win prob 20-80%)."),
    "pass_rate":            ("float", "Dropbacks / plays."),
    "proe":                 ("float", "Pass rate over expectation, from a down/distance/score model."),
    "rz_pass_rate":         ("float", "Pass rate inside the 20."),
    "rz_trips_per_game":    ("float", "Red zone trips per game."),
    "inside5_plays_pg":     ("float", "Snaps inside the opponent 5 per game."),
    "points_per_game":      ("float", "Points scored per game."),
    "off_td_per_game":      ("float", "Offensive touchdowns per game (rush + rec)."),
    "rush_td_share":        ("float", "Share of offensive TDs that were rushing TDs."),
    "off_epa_per_play":     ("float", "Offensive EPA per play."),
    "team_pass_att_pg":     ("float", "Pass attempts per game."),
    "team_rush_att_pg":     ("float", "Rush attempts per game."),
    "team_pass_yds_pg":     ("float", "Passing yards per game."),
    "team_rush_yds_pg":     ("float", "Rushing yards per game."),
    "ol_passblock_grade":   ("float", "0-100 unit pass-block grade (PFF or equivalent)."),
    "ol_runblock_grade":    ("float", "0-100 unit run-block grade."),
    "ol_snaps_returning":   ("float", "Fraction of prior-season OL snaps returning."),
    "wr_corps_grade":       ("float", "0-100 aggregate receiver-room grade."),
    "qb_grade":             ("float", "0-100 grade of the projected starting QB."),
    "offensive_coordinator":("str",   "OC name. Used to detect scheme continuity."),
    "head_coach":           ("str",   "HC name."),
    "play_caller":          ("str",   "Whoever actually calls plays."),
    "scheme_run":           ("str",   "zone or gap or mixed."),
    "scheme_pass":          ("str",   "westcoast or airraid or playaction or spread or mixed."),
}

# ---------------------------------------------------------------------------
# TEAM SEASON TABLE (defense)
# ---------------------------------------------------------------------------
TEAM_SEASON_DEFENSE: Dict[str, Col] = {
    "team":                     ("str",   "Team abbreviation."),
    "season":                   ("int",   "Season year."),
    "games":                    ("int",   "Games played."),
    "def_epa_per_play_allowed": ("float", "EPA per play allowed. Negative = good defense."),
    "def_epa_per_pass_allowed": ("float", "EPA per dropback allowed."),
    "def_epa_per_rush_allowed": ("float", "EPA per rush allowed."),
    "def_ypa_allowed":          ("float", "Yards per pass attempt allowed."),
    "def_ypc_allowed":          ("float", "Yards per carry allowed."),
    "def_pass_yds_pg_allowed":  ("float", "Passing yards per game allowed."),
    "def_rush_yds_pg_allowed":  ("float", "Rushing yards per game allowed."),
    "def_plays_pg_faced":       ("float", "Opponent plays per game faced (pace of opponents + own pace)."),
    "def_pace_sec_allowed":     ("float", "Seconds per play allowed."),
    "def_rz_td_rate_allowed":   ("float", "Opponent TD rate on red zone trips."),
    "def_rush_td_pg_allowed":   ("float", "Rushing TDs allowed per game."),
    "def_pass_td_pg_allowed":   ("float", "Passing TDs allowed per game."),
    "def_pressure_rate":        ("float", "Pressure rate generated."),
    "def_light_box_rate":       ("float", "Rate of 6-or-fewer defenders in the box."),
    "def_man_rate":             ("float", "Man coverage rate. Drives WR/CB matchup leverage."),
    "def_zone_rate":            ("float", "Zone coverage rate."),
    "def_vs_wr_out":            ("float", "EPA allowed to perimeter WRs (X/Z alignment)."),
    "def_vs_wr_slot":           ("float", "EPA allowed to slot receivers."),
    "def_vs_te":                ("float", "EPA allowed to tight ends."),
    "def_vs_rb_pass":           ("float", "EPA allowed to running backs in the pass game."),
    "cb1_grade":                ("float", "0-100 grade of the top corner."),
    "cb2_grade":                ("float", "0-100 grade of the second corner."),
    "slot_cb_grade":            ("float", "0-100 grade of the slot corner."),
    "shadow_cb":                ("bool",  "True if CB1 travels with the opponent WR1."),
    "defensive_coordinator":    ("str",   "DC name."),
}

# ---------------------------------------------------------------------------
# CURRENT-SEASON GAME LOGS (one row per player-game, appended weekly)
# ---------------------------------------------------------------------------
PLAYER_GAMELOG: Dict[str, Col] = {
    "player_id":        ("str",   "Player ID."),
    "season":           ("int",   "Current season."),
    "week":             ("int",   "Week number."),
    "team":             ("str",   "Team played for that week."),
    "opponent":         ("str",   "Opponent."),
    "position":         ("str",   "Position."),
    "snaps":            ("int",   "Offensive snaps."),
    "team_snaps":       ("int",   "Team offensive snaps."),
    "routes":           ("int",   "Routes run."),
    "team_dropbacks":   ("int",   "Team dropbacks."),
    "targets":          ("int",   "Targets."),
    "team_targets":     ("int",   "Team targets."),
    "receptions":       ("int",   "Receptions."),
    "receiving_yards":  ("float", "Receiving yards."),
    "air_yards":        ("float", "Air yards on targets."),
    "team_air_yards":   ("float", "Team air yards."),
    "carries":          ("int",   "Carries."),
    "team_carries":     ("int",   "Team carries."),
    "rushing_yards":    ("float", "Rushing yards."),
    "pass_attempts":    ("int",   "Pass attempts (QB)."),
    "passing_yards":    ("float", "Passing yards (QB)."),
    "rz_targets":       ("int",   "Red zone targets."),
    "team_rz_targets":  ("int",   "Team red zone targets."),
    "rz_carries":       ("int",   "Red zone carries."),
    "team_rz_carries":  ("int",   "Team red zone carries."),
    "inside5_carries":  ("int",   "Inside-5 carries."),
    "team_inside5_carries": ("int", "Team inside-5 carries."),
    "rush_td":          ("int",   "Rushing TDs."),
    "rec_td":           ("int",   "Receiving TDs."),
    "pass_td":          ("int",   "Passing TDs."),
    "anytime_td":       ("int",   "1 if the player scored a rush or rec TD (label for ATD model)."),
}

# ---------------------------------------------------------------------------
# ROSTER CHANGE TABLE (hand-maintained + feed-driven; the key new-season input)
# ---------------------------------------------------------------------------
ROSTER_CHANGES: Dict[str, Col] = {
    "player_id":            ("str",   "Player ID."),
    "season":               ("int",   "Season the change applies to."),
    "position":             ("str",   "Current position. Fills in players missing from prior_season.csv (rookies, players with no meaningful prior-season stats)."),
    "prior_team":           ("str",   "Team last season. Empty for rookies."),
    "new_team":             ("str",   "Team this season."),
    "changed_team":         ("bool",  "True if prior_team != new_team."),
    "rookie":               ("bool",  "No prior NFL season."),
    "new_oc":               ("bool",  "Team hired a new offensive coordinator / play caller."),
    "new_hc_offense":       ("bool",  "New head coach who sets offensive identity."),
    "scheme_change_major":  ("bool",  "Run or pass scheme family changed."),
    "new_qb":               ("bool",  "Different projected starting QB than last season."),
    "depth_chart_rank_prior":("int",  "Positional depth rank last season (1 = starter)."),
    "depth_chart_rank_new": ("int",   "Projected positional depth rank this season."),
    "depth_chart_volatility":("float","0-1. Camp/beat-report disagreement about the role."),
    "returning_from_ir":    ("bool",  "Missed 6+ games last season with injury."),
    "usage_shift_tags":     ("str",   "Semicolon-joined keys from config.USAGE_SHIFT."),
    "manual_share_override":("float", "Optional. Hard-set the projected usage share; NaN to ignore."),
    "notes":                ("str",   "Free text for the analyst."),
}

# ---------------------------------------------------------------------------
# WEEKLY SLATE / GAME ENVIRONMENT (from the market)
# ---------------------------------------------------------------------------
GAME_ENVIRONMENT: Dict[str, Col] = {
    "season":           ("int",   "Season."),
    "week":             ("int",   "Week."),
    "team":             ("str",   "Team."),
    "opponent":         ("str",   "Opponent."),
    "home":             ("bool",  "True if at home."),
    "spread":           ("float", "Team spread. POSITIVE = underdog by that many points."),
    "total":            ("float", "Game total."),
    "implied_total":    ("float", "total/2 - spread/2. Team implied points."),
    "dome":             ("bool",  "Indoor or closed roof."),
    "wind_mph":         ("float", "Forecast wind."),
    "precip_prob":      ("float", "Forecast precipitation probability."),
}

# ---------------------------------------------------------------------------
# SPORTSBOOK LINES
# ---------------------------------------------------------------------------
PROP_LINES: Dict[str, Col] = {
    "season":       ("int",   "Season."),
    "week":         ("int",   "Week."),
    "player_id":    ("str",   "Player ID."),
    "market":       ("str",   "anytime_td or passing_yards or rushing_yards or receiving_yards."),
    "line":         ("float", "The posted number. NaN for anytime_td."),
    "over_odds":    ("int",   "American odds for over / yes."),
    "under_odds":   ("int",   "American odds for under / no."),
    "book":         ("str",   "Sportsbook name."),
    "timestamp":    ("str",   "ISO timestamp of the quote."),
}

ALL_SCHEMAS = {
    "player_season": PLAYER_SEASON,
    "team_season_offense": TEAM_SEASON_OFFENSE,
    "team_season_defense": TEAM_SEASON_DEFENSE,
    "player_gamelog": PLAYER_GAMELOG,
    "roster_changes": ROSTER_CHANGES,
    "game_environment": GAME_ENVIRONMENT,
    "prop_lines": PROP_LINES,
}


def required_columns(name: str):
    return list(ALL_SCHEMAS[name].keys())


def validate(df, name: str, strict: bool = False):
    """Check a dataframe against a schema. Returns the list of missing columns."""
    missing = [c for c in ALL_SCHEMAS[name] if c not in df.columns]
    if missing and strict:
        raise ValueError(f"{name} is missing columns: {missing}")
    return missing


def data_dictionary_markdown() -> str:
    out = []
    for name, cols in ALL_SCHEMAS.items():
        out.append(f"\n### {name}\n")
        out.append("| column | dtype | description |")
        out.append("|---|---|---|")
        for c, (dt, desc) in cols.items():
            out.append(f"| {c} | {dt} | {desc} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(data_dictionary_markdown())
