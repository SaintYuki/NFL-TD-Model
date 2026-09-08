"""
Figures out which NFL season and week "today" falls in, so the scheduled
GitHub Actions workflow does not need the week hardcoded and edited every
Wednesday.

MAINTENANCE: add next year's kickoff date to SEASON_WEEK1_KICKOFF once the
league releases the schedule (usually mid-May). That is the only edit this
file needs, once a year.

Usage:
    python scripts/current_nfl_week.py
    -> prints
       season=2026
       week=3

The two lines are printed in `KEY=value` form so a GitHub Actions step can
capture them directly with `python scripts/current_nfl_week.py >> $GITHUB_OUTPUT`.
"""

import argparse
import datetime

# The Wednesday/Thursday the regular season kicks off, one entry per season.
# Source: nfl.com schedule release.
SEASON_WEEK1_KICKOFF = {
    2025: datetime.date(2025, 9, 4),
    2026: datetime.date(2026, 9, 9),
}

REGULAR_SEASON_WEEKS = 18


def current_season_and_week(today: datetime.date = None) -> tuple[int, int]:
    today = today or datetime.date.today()
    seasons = sorted(SEASON_WEEK1_KICKOFF)

    # latest season whose kickoff is on or before today
    candidate = None
    for s in seasons:
        if SEASON_WEEK1_KICKOFF[s] <= today:
            candidate = s
    if candidate is None:
        # before the earliest configured kickoff at all
        return seasons[0], 1

    kickoff = SEASON_WEEK1_KICKOFF[candidate]
    days_since = (today - kickoff).days
    week = days_since // 7 + 1
    if week <= REGULAR_SEASON_WEEKS:
        return candidate, week

    # that season's regular season has already ended (we're in the playoffs
    # or offseason). If the next season's kickoff is on file, point at its
    # Week 1 instead of reporting a stale season 54 weeks deep -- this is
    # what makes a run in late August/early September, before kickoff,
    # produce something sensible.
    idx = seasons.index(candidate)
    if idx + 1 < len(seasons):
        return seasons[idx + 1], 1
    return candidate, REGULAR_SEASON_WEEKS


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD, defaults to today")
    args = ap.parse_args()
    d = datetime.date.fromisoformat(args.date) if args.date else None
    season, week = current_season_and_week(d)
    print(f"season={season}")
    print(f"week={week}")
