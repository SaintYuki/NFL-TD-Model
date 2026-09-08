"""Sanity tests. python scripts/test_invariants.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from nflprops import DataStore, ProjectionEngine
from nflprops.data.blend import current_weight, blend_value
from nflprops.probability import (implied_prob, american_odds, devig_two_way,
                                  YardageDistribution)

fails = []
def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond: fails.append(name)

# odds round trip
check("odds round trip", all(american_odds(implied_prob(o)) == o
                             for o in (-300, -150, -110, 120, 250, 600)))
po, pu = devig_two_way(-110, -110)
check("devig sums to 1", abs(po + pu - 1) < 1e-9)

# blend behavior
check("week1 uses no current data", current_weight("target_share", 0) == 0.0)
check("K is the 50/50 crossover", abs(current_weight("target_share", 4) - 0.5) < 1e-9)
check("usage stabilizes faster than efficiency",
      current_weight("rush_share", 5) > current_weight("ypc", 5))
check("rookie regresses fully to baseline",
      abs(blend_value("target_share", None, 0.30, 0.155, 0, 0.0) - 0.155) < 1e-9)
check("missing league baseline falls back to prior",
      abs(blend_value("ypc", None, 4.9, np.nan, 0, 0.8) - 4.9) < 1e-9)

# distributions
d = YardageDistribution("receiving_yards", 60.0, 35.0, 0.03)
check("survival is monotone decreasing", d.sf(20) > d.sf(60) > d.sf(120))
check("cdf and sf are complementary", abs(d.cdf(55) + d.sf(55) - 1) < 1e-9)
check("80 pct interval is wider than 50 pct",
      (d.interval(.8)[1] - d.interval(.8)[0]) > (d.interval(.5)[1] - d.interval(.5)[0]))
dn = YardageDistribution("passing_yards", 250.0, 66.0)
check("normal median equals mean", abs(dn.median() - 250.0) < 0.5)

# end to end
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
if os.path.exists(root):
    for wk in (1, 5):
        store = DataStore(root, 2026, wk)
        eng = ProjectionEngine(store)
        out = eng.project_slate()
        errs = [o for o in out if "error" in o]
        check(f"week {wk} slate runs clean", len(errs) == 0)

        u = eng.team_adjusted_usage().reset_index()
        rec = u[u.position.isin(["RB", "WR", "TE"])]
        tot = rec.groupby("team")["target_share"].sum()
        check(f"week {wk} target shares normalized", bool(((tot - 1.0).abs() < 1e-6).all()))

        qb = [o for o in out if o.get("position") == "QB"][0]
        py = qb["markets"]["passing_yards"]
        check(f"week {wk} QB passing yards plausible", 120 < py["projection"] < 400)
        check(f"week {wk} CI brackets the projection",
              py["ci_80"][0] < py["projection"] < py["ci_80"][1])
        atd = [o["markets"]["anytime_td"]["probability"] for o in out]
        check(f"week {wk} ATD probs in range", all(0 <= p <= 0.97 for p in atd))
        check(f"week {wk} ATD mean plausible", 0.08 < float(np.mean(atd)) < 0.35)

        rb = [o for o in out if o.get("position") == "RB"
              and o["markets"].get("rushing_yards")]
        check(f"week {wk} no rushing market for pure receivers",
              all("rushing_yards" not in o["markets"]
                  for o in out if o.get("position") == "TE"))

print(f"\n{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
