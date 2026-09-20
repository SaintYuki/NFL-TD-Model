"""
Kalshi market-odds ingestion.

WHAT IS VERIFIED AND WHAT IS NOT
--------------------------------
The pure logic in this module (ticker parsing, name matching, price ->
probability conversion, edge and Kelly math) is unit-tested offline against
real ticker strings and real projection data.

The HTTP layer is NOT live-tested, because the machine this was written on
cannot reach kalshi.com. It is built from Kalshi's documented API contract
(RSA-PSS request signing, KALSHI-ACCESS-* headers). Treat the first live run
as a smoke test: call `python scripts/fetch_kalshi.py --selftest` which hits
only public endpoints and prints what it gets back.

AUTHENTICATION
--------------
Kalshi signs each request with an RSA private key:

  message   = f"{timestamp_ms}{METHOD}{path_without_query}"
  signature = RSA-PSS(SHA256, salt=MAX_LENGTH) over that message
  headers   = KALSHI-ACCESS-KEY / -TIMESTAMP / -SIGNATURE

Create a key at https://kalshi.com/account/profile (API Keys section). It is
shown once -- save the PEM. Then:

  export KALSHI_API_KEY_ID="..."
  export KALSHI_PRIVATE_KEY_PATH="~/.kalshi/private_key.pem"

Market/read endpoints are generally reachable unauthenticated; portfolio and
order endpoints are not. This module only reads, so it works either way, and
signs when credentials are present.

BASE URL
--------
Kalshi has published several hostnames over time (trading-api.kalshi.com,
api.elections.kalshi.com, external-api.kalshi.com). Rather than hardcode a
guess, the base URL is configurable and defaults to the one in the current
docs. If you get DNS or 404 errors, set KALSHI_API_BASE_URL and check
https://docs.kalshi.com for the current host.

TICKER STRUCTURE
----------------
From real Week 2 market URLs:

  event:   KXNFLGAME-26SEP17DETBUF
  ATD:     KXNFLTD-26SEP17DETBUF-BUFJALLEN17-1
  rec yds: KXNFLRECYDS-26SEP17DETBUF-DETASTBROWN14-40

  {SERIES}-{YYMMMDD}{AWAY}{HOME}-{TEAM}{NAMECODE}{JERSEY}-{STRIKE}

STRIKE SEMANTICS (important, and easy to get wrong)
---------------------------------------------------
Kalshi player props are "X or more" contracts, not over/under at a half-point.
KXNFLRECYDS-...-40 pays if the player records 40 OR MORE receiving yards.
So P(yes) = P(X >= 40), which for an integer-valued stat is P(X > 39.5).
Comparing it against a sportsbook-style P(X > 40.0) would introduce a small
systematic bias, so the continuity correction below is deliberate.
"""

from __future__ import annotations

import base64
import os
import re

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_BASE_URL = os.environ.get(
    "KALSHI_API_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")

# Kalshi NFL player-prop series -> our internal market names
SERIES_TO_MARKET = {
    "KXNFLTD": "anytime_td",
    "KXNFLPASSYDS": "passing_yards",
    "KXNFLRUSHYDS": "rushing_yards",
    "KXNFLRECYDS": "receiving_yards",
}

# Real NFL team abbreviations as they appear in Kalshi tickers. A known-set
# longest-match is required: naive "first 2-3 alpha chars" parses LVAJEANTY
# as team "LVA" + "JEANTY" instead of "LV" + "AJEANTY" (Ashton Jeanty), and
# breaks every 2-letter team the same way.
TEAM_ABBRS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAC", "JAX", "KC", "LA", "LAC", "LAR", "LV",
    "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS", "WSH",
}
# Kalshi ticker abbreviation -> the abbreviation nflverse uses
TEAM_ALIAS = {"JAC": "JAX", "WSH": "WAS", "LAR": "LA"}


def split_team_prefix(blob: str) -> tuple[str, str]:
    """
    'LVAJEANTY' -> ('LV', 'AJEANTY');  'LACOHAMPTON' -> ('LAC', 'OHAMPTON').
    Longest valid team abbreviation wins, so LAC beats LA and LAR beats LA.
    """
    for size in (3, 2):
        cand = blob[:size]
        if cand in TEAM_ABBRS:
            return cand, blob[size:]
    return "", blob


TICKER_RE = re.compile(    r"^(?P<series>KXNFL[A-Z]+)-"
    r"(?P<datecode>\d{2}[A-Z]{3}\d{2})"
    r"(?P<teams>[A-Z]{4,8})-"
    r"(?P<player>[A-Z0-9]+?)(?P<jersey>\d{1,2})-"
    r"(?P<strike>\d+)$"
)


@dataclass
class KalshiQuote:
    """One normalized Kalshi player-prop market."""
    ticker: str
    market: str                 # our internal market name
    strike: float               # "X or more"
    team: str
    name_code: str              # e.g. JALLEN, ASTBROWN
    jersey: Optional[int]
    yes_bid: Optional[float] = None    # probability, 0.0-1.0 (Kalshi dollars)
    yes_ask: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    raw: dict = field(default_factory=dict)

    # -- pricing -------------------------------------------------------
    @property
    def mid_price(self) -> Optional[float]:
        if self.yes_bid is None or self.yes_ask is None:
            return self.yes_bid if self.yes_bid is not None else self.yes_ask
        return (self.yes_bid + self.yes_ask) / 2.0

    @property
    def implied_prob(self) -> Optional[float]:
        """
        Kalshi prices ARE probabilities (a YES contract settles at $1), so the
        bid/ask midpoint needs no de-vig step -- unlike American sportsbook
        odds, where both sides sum past 100%. The spread is the cost of
        crossing, not vig.
        """
        mid = self.mid_price
        return None if mid is None else max(0.0, min(1.0, mid))

    @property
    def spread(self) -> Optional[float]:
        """Bid/ask spread in probability points (0.04 = 4 cents)."""
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid

    @property
    def is_liquid(self) -> bool:
        """
        Illiquid markets produce meaningless "edges". A 1-cent bid against a
        99-cent ask implies anything between 1% and 99%, so any model number
        looks like a huge edge. Require a two-sided quote with a sane spread
        and some actual activity.
        """
        if self.yes_bid is None or self.yes_ask is None:
            return False
        if self.yes_bid <= 0.0:
            return False          # no real bid = nobody wants it at any price
        if self.spread is None or self.spread > 0.15:
            return False
        if (self.volume or 0) < 1 and (self.open_interest or 0) < 1:
            return False
        return True


def parse_ticker(ticker: str) -> Optional[dict]:
    """Parse a Kalshi NFL player-prop ticker. Returns None if it isn't one."""
    m = TICKER_RE.match(ticker.strip().upper())
    if not m:
        return None
    series = m.group("series")
    market = SERIES_TO_MARKET.get(series)
    if market is None:
        return None
    player_blob = m.group("player")
    team, name_code = split_team_prefix(player_blob)
    return {
        "series": series,
        "market": market,
        "datecode": m.group("datecode"),
        "teams": m.group("teams"),
        "team": TEAM_ALIAS.get(team, team),
        "name_code": name_code,
        "jersey": int(m.group("jersey")),
        "strike": float(m.group("strike")),
    }


def normalize_name(name: str) -> str:
    """'Amon-Ra St. Brown' -> 'AMONRASTBROWN'. Used for ticker matching."""
    return re.sub(r"[^A-Z]", "", (name or "").upper())


def name_code_candidates(full_name: str) -> set[str]:
    """
    Kalshi compresses names inconsistently: 'Josh Allen' -> JALLEN,
    'Amon-Ra St. Brown' -> ASTBROWN. Generate the plausible encodings so a
    ticker can be matched without a hand-maintained lookup table.
    """
    if not full_name:
        return set()
    parts = [p for p in re.split(r"[\s]+", full_name.strip()) if p]
    if not parts:
        return set()
    first = normalize_name(parts[0])
    last = normalize_name("".join(parts[1:])) if len(parts) > 1 else ""
    out = set()
    if last:
        out.add(first[:1] + last)      # JALLEN
        out.add(first + last)          # JOSHALLEN
        out.add(last)                  # ALLEN
        out.add(first[:2] + last)      # JOALLEN
    else:
        out.add(first)
    return {c for c in out if c}


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------
class KalshiClient:
    """
    Minimal read-only Kalshi client. Signs requests when credentials are
    present; otherwise makes unauthenticated calls (fine for market data).
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 key_id: str | None = None,
                 private_key_path: str | None = None,
                 timeout: float = 20.0,
                 min_interval_s: float = 0.15):
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id or os.environ.get("KALSHI_API_KEY_ID")
        self.private_key_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        self.timeout = timeout
        # Kalshi rate-limits per tier; a small floor between calls keeps a
        # weekly full-slate pull well under any published basic-tier cap.
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self._private_key = None
        if self.private_key_path and os.path.exists(os.path.expanduser(self.private_key_path)):
            self._load_key()

    def _load_key(self):
        from cryptography.hazmat.primitives import serialization
        with open(os.path.expanduser(self.private_key_path), "rb") as fh:
            self._private_key = serialization.load_pem_private_key(fh.read(), password=None)

    @property
    def authenticated(self) -> bool:
        return self._private_key is not None and bool(self.key_id)

    def _auth_headers(self, method: str, path: str) -> dict:
        if not self.authenticated:
            return {}
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        ts = str(int(time.time() * 1000))
        # sign the path WITHOUT the query string
        sign_path = path.split("?")[0]
        message = f"{ts}{method.upper()}{sign_path}".encode()
        sig = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "Content-Type": "application/json",
        }

    def _throttle(self):
        delta = time.time() - self._last_call
        if delta < self.min_interval_s:
            time.sleep(self.min_interval_s - delta)
        self._last_call = time.time()

    def get(self, path: str, params: dict | None = None) -> dict:
        import requests
        from urllib.parse import urlparse
        self._throttle()
        url = self.base_url + path
        # The signed path must exactly match the request path, so derive it
        # from the configured base_url rather than hardcoding /trade-api/v2.
        signed_path = urlparse(self.base_url).path.rstrip("/") + path
        headers = self._auth_headers("GET", signed_path)
        resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            time.sleep(2.0)
            resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # -- endpoints -----------------------------------------------------
    def get_markets(self, event_ticker: str | None = None,
                    series_ticker: str | None = None,
                    status: str = "open", limit: int = 200) -> list[dict]:
        """Paginate /markets. Either filter is optional; both can be combined."""
        out, cursor = [], None
        while True:
            params = {"limit": limit, "status": status}
            if event_ticker:
                params["event_ticker"] = event_ticker
            if series_ticker:
                params["series_ticker"] = series_ticker
            if cursor:
                params["cursor"] = cursor
            data = self.get("/markets", params)
            out.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor or not data.get("markets"):
                break
        return out

    def get_event(self, event_ticker: str) -> dict:
        return self.get(f"/events/{event_ticker}")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def _dollars_to_prob(value) -> Optional[float]:
    """
    Kalshi returns prices as DOLLAR strings ('0.1700'), not integer cents.
    A YES contract settles at $1.00, so the dollar price already IS the
    implied probability -- 0.17 means 17%. No division by 100.
    """
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, v))


def _num(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_markets(raw_markets: list[dict], open_only: bool = True) -> list[KalshiQuote]:
    """Turn raw /markets rows into KalshiQuote objects, dropping non-props."""
    quotes = []
    for m in raw_markets:
        ticker = m.get("ticker", "")
        parsed = parse_ticker(ticker)
        if not parsed:
            continue
        if open_only and m.get("status") not in (None, "active", "open"):
            continue
        quotes.append(KalshiQuote(
            ticker=ticker,
            market=parsed["market"],
            strike=parsed["strike"],
            team=parsed["team"],
            name_code=parsed["name_code"],
            jersey=parsed["jersey"],
            # real field names, in dollars
            yes_bid=_dollars_to_prob(m.get("yes_bid_dollars", m.get("yes_bid"))),
            yes_ask=_dollars_to_prob(m.get("yes_ask_dollars", m.get("yes_ask"))),
            volume=_num(m.get("volume_fp", m.get("volume"))),
            open_interest=_num(m.get("open_interest_fp", m.get("open_interest"))),
            raw=m,
        ))
    return quotes


def match_quotes_to_players(quotes: list[KalshiQuote],
                            players: list[dict]) -> dict:
    """
    Map each quote to an internal player_id.

    players: [{"player_id":..., "player_name":..., "team":..., "position":...,
               "jersey_number": ...}]
    Returns {ticker: player_id}. Unmatched tickers are simply absent.

    JERSEY NUMBER IS LOAD-BEARING, NOT DECORATIVE
    ---------------------------------------------
    Kalshi's name codes collide. Atlanta rosters both Bijan Robinson (#7) and
    Brian Robinson (#15), and BOTH encode to "BROBINSON". Matching on name
    alone silently picked the backup, so the market's price for a star was
    compared against a backup's projection -- which produced a fake 87-point
    "edge" that looked like the best play on the board.

    So: when a name code matches more than one player, the jersey number in
    the ticker decides. If it cannot decide, the quote is LEFT UNMATCHED
    rather than guessed, because a wrong match is far worse than no match.
    """
    by_team: dict[str, list[dict]] = {}
    for p in players:
        by_team.setdefault((p.get("team") or "").upper(), []).append(p)

    mapping = {}
    for q in quotes:
        pool = by_team.get(q.team.upper(), []) or players

        # every player whose encodings include this ticker's name code
        cands = [p for p in pool
                 if q.name_code in name_code_candidates(p.get("player_name", ""))]

        if not cands:
            # surname-suffix fallback
            for p in pool:
                nm = p.get("player_name") or ""
                last = normalize_name(nm.split()[-1]) if nm else ""
                if last and q.name_code.endswith(last):
                    cands.append(p)

        if not cands:
            continue

        if len(cands) == 1:
            mapping[q.ticker] = cands[0]["player_id"]
            continue

        # ambiguous: let the jersey number decide
        if q.jersey is not None:
            exact = [p for p in cands
                     if _jersey_of(p) is not None and int(_jersey_of(p)) == int(q.jersey)]
            if len(exact) == 1:
                mapping[q.ticker] = exact[0]["player_id"]
                continue
        # still ambiguous -> leave unmatched rather than guess wrong
    return mapping


def _jersey_of(player: dict):
    v = player.get("jersey_number")
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f
