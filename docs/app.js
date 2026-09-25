const MARKET_LABEL = {
  anytime_td: "Anytime TD",
  passing_yards: "Passing Yds",
  rushing_yards: "Rushing Yds",
  receiving_yards: "Receiving Yds",
};

// Real primary team colors, used for the matchup chips. No external logo
// images are loaded here on purpose -- see README for why.
const TEAM_COLOR = {
  ARI: "#97233F", ATL: "#A71930", BAL: "#241773", BUF: "#00338D",
  CAR: "#0085CA", CHI: "#0B162A", CIN: "#FB4F14", CLE: "#311D00",
  DAL: "#041E42", DEN: "#FB4F14", DET: "#0076B6", GB: "#203731",
  HOU: "#03202F", IND: "#002C5F", JAX: "#006778", KC: "#E31837",
  LA: "#003594", LAC: "#0080C6", LV: "#000000", MIA: "#008E97",
  MIN: "#4F2683", NE: "#002244", NO: "#D3BC8D", NYG: "#0B2265",
  NYJ: "#125740", PHI: "#004C54", PIT: "#FFB612", SEA: "#69BE28",
  SF: "#AA0000", TB: "#D50A0A", TEN: "#4B92DB", WAS: "#5A1414",
};

const POSITION_COLOR = { QB: "#5b9dff", RB: "#ff9a4d", WR: "#c98bff", TE: "#34d399" };

let ROWS = [];
let HISTORY = {}; // player_id -> [{week, projection, market}, ...]
let FAVORITES = new Set(JSON.parse(localStorage.getItem("nflprops_favorites") || "[]"));
let ACTIVE_PRESET = null;
let SORT_KEY = "edge";
let SORT_DIR = -1;
let EXPANDED_ROW = null;
let KALSHI_FETCHED_AT = null;

// ---------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------
async function load() {
  let meta;
  try {
    const metaRes = await fetch("data/meta.json", { cache: "no-store" });
    if (!metaRes.ok) throw new Error(`HTTP ${metaRes.status} fetching meta.json`);
    meta = await metaRes.json();
  } catch (err) {
    document.getElementById("subtitle").textContent = "Could not load meta.json: " + err.message;
    console.error("meta load failed:", err);
    return;
  }

  try {
    const propsRes = await fetch("data/props.json", { cache: "no-store" });
    if (!propsRes.ok) throw new Error(`HTTP ${propsRes.status} fetching props.json`);
    ROWS = await propsRes.json();
  } catch (err) {
    document.getElementById("props-body").innerHTML =
      `<tr><td colspan="11" class="loading">Could not load props.json: ${err.message}</td></tr>`;
    console.error("props load failed:", err);
    return;
  }

  const genDate = new Date(meta.generated_at);
  document.getElementById("subtitle").textContent =
    `Season ${meta.season}, Week ${meta.week} \u00b7 ${meta.n_players} players \u00b7 updated ${genDate.toLocaleString()}`;
  document.getElementById("stat-updated").textContent = "updated " + genDate.toLocaleTimeString();

  KALSHI_FETCHED_AT = meta.kalshi_fetched_at || null;
  const age = marketAgeLabel();
  if (age) {
    const el = document.getElementById("stat-updated");
    el.insertAdjacentHTML("afterend",
      `<span class="dot"></span><span class="stat" title="Kalshi prices are a snapshot and move during the week">
         market prices <strong style="color:${age.level === "fresh" ? "var(--pos)" : age.level === "aging" ? "#fcd34d" : "var(--neg)"}">${age.txt}</strong></span>`);
  }

  loadHistory(meta.season).catch(() => {}); // best-effort, never blocks the table

  render();
}

async function loadHistory(season) {
  try {
    const idxRes = await fetch("data/history/index.json", { cache: "no-store" });
    if (!idxRes.ok) return;
    const weeks = await idxRes.json();
    const recent = weeks.slice(-3); // last 3 weeks is all the sparkline needs
    for (const wk of recent) {
      const r = await fetch(`data/history/${season}_wk${wk}.json`, { cache: "no-store" });
      if (!r.ok) continue;
      const rows = await r.json();
      for (const row of rows) {
        const key = row.player_id + "|" + row.market;
        if (!HISTORY[key]) HISTORY[key] = [];
        HISTORY[key].push({ week: wk, projection: row.projection, line: row.line });
      }
    }
  } catch (err) {
    console.warn("history load skipped:", err);
  }
}

// ---------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------
function activePosition() {
  const el = document.querySelector('#position-pills .pill.active');
  return el ? el.dataset.position : "all";
}
function activeMarket() {
  const el = document.querySelector('#market-pills .pill.active');
  return el ? el.dataset.market : "all";
}

function passesPreset(row, preset) {
  if (!preset) return true;
  if (preset === "high-edge") return row.recommended_side && row.recommended_side !== "pass";
  if (preset === "all-td") return row.market === "anytime_td";
  if (preset === "qb-passing") return row.market === "passing_yards";
  if (preset === "longshot-td") return row.market === "anytime_td" && (row.probability ?? 1) < 0.10;
  if (preset === "favorites") return FAVORITES.has(row.player_id + "|" + row.market + "|" + (row.line ?? ""));
  return true;
}

function fuzzyMatch(haystack, needle) {
  if (!needle) return true;
  haystack = haystack.toLowerCase();
  needle = needle.toLowerCase();
  if (haystack.includes(needle)) return true;
  // loose subsequence match so "mahms" still finds "mahomes"
  let i = 0;
  for (const ch of haystack) {
    if (ch === needle[i]) i++;
    if (i === needle.length) return true;
  }
  return false;
}

function filteredRows() {
  const search = document.getElementById("search").value.trim();
  const position = activePosition();
  const market = activeMarket();
  return ROWS.filter(r => {
    if (position !== "all" && r.position !== position) return false;
    if (market !== "all" && r.market !== market) return false;
    if (!passesPreset(r, ACTIVE_PRESET)) return false;
    if (search) {
      const hay = `${r.player_name} ${r.team} ${r.opponent}`;
      if (!fuzzyMatch(hay, search)) return false;
    }
    return true;
  });
}

// ---------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------
function fmt(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return Number(n).toFixed(digits);
}

function initials(name) {
  if (!name) return "?";
  const parts = name.trim().split(/\s+/);
  return ((parts[0]?.[0] || "") + (parts[parts.length - 1]?.[0] || "")).toUpperCase();
}

function teamChip(team) {
  const color = TEAM_COLOR[team] || "#334155";
  return `<span class="team-chip" style="background:${color}22;color:${color}">${team || "-"}</span>`;
}

function probClass(row) {
  const p = row.probability;
  if (p === null || p === undefined) return "prob-none";
  // anytime_td probabilities cluster low; yardage over/under cluster near 50%.
  // Use different bands per market so the heatmap is meaningful either way.
  if (row.market === "anytime_td") {
    if (p < 0.05) return "prob-1";
    if (p < 0.15) return "prob-2";
    if (p < 0.30) return "prob-3";
    return "prob-4";
  }
  if (p < 0.35) return "prob-1";
  if (p < 0.50) return "prob-2";
  if (p < 0.65) return "prob-3";
  return "prob-4";
}

function edgeClass(row) {
  const e = row.edge;
  if (e === null || e === undefined) return "edge-none";
  const abs = Math.abs(e);
  if (abs >= 0.10) return "edge-high" + (e < 0 ? " neg" : "");
  if (abs >= 0.05) return "edge-medium";
  if (abs >= 0.02) return "edge-small";
  return "edge-none";
}

function sideClass(side) { return `side-${side || "pass"}`; }

function favKey(row) { return row.player_id + "|" + row.market + "|" + (row.line ?? ""); }

function toggleFavorite(row) {
  const key = favKey(row);
  if (FAVORITES.has(key)) FAVORITES.delete(key); else FAVORITES.add(key);
  localStorage.setItem("nflprops_favorites", JSON.stringify([...FAVORITES]));
}

// ---------------------------------------------------------------------
// Drawer (expanded row detail)
// ---------------------------------------------------------------------
function sparklineSvg(points) {
  if (!points || points.length < 2) {
    return `<span class="sparkline-empty">Not enough weeks of history yet</span>`;
  }
  const vals = points.map(p => p.projection).filter(v => v !== null && v !== undefined);
  if (vals.length < 2) return `<span class="sparkline-empty">Not enough weeks of history yet</span>`;
  const w = 110, h = 30, pad = 3;
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = (max - min) || 1;
  const step = (w - pad * 2) / (vals.length - 1);
  const coords = vals.map((v, i) => {
    const x = pad + i * step;
    const y = h - pad - ((v - min) / range) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    <polyline points="${coords.join(" ")}" fill="none" stroke="#5b9dff" stroke-width="2"/>
    ${coords.map(c => `<circle cx="${c.split(",")[0]}" cy="${c.split(",")[1]}" r="2.2" fill="#5b9dff"/>`).join("")}
  </svg>`;
}

function attributionHtml(row) {
  const a = row.attribution;
  if (!a) return "";
  if (a.market === "anytime_td") {
    const tp = a.team_pool || {}, ps = a.player_share || {};
    return `<div class="drawer-block">
      <h4>How this TD probability is built</h4>
      <div class="row"><span>Implied team total</span><span>${tp.implied_team_total ?? "-"}</span></div>
      <div class="row"><span>Expected team TDs</span><span>${tp.expected_offensive_td ?? "-"}</span></div>
      <div class="row"><span>&nbsp;&nbsp;receiving pool</span><span>${tp.expected_receiving_td ?? "-"}</span></div>
      <div class="row"><span>&nbsp;&nbsp;rushing pool</span><span>${tp.expected_rushing_td ?? "-"}</span></div>
      <div class="row"><span>Inside-10 target share</span><span>${ps.inside10_target_share ?? "-"}</span></div>
      <div class="row"><span>Red zone target share</span><span>${ps.rz_target_share ?? "-"}</span></div>
      <div class="row"><span>Inside-5 carry share</span><span>${ps.inside5_rush_share ?? "-"}</span></div>
      <div class="row"><span><strong>Expected TDs (lambda)</strong></span><span><strong>${a.lambda_total ?? "-"}</strong></span></div>
    </div>`;
  }
  const opp = a.opportunity || {}, eff = a.efficiency || {};
  const oppRows = Object.entries(opp).map(([k, v]) =>
    `<div class="row"><span>${k.replace(/_/g, " ")}</span><span>${v}</span></div>`).join("");
  const stepRows = (a.steps || []).map(s => {
    const d = s.delta === null || s.delta === undefined ? "" :
      `<span class="${s.delta >= 0 ? "side-over" : "side-under"}">${s.delta >= 0 ? "+" : ""}${s.delta}</span>`;
    return `<div class="row" title="${(s.detail || "").replace(/"/g, "&quot;")}">
        <span>${s.label}</span><span>${s.yards} ${d}</span></div>`;
  }).join("");
  return `<div class="drawer-block">
      <h4>Opportunity</h4>${oppRows}
      <div class="row"><span>yds per opportunity</span><span>${Object.values(eff)[0] ?? "-"}</span></div>
    </div>
    <div class="drawer-block" style="grid-column: span 2;">
      <h4>How the projection is built (yards)</h4>${stepRows}
      <div class="row" style="border-top:1px solid var(--border);margin-top:4px;padding-top:6px">
        <span><strong>Projection</strong></span><span><strong>${a.projection}</strong></span></div>
    </div>`;
}

function marketAgeLabel() {
  if (!KALSHI_FETCHED_AT) return null;
  const mins = (Date.now() - new Date(KALSHI_FETCHED_AT).getTime()) / 60000;
  if (Number.isNaN(mins)) return null;
  const txt = mins < 90 ? `${Math.round(mins)} min old`
    : mins < 60 * 36 ? `${(mins / 60).toFixed(1)} hr old`
    : `${(mins / 1440).toFixed(1)} days old`;
  // Kalshi NFL prices drift through the week on injury news. Past a few
  // hours the displayed price is a historical quote, not something you can
  // trade at, so it is flagged rather than shown as if it were live.
  const level = mins < 120 ? "fresh" : mins < 60 * 12 ? "aging" : "stale";
  return { txt, level, mins };
}

function marketHtml(row) {
  if (!row.market_ticker) return "";
  const age = marketAgeLabel();
  const ageWarn = !age ? "" :
    `<div class="row" style="margin-top:6px">
       <span>Prices captured</span>
       <span style="color:${age.level === "fresh" ? "var(--pos)" : age.level === "aging" ? "#fcd34d" : "var(--neg)"}">
         ${age.txt}${age.level === "stale" ? " - RE-FETCH BEFORE BETTING" : ""}</span></div>`;
  const d = row.ladder_divergence;
  const pct = v => (v === null || v === undefined) ? "-" : (v * 100).toFixed(1) + "%";
  let rows = `
    <div class="row"><span>Market implied prob</span><span>${pct(row.market_implied_probability)}</span></div>
    <div class="row"><span>Model prob</span><span>${pct(row.model_probability)}</span></div>
    <div class="row"><span>Strike</span><span>${row.market_strike ?? "-"}</span></div>
    <div class="row"><span>Bid / Ask</span><span>${row.yes_bid ?? "-"} / ${row.yes_ask ?? "-"}</span></div>`;
  if (d && d.status === "ok") {
    rows += `
    <div class="row" style="border-top:1px solid var(--border);margin-top:4px;padding-top:6px">
      <span>Model projection</span><span>${d.model_projection}</span></div>
    <div class="row"><span>Market implied median</span><span>${d.market_implied_median}</span></div>
    <div class="row"><span>Level gap</span><span>${d.level_gap > 0 ? "+" : ""}${d.level_gap}</span></div>
    <div class="row"><span>Tail prob gap</span><span>${pct(d.mean_tail_prob_gap)}</span></div>`;
  }
  const ladder = (row.kalshi_ladder || []).filter(c => c.liquid).map(c =>
    `<div class="row"><span>${c.strike}+</span><span>mkt ${pct(c.market_prob)} / model ${pct(c.model_prob)}</span></div>`
  ).join("");
  const verdict = d && d.status === "ok"
    ? `<div style="margin-top:8px;font-size:11.5px;color:var(--text-dim);line-height:1.45">
         <strong style="color:var(--text)">${d.verdict.replace(/_/g, " ")}</strong><br>${d.interpretation}</div>`
    : "";
  return `<div class="drawer-block" style="grid-column: span 2;">
      <h4>Market (Kalshi)</h4>${rows}
      ${ladder ? `<div style="margin-top:8px"><h4>Ladder</h4>${ladder}</div>` : ""}
      ${ageWarn}
      ${verdict}
    </div>`;
}

function drawerHtml(row) {
  const histKey = row.player_id + "|" + row.market;
  const hist = HISTORY[histKey] || [];
  const ci = row.ci_80 ? `${fmt(row.ci_80[0])} - ${fmt(row.ci_80[1])}` : "-";
  return `<div class="drawer">
    <div class="drawer-block">
      <h4>Game context</h4>
      <div class="row"><span>Spread</span><span>${row.spread ?? "-"}</span></div>
      <div class="row"><span>Total</span><span>${row.total ?? "-"}</span></div>
      <div class="row"><span>Implied team total</span><span>${fmt(row.implied_team_total)}</span></div>
      <div class="row"><span>Opponent defense rank</span><span>${row.opp_def_rank ? "#" + row.opp_def_rank + " of 32" : "-"}</span></div>
    </div>
    <div class="drawer-block">
      <h4>Projection detail</h4>
      <div class="big">${fmt(row.projection, row.market === "anytime_td" ? 2 : 1)}</div>
      <div class="row"><span>80% range</span><span>${ci}</span></div>
      <div class="row"><span>Top driver</span><span>${row.top_driver || "-"}</span></div>
    </div>
    <div class="drawer-block">
      <h4>Trend (last ${Math.max(hist.length, 1)} wk${hist.length === 1 ? "" : "s"})</h4>
      <div class="sparkline-wrap">${sparklineSvg(hist)}</div>
    </div>
    ${attributionHtml(row)}
    ${marketHtml(row)}
  </div>`;
}

// ---------------------------------------------------------------------
// Autocomplete
// ---------------------------------------------------------------------
function renderAutocomplete(query) {
  const box = document.getElementById("autocomplete");
  if (!query) { box.classList.remove("open"); box.innerHTML = ""; return; }
  const seen = new Set();
  const matches = [];
  for (const r of ROWS) {
    if (seen.has(r.player_id)) continue;
    if (fuzzyMatch(`${r.player_name} ${r.team}`, query)) {
      seen.add(r.player_id);
      matches.push(r);
    }
    if (matches.length >= 8) break;
  }
  if (!matches.length) { box.classList.remove("open"); box.innerHTML = ""; return; }
  box.innerHTML = matches.map(r =>
    `<div class="ac-item" data-player="${r.player_id}">
       <span>${r.player_name}</span>
       <span class="ac-meta">${r.position} \u00b7 ${r.team}</span>
     </div>`).join("");
  box.classList.add("open");
  box.querySelectorAll(".ac-item").forEach(el => {
    el.addEventListener("click", () => {
      const row = ROWS.find(r => r.player_id === el.dataset.player);
      document.getElementById("search").value = row.player_name;
      box.classList.remove("open");
      render();
    });
  });
}

// ---------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------
function sortRows(rows) {
  return rows.slice().sort((a, b) => {
    let av = a[SORT_KEY], bv = b[SORT_KEY];
    if (SORT_KEY === "edge") { av = Math.abs(av || 0); bv = Math.abs(bv || 0); }
    if (typeof av === "string" || typeof bv === "string") {
      av = (av || "").toString(); bv = (bv || "").toString();
      return SORT_DIR * av.localeCompare(bv);
    }
    return SORT_DIR * ((av ?? -Infinity) - (bv ?? -Infinity));
  });
}

// ---------------------------------------------------------------------
// Render: desktop table
// ---------------------------------------------------------------------
function rowHtml(r, idx) {
  const flagBadges = (r.flags || [])
    .filter(f => f !== "no_current_season_sample")
    .map(f => `<span class="flag-icon" title="${f}">!</span>`).join("");
  const prob = r.probability !== undefined && r.probability !== null ? fmt(r.probability * 100, 1) + "%" : "-";
  const edgeText = r.edge !== null && r.edge !== undefined ? fmt(r.edge * 100, 1) + "pp" : "-";
  const isFav = FAVORITES.has(favKey(r));
  const rowId = `row-${idx}`;

  return `<tr class="prop-row" data-idx="${idx}" id="${rowId}">
    <td>
      <div class="player-cell">
        <div class="player-avatar" style="color:${POSITION_COLOR[r.position] || "#9aa8bc"}">${initials(r.player_name)}</div>
        <div>
          <div class="player-name">${r.player_name}${flagBadges}</div>
        </div>
      </div>
    </td>
    <td><span class="pos-badge" style="color:${POSITION_COLOR[r.position] || "#9aa8bc"}">${r.position}</span></td>
    <td>
      <div class="matchup">${teamChip(r.team)}<span class="vs">@</span>${teamChip(r.opponent)}</div>
    </td>
    <td>
      <span class="market-badge ${r.market}"><span class="dot"></span>${MARKET_LABEL[r.market] || r.market}</span>
    </td>
    <td>${r.line !== null && r.line !== undefined ? r.line : "-"}</td>
    <td>
      <div class="proj-value">${fmt(r.projection, r.market === "anytime_td" ? 2 : 1)}</div>
      ${r.ci_80 ? `<div class="proj-ci">${fmt(r.ci_80[0])}-${fmt(r.ci_80[1])}</div>` : ""}
    </td>
    <td><span class="prob-cell ${probClass(r)}">${prob}</span></td>
    <td><span class="edge-cell ${edgeClass(r)}">${edgeText}</span></td>
    <td class="${sideClass(r.recommended_side)}">${r.recommended_side || "-"}</td>
    <td>${r.kelly ? fmt(r.kelly * 100, 1) + "%" : "-"}</td>
    <td><span class="fav-star ${isFav ? "active" : ""}" data-fav-idx="${idx}">\u2605</span></td>
  </tr>`;
}

function renderTable(rows) {
  const body = document.getElementById("props-body");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="11" class="loading">No rows match those filters.</td></tr>`;
    return;
  }
  const limited = rows.slice(0, 500);
  body.innerHTML = limited.map((r, i) => rowHtml(r, i)).join("");

  body.querySelectorAll(".prop-row").forEach(tr => {
    tr.addEventListener("click", (e) => {
      if (e.target.classList.contains("fav-star")) return;
      const idx = tr.dataset.idx;
      toggleDrawer(tr, limited[idx]);
    });
  });
  body.querySelectorAll(".fav-star").forEach(star => {
    star.addEventListener("click", (e) => {
      e.stopPropagation();
      const row = limited[star.dataset.favIdx];
      toggleFavorite(row);
      star.classList.toggle("active");
    });
  });
}

function toggleDrawer(tr, row) {
  const existing = tr.nextElementSibling;
  if (existing && existing.classList.contains("drawer-row")) {
    existing.remove();
    tr.classList.remove("expanded");
    EXPANDED_ROW = null;
    return;
  }
  document.querySelectorAll("tr.drawer-row").forEach(d => d.remove());
  document.querySelectorAll("tr.expanded").forEach(d => d.classList.remove("expanded"));

  tr.classList.add("expanded");
  const drawerRow = document.createElement("tr");
  drawerRow.className = "drawer-row";
  const td = document.createElement("td");
  td.colSpan = 11;
  td.innerHTML = drawerHtml(row);
  drawerRow.appendChild(td);
  tr.after(drawerRow);
  EXPANDED_ROW = row.player_id + row.market;
}

// ---------------------------------------------------------------------
// Render: mobile cards
// ---------------------------------------------------------------------
function cardHtml(r, idx) {
  const isFav = FAVORITES.has(favKey(r));
  const prob = r.probability !== undefined && r.probability !== null ? fmt(r.probability * 100, 1) + "%" : "-";
  const edgeText = r.edge !== null && r.edge !== undefined ? fmt(r.edge * 100, 1) + "pp" : "-";
  return `<div class="prop-card" data-idx="${idx}">
    <div class="prop-card-top">
      <div>
        <div class="player-name">${r.player_name} <span class="pos-badge">${r.position}</span></div>
        <div class="prop-card-matchup">${r.team} @ ${r.opponent} \u00b7 ${MARKET_LABEL[r.market] || r.market}</div>
      </div>
      <span class="fav-star ${isFav ? "active" : ""}" data-fav-idx="${idx}">\u2605</span>
    </div>
    <div class="prop-card-body">
      <span class="prob-cell ${probClass(r)}">${prob}</span>
      <div class="prop-card-proj">
        <div class="proj-value">${fmt(r.projection, r.market === "anytime_td" ? 2 : 1)}</div>
        <div class="proj-ci">${r.line !== null && r.line !== undefined ? "line " + r.line : ""}</div>
      </div>
    </div>
    <div class="prop-card-footer">
      <span class="edge-cell ${edgeClass(r)}">${edgeText}</span>
      <span class="${sideClass(r.recommended_side)}">${r.recommended_side || "-"}</span>
    </div>
  </div>`;
}

function renderCards(rows) {
  const list = document.getElementById("card-list");
  if (!rows.length) {
    list.innerHTML = `<div class="loading">No rows match those filters.</div>`;
    return;
  }
  const limited = rows.slice(0, 200);
  list.innerHTML = limited.map((r, i) => cardHtml(r, i)).join("");
  list.querySelectorAll(".fav-star").forEach(star => {
    star.addEventListener("click", (e) => {
      e.stopPropagation();
      const row = limited[star.dataset.favIdx];
      toggleFavorite(row);
      star.classList.toggle("active");
    });
  });
}

// ---------------------------------------------------------------------
// Main render
// ---------------------------------------------------------------------
function render() {
  const rows = sortRows(filteredRows());
  renderTable(rows);
  renderCards(rows);

  document.getElementById("stat-total").textContent = rows.length;
  document.getElementById("stat-edges").textContent =
    rows.filter(r => r.recommended_side && r.recommended_side !== "pass").length;

  const activeFilters = [];
  if (activePosition() !== "all") activeFilters.push(activePosition());
  if (activeMarket() !== "all") activeFilters.push(MARKET_LABEL[activeMarket()] || activeMarket());
  if (ACTIVE_PRESET) activeFilters.push(ACTIVE_PRESET);
  document.getElementById("stat-filters").textContent =
    activeFilters.length ? activeFilters.join(", ") : "No filters applied";

  document.querySelectorAll("th[data-sort]").forEach(th => {
    th.classList.toggle("sorted", th.dataset.sort === SORT_KEY);
    th.querySelector(".arrow")?.remove();
    if (th.dataset.sort === SORT_KEY) {
      const arrow = document.createElement("span");
      arrow.className = "arrow";
      arrow.textContent = SORT_DIR === 1 ? "\u2191" : "\u2193";
      th.appendChild(arrow);
    }
  });
}

// ---------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------
document.getElementById("search").addEventListener("input", (e) => {
  renderAutocomplete(e.target.value.trim());
  render();
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-row")) document.getElementById("autocomplete").classList.remove("open");
});

document.querySelectorAll("#position-pills .pill").forEach(pill => {
  pill.addEventListener("click", () => {
    document.querySelectorAll("#position-pills .pill").forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
    render();
  });
});
document.querySelectorAll("#market-pills .pill").forEach(pill => {
  pill.addEventListener("click", () => {
    document.querySelectorAll("#market-pills .pill").forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
    render();
  });
});
document.querySelectorAll("#preset-pills .pill").forEach(pill => {
  pill.addEventListener("click", () => {
    const preset = pill.dataset.preset;
    if (ACTIVE_PRESET === preset) {
      ACTIVE_PRESET = null;
      pill.classList.remove("active");
    } else {
      document.querySelectorAll("#preset-pills .pill").forEach(p => p.classList.remove("active"));
      ACTIVE_PRESET = preset;
      pill.classList.add("active");
    }
    render();
  });
});
document.querySelectorAll(".bottom-nav .pill").forEach(pill => {
  pill.addEventListener("click", () => {
    document.querySelectorAll(".bottom-nav .pill").forEach(p => p.classList.remove("active"));
    pill.classList.add("active");
    const f = pill.dataset.mobileFilter;
    if (f === "all") { ACTIVE_PRESET = null; }
    else if (f === "high-edge" || f === "favorites") { ACTIVE_PRESET = f; }
    else {
      document.querySelectorAll("#market-pills .pill").forEach(p => p.classList.remove("active"));
      document.querySelector(`#market-pills .pill[data-market="${f}"]`)?.classList.add("active");
    }
    render();
  });
});
document.querySelectorAll("th[data-sort]").forEach(th => {
  th.addEventListener("click", () => {
    const key = th.dataset.sort;
    if (SORT_KEY === key) { SORT_DIR *= -1; } else { SORT_KEY = key; SORT_DIR = -1; }
    render();
  });
});

load();
