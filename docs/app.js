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
      `<tr><td colspan="13" class="loading">Could not load props.json: ${err.message}</td></tr>`;
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

  populateTeamFilter();
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

function activeTeam() {
  const el = document.getElementById("team-filter");
  return el ? el.value : "all";
}

function populateTeamFilter() {
  const sel = document.getElementById("team-filter");
  if (!sel) return;
  const teams = [...new Set(ROWS.flatMap(r => [r.team, r.opponent]).filter(Boolean))].sort();
  sel.innerHTML = `<option value="all">All teams (${teams.length})</option>` +
    teams.map(t => `<option value="${t}">${t}</option>`).join("");
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
  const team = activeTeam();
  return ROWS.filter(r => {
    if (position !== "all" && r.position !== position) return false;
    if (market !== "all" && r.market !== market) return false;
    // Exact match on either side of the matchup. Typing "SEA" in the search
    // box also matched player names containing those letters; a dedicated
    // exact team filter is the fix, and it covers the opponent too so you
    // can pull up a whole game.
    if (team !== "all" && r.team !== team && r.opponent !== team) return false;
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

function tdDistHtml(row) {
  const d = row.td_distribution;
  if (!d) return "";
  const segs = [["0 TD", d.p_0_td, "#64748b"], ["1 TD", d.p_1_td, "#34d399"],
                ["2 TD", d.p_2_td, "#5b9dff"], ["3+ TD", d.p_3plus_td, "#ffd23f"]];
  return `<div class="drawer-block">
    <h4>TD outcome distribution</h4>
    ${segs.map(([l, v, c]) =>
      `<div class="row"><span>${l}</span><span style="color:${c}">${((v||0)*100).toFixed(1)}%</span></div>`).join("")}
    <div class="td-dist">${segs.map(([, v, c]) =>
      `<div class="seg" style="flex:${Math.max(v||0, 0.001)};background:${c}"></div>`).join("")}</div>
  </div>`;
}

function confidenceHtml(row) {
  if (row.confidence === undefined || row.confidence === null) return "";
  const p = row.confidence_parts || {};
  return `<div class="drawer-block">
    <h4>Confidence ${row.confidence_grade ?? ""}</h4>
    <div class="big">${row.confidence}</div>
    <div class="row"><span>role certainty</span><span>${p.role ?? "-"}</span></div>
    <div class="row"><span>sample size</span><span>${p.sample ?? "-"}</span></div>
    <div class="row"><span>market quality</span><span>${p.market_quality ?? "-"}</span></div>
    <div class="row"><span>model/market agreement</span><span>${p.agreement ?? "-"}</span></div>
  </div>`;
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
    ${tdDistHtml(row)}
    ${confidenceHtml(row)}
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
    <td><span class="edge-cell ${edgeClass(r)}">${edgeText}</span>
        ${r.model_probability !== undefined && r.model_probability !== null
          ? `<span class="edge-sub">model ${(r.model_probability*100).toFixed(0)}% / mkt ${((r.market_implied_probability??0)*100).toFixed(0)}%</span>` : ""}</td>
    <td><span class="conf-pill">${r.confidence_grade ?? "-"}</span></td>
    <td><span class="tag tag-${(r.bettable||"NO EDGE").replace(/\s/g,"")}">${r.bettable ?? "-"}</span></td>
    <td class="${sideClass(r.recommended_side)}">${r.recommended_side || "-"}</td>
    <td>${r.kelly ? fmt(r.kelly * 100, 1) + "%" : "-"}</td>
    <td><span class="fav-star ${isFav ? "active" : ""}" data-fav-idx="${idx}">\u2605</span></td>
  </tr>`;
}

function renderTable(rows) {
  const body = document.getElementById("props-body");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="13" class="loading">No rows match those filters.</td></tr>`;
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
  td.colSpan = 13;
  td.innerHTML = drawerHtml(row);
  drawerRow.appendChild(td);
  tr.after(drawerRow);
  EXPANDED_ROW = row.player_id + row.market;
}

// ---------------------------------------------------------------------
// Render: mobile cards
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// Mobile cards
// ---------------------------------------------------------------------
const CARD_PAGE = 50;
let CARD_LIMIT = CARD_PAGE;
let CARD_ROWS = [];
let LAST_SIG = "";

function pct(v, d = 0) {
  return v === null || v === undefined || Number.isNaN(v) ? "-" : (v * 100).toFixed(d) + "%";
}

function cardHtml(r, idx) {
  const isFav = FAVORITES.has(favKey(r));
  const isTD = r.market === "anytime_td";
  const hasEdge = r.edge !== null && r.edge !== undefined;
  const modelP = r.model_probability ?? (isTD ? r.probability : null);

  // Edge is the reason the page exists, so it is the biggest thing on the
  // card whenever a market price is available. Without one, fall back to the
  // model's headline number rather than showing an empty dash.
  let hero, heroLabel, heroClass;
  if (hasEdge) {
    const e = r.edge * 100;
    hero = (e > 0 ? "+" : "") + e.toFixed(1);
    heroLabel = "edge pp";
    heroClass = edgeClass(r);
  } else if (isTD) {
    hero = pct(r.probability, 0);
    heroLabel = "TD prob";
    heroClass = "";
  } else {
    hero = fmt(r.projection, 1);
    heroLabel = "proj yds";
    heroClass = "";
  }

  const stats = [
    ["Model", pct(modelP)],
    ["Market", pct(r.market_implied_probability)],
    [isTD ? "Exp TD" : "Proj", fmt(r.projection, isTD ? 2 : 1)],
    ["Line", r.line !== null && r.line !== undefined ? r.line + "+" : "-"],
  ].map(([l, v]) => `<div class="pc-stat"><label>${l}</label><b>${v}</b></div>`).join("");

  const side = (r.recommended_side || "").toLowerCase();
  const sideTxt = side === "yes" ? "YES" : side === "no" ? "NO" : "";
  const tag = r.bettable
    ? `<span class="tag tag-${r.bettable.replace(/\s/g, "")}">${r.bettable}</span>` : "";
  const conf = r.confidence_grade ? `<span class="conf-pill">${r.confidence_grade}</span>` : "";
  const kelly = r.kelly ? `<span class="pc-kelly">${(r.kelly * 100).toFixed(1)}% kelly</span>` : "";

  return `<div class="prop-card" data-idx="${idx}">
    <div class="pc-head">
      <div class="pc-id">
        <div class="player-name">${r.player_name} <span class="pos-badge">${r.position}</span></div>
        <div class="pc-sub">${r.team} @ ${r.opponent || "BYE"} &middot; ${MARKET_LABEL[r.market] || r.market}</div>
      </div>
      <div class="pc-hero-wrap">
        <span class="pc-hero ${heroClass}">${hero}</span>
        <div class="pc-hero-label">${heroLabel}</div>
      </div>
      <span class="fav-star ${isFav ? "active" : ""}" data-fav-idx="${idx}">&#9733;</span>
    </div>
    <div class="pc-stats">${stats}</div>
    <div class="pc-foot">
      ${tag}${conf}${sideTxt ? `<span class="pc-side ${sideClass(side)}">${sideTxt}</span>` : ""}${kelly}
      <span class="pc-chev">Details &#9662;</span>
    </div>
    <div class="pc-detail"></div>
  </div>`;
}

function renderCards(rows) {
  CARD_ROWS = rows;
  const list = document.getElementById("card-list");
  if (!rows.length) {
    list.innerHTML = `<div class="loading">No rows match those filters.</div>`;
    return;
  }
  const shown = rows.slice(0, CARD_LIMIT);
  const remaining = rows.length - shown.length;
  list.innerHTML = shown.map((r, i) => cardHtml(r, i)).join("") +
    (remaining > 0
      ? `<button class="card-more" id="card-more">Show ${Math.min(CARD_PAGE, remaining)} more <span>(${remaining} remaining)</span></button>`
      : "");
}

// Keeps the mobile-only controls in step with the real filter state.
function syncMobileUI(rows) {
  let n = 0;
  if (activePosition() !== "all") n++;
  if (activeTeam() !== "all") n++;
  if (ACTIVE_PRESET) n++;
  const badge = document.getElementById("filter-count");
  if (badge) { badge.textContent = n || ""; badge.hidden = !n; }

  const apply = document.getElementById("sheet-apply");
  if (apply) apply.textContent = `Show ${rows.length} result${rows.length === 1 ? "" : "s"}`;

  const mSort = document.getElementById("m-sort");
  if (mSort && [...mSort.options].some(o => o.value === SORT_KEY)) mSort.value = SORT_KEY;
  const dir = document.getElementById("m-sort-dir");
  if (dir) dir.textContent = SORT_DIR === 1 ? "\u2191" : "\u2193";

  const mk = activeMarket();
  let cur = null;
  if (ACTIVE_PRESET === "high-edge") cur = "high-edge";
  else if (ACTIVE_PRESET === "favorites") cur = "favorites";
  else if (!ACTIVE_PRESET && mk === "anytime_td") cur = "anytime_td";
  else if (!ACTIVE_PRESET && mk === "all") cur = "all";
  document.querySelectorAll(".bottom-nav .pill").forEach(p =>
    p.classList.toggle("active", p.dataset.mobileFilter === cur));
}

// ---------------------------------------------------------------------
// Main render
// ---------------------------------------------------------------------
function renderTopPlays(rows) {
  const el = document.getElementById("top-plays");
  if (!el) return;
  const plays = rows
    .filter(r => ["STRONG", "PLAYABLE"].includes(r.bettable))
    .sort((a, b) => Math.abs(b.edge || 0) - Math.abs(a.edge || 0))
    .slice(0, 5);
  if (!plays.length) {
    // Saying "none" is the honest output when nothing clears the bar. A
    // top-plays panel that always lists five names would just be the five
    // largest random errors on a day with no edge.
    el.innerHTML = `<h3>Top plays</h3><div class="empty">
      Nothing clears the playable threshold on this slate.
      ${ROWS.some(r => r.market_ticker) ? "" : "Market prices not loaded - run fetch_kalshi."}
    </div>`;
    return;
  }
  el.innerHTML = `<h3>Top plays</h3>` + plays.map((r, i) => `
    <div class="top-play-row" data-player="${r.player_name}">
      <span class="top-play-rank">${i + 1}</span>
      <span class="top-play-name">${r.player_name}
        <span class="player-sub">${r.position} ${r.team} &middot; ${MARKET_LABEL[r.market] || r.market}${r.line != null ? " " + r.line + "+" : ""}</span>
      </span>
      <span class="tag tag-${(r.bettable||"").replace(/\s/g,"")}">${r.bettable}</span>
      <span class="top-play-edge ${(r.edge||0) > 0 ? "side-over" : "side-under"}">
        ${(r.edge||0) > 0 ? "+" : ""}${((r.edge||0)*100).toFixed(1)}pp</span>
    </div>`).join("");
  // On a phone only the first three show until tapped; the panel otherwise
  // fills the whole first screen and pushes search and the cards off it.
  if (plays.length > 3) {
    const more = document.createElement("div");
    more.className = "tp-more";
    const label = () => el.classList.contains("expanded")
      ? "Show fewer \u25B4" : `Show all ${plays.length} \u25BE`;
    more.textContent = label();
    more.addEventListener("click", () => {
      el.classList.toggle("expanded");
      more.textContent = label();
    });
    el.appendChild(more);
  }
  el.querySelectorAll(".top-play-row").forEach(row => {
    row.addEventListener("click", () => {
      document.getElementById("search").value = row.dataset.player;
      render();
    });
  });
}

function render() {
  const rows = sortRows(filteredRows());

  // New filters or a new sort order start the card list from the top again.
  const sig = JSON.stringify([document.getElementById("search").value,
    activePosition(), activeMarket(), activeTeam(), ACTIVE_PRESET, SORT_KEY, SORT_DIR]);
  if (sig !== LAST_SIG) {
    CARD_LIMIT = CARD_PAGE;
    if (LAST_SIG !== "" && window.matchMedia("(max-width: 760px)").matches) {
      window.scrollTo(0, 0);
    }
    LAST_SIG = sig;
  }

  renderTopPlays(ROWS);
  renderTable(rows);
  renderCards(rows);
  syncMobileUI(rows);

  document.getElementById("stat-total").textContent = rows.length;
  document.getElementById("stat-edges").textContent =
    rows.filter(r => r.recommended_side && r.recommended_side !== "pass").length;

  const activeFilters = [];
  if (activePosition() !== "all") activeFilters.push(activePosition());
  if (activeMarket() !== "all") activeFilters.push(MARKET_LABEL[activeMarket()] || activeMarket());
  if (ACTIVE_PRESET) activeFilters.push(ACTIVE_PRESET);
  if (activeTeam() !== "all") activeFilters.push(activeTeam());
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
    if (f === "all") {
      ACTIVE_PRESET = null;
      document.querySelectorAll("#preset-pills .pill").forEach(p => p.classList.remove("active"));
      document.querySelectorAll("#market-pills .pill").forEach(p => p.classList.remove("active"));
      document.querySelector('#market-pills .pill[data-market="all"]')?.classList.add("active");
    }
    else if (f === "high-edge" || f === "favorites") { ACTIVE_PRESET = f; }
    else {
      document.querySelectorAll("#market-pills .pill").forEach(p => p.classList.remove("active"));
      document.querySelector(`#market-pills .pill[data-market="${f}"]`)?.classList.add("active");
    }
    render();
  });
});
// ---------------------------------------------------------------------
// Toolbar, filter sheet, sort and card taps. Registered ONCE, here.
//
// An earlier version of this wiring was pasted inside render(), so every
// re-render registered another copy of each handler. The Filters button then
// fired two or three times per tap, toggling the sheet open and shut in the
// same instant, which is why it never appeared on a phone.
// ---------------------------------------------------------------------
const teamSel = document.getElementById("team-filter");
if (teamSel) teamSel.addEventListener("change", render);

document.getElementById("clear-filters")?.addEventListener("click", () => {
  document.getElementById("search").value = "";
  if (teamSel) teamSel.value = "all";
  document.querySelectorAll("#position-pills .pill").forEach(p => p.classList.remove("active"));
  document.querySelector('#position-pills .pill[data-position="all"]')?.classList.add("active");
  document.querySelectorAll("#market-pills .pill").forEach(p => p.classList.remove("active"));
  document.querySelector('#market-pills .pill[data-market="all"]')?.classList.add("active");
  document.querySelectorAll("#preset-pills .pill").forEach(p => p.classList.remove("active"));
  ACTIVE_PRESET = null;
  render();
});

// sort control (phones have no sortable table header)
const mSort = document.getElementById("m-sort");
mSort?.addEventListener("change", () => {
  SORT_KEY = mSort.value;
  SORT_DIR = SORT_KEY === "player_name" ? 1 : -1;
  render();
});
document.getElementById("m-sort-dir")?.addEventListener("click", () => {
  SORT_DIR *= -1;
  render();
});

// filter sheet
(function setupFilterSheet() {
  const btn = document.getElementById("filter-fab");
  const sheet = document.getElementById("controls");
  if (!btn || !sheet) return;
  const scrim = document.createElement("div");
  scrim.className = "filter-scrim";
  document.body.appendChild(scrim);
  const setOpen = (open) => {
    sheet.classList.toggle("open", open);
    scrim.classList.toggle("open", open);
    document.body.classList.toggle("sheet-open", open);
    btn.setAttribute("aria-expanded", String(open));
  };
  btn.addEventListener("click", () => setOpen(!sheet.classList.contains("open")));
  scrim.addEventListener("click", () => setOpen(false));
  document.getElementById("sheet-close")?.addEventListener("click", () => setOpen(false));
  document.getElementById("sheet-apply")?.addEventListener("click", () => setOpen(false));
  document.addEventListener("keydown", e => { if (e.key === "Escape") setOpen(false); });
  // The sheet stays open while choosing, so several filters can be combined;
  // "Show N results" closes it.
})();

// card taps: expand details, favourite, show more (delegated, once)
document.getElementById("card-list")?.addEventListener("click", (e) => {
  if (e.target.closest("#card-more")) {
    CARD_LIMIT += CARD_PAGE;
    renderCards(CARD_ROWS);
    return;
  }
  const star = e.target.closest(".fav-star");
  if (star) {
    toggleFavorite(CARD_ROWS[+star.dataset.favIdx]);
    star.classList.toggle("active");
    return;
  }
  const card = e.target.closest(".prop-card");
  if (!card) return;
  const box = card.querySelector(".pc-detail");
  if (card.classList.contains("open")) {
    card.classList.remove("open");
    box.innerHTML = "";
    return;
  }
  box.innerHTML = drawerHtml(CARD_ROWS[+card.dataset.idx]);
  card.classList.add("open");
});

document.querySelectorAll("th[data-sort]").forEach(th => {
  th.addEventListener("click", () => {
    const key = th.dataset.sort;
    if (SORT_KEY === key) { SORT_DIR *= -1; } else { SORT_KEY = key; SORT_DIR = -1; }
    render();
  });
});

load();
