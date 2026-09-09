const MARKET_LABEL = {
  anytime_td: "Anytime TD",
  passing_yards: "Passing Yds",
  rushing_yards: "Rushing Yds",
  receiving_yards: "Receiving Yds",
};

let ROWS = [];

async function load() {
  let meta;
  try {
    const metaRes = await fetch("data/meta.json", { cache: "no-store" });
    if (!metaRes.ok) throw new Error(`HTTP ${metaRes.status} fetching meta.json`);
    meta = await metaRes.json();
  } catch (err) {
    document.getElementById("meta").textContent = "Could not load meta.json: " + err.message;
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
  document.getElementById("meta").textContent =
    `Season ${meta.season}, Week ${meta.week} · ${meta.n_players} players ` +
    `· updated ${genDate.toLocaleString()}`;

  render();
}

function passesFilters(row, search, market, position, side) {
  if (market !== "all" && row.market !== market) return false;
  if (position !== "all" && row.position !== position) return false;
  if (side === "edges" && (!row.recommended_side || row.recommended_side === "pass")) return false;
  if (side === "over" && !["over", "yes"].includes(row.recommended_side)) return false;
  if (side === "under" && !["under", "no"].includes(row.recommended_side)) return false;
  if (search) {
    const hay = `${row.player_name} ${row.team} ${row.opponent}`.toLowerCase();
    if (!hay.includes(search.toLowerCase())) return false;
  }
  return true;
}

function fmt(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return Number(n).toFixed(digits);
}

function sideClass(side) {
  return `side-${side || "pass"}`;
}

function render() {
  const search = document.getElementById("search").value.trim();
  const market = document.getElementById("market-filter").value;
  const position = document.getElementById("position-filter").value;
  const side = document.getElementById("side-filter").value;
  const sortBy = document.getElementById("sort-by").value;

  let rows = ROWS.filter(r => passesFilters(r, search, market, position, side));

  rows.sort((a, b) => {
    if (sortBy === "edge") return Math.abs(b.edge || 0) - Math.abs(a.edge || 0);
    if (sortBy === "projection") return (b.projection || 0) - (a.projection || 0);
    return (a.player_name || "").localeCompare(b.player_name || "");
  });

  const body = document.getElementById("props-body");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="11" class="loading">No rows match those filters.</td></tr>`;
    return;
  }

  body.innerHTML = rows.slice(0, 500).map(r => {
    const flagBadges = (r.flags || [])
      .filter(f => f !== "no_current_season_sample")
      .map(f => `<span class="flag" title="${f}">!</span>`).join("");
    const prob = r.probability !== undefined ? fmt(r.probability * 100, 1) + "%" : "-";
    return `<tr>
      <td>${r.player_name}${flagBadges}</td>
      <td>${r.position}</td>
      <td>${r.team}</td>
      <td>${r.opponent || "-"}</td>
      <td>${MARKET_LABEL[r.market] || r.market}</td>
      <td>${r.line !== null && r.line !== undefined ? r.line : "-"}</td>
      <td>${fmt(r.projection, r.market === "anytime_td" ? 2 : 1)}</td>
      <td>${prob}</td>
      <td>${r.edge !== null && r.edge !== undefined ? fmt(r.edge * 100, 1) + "pp" : "-"}</td>
      <td class="${sideClass(r.recommended_side)}">${r.recommended_side || "-"}</td>
      <td>${r.kelly ? fmt(r.kelly * 100, 1) + "%" : "-"}</td>
    </tr>`;
  }).join("");
}

["search", "market-filter", "position-filter", "side-filter", "sort-by"].forEach(id => {
  document.getElementById(id).addEventListener("input", render);
  document.getElementById(id).addEventListener("change", render);
});

load();
