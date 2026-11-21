"use strict";
// SWARM public site: replays prerecorded episodes (data/episode-*.json) and archived results
// (data/results.json). Nothing here simulates; frames are the simulator's recorded snapshots.

const $ = (id) => document.getElementById(id);
const REPO = "https://github.com/Exit0Hero/SWARM/blob/main/";
const COLORS = ["#2f5d8a", "#c2701e", "#2e7d4f", "#7a4fa0", "#a8326e", "#5a6f2b", "#2d8a8a", "#8a5a2d"];
const TERRAIN = { 1: "#cdd3d8", 2: "#3d4a56", 3: "#bcd3ea" };
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const fmt = (v, d = 1) => (v === null || v === undefined ? "—" : Number.isInteger(v) ? String(v) : v.toFixed(d));
const label = { frontier: "Independent frontier", frontier_shared: "Shared belief maps", mappo: "Learned (MAPPO)",
  auction_comm: "Shared beliefs + auction", assignment: "Centralized assignment (reference)" };

let manifest = null, run = null, frame = 0, timer = null, view = "2d";
const cache = {};

async function getJSON(path) {
  const res = await fetch(path, { cache: "force-cache" });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

// ------------------------------------------------------------------ replay
async function boot() {
  try {
    manifest = await getJSON("data/manifest.json");
  } catch (e) {
    $("ep-title").innerHTML = `<span class="err">The recorded runs could not be loaded (${e.message}). Reload the page; the code and data are also on GitHub.</span>`;
    return;
  }
  $("scenarios").innerHTML = manifest.episodes.map((ep) =>
    `<button class="scenario" data-id="${ep.id}" aria-pressed="false"><strong>${ep.title}</strong>
      <span>${ep.steps} steps, ${ep.end_reason.replace("_", " ")}</span></button>`).join("");
  document.querySelectorAll(".scenario").forEach((b) => (b.onclick = () => select(b.dataset.id, true)));
  $("foot").innerHTML = `SWARM ${manifest.swarm_version}, recorded at commit <a href="https://github.com/Exit0Hero/SWARM/commit/${manifest.commit}">${manifest.commit}</a>. MIT licence. <a href="https://github.com/Exit0Hero/SWARM">GitHub</a>`;
  await select(manifest.episodes[0].id, !reduceMotion);
}

async function select(id, autoplay) {
  const ep = manifest.episodes.find((e) => e.id === id);
  document.querySelectorAll(".scenario").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
  stop();
  $("ep-title").textContent = `${ep.title}: loading…`;
  try {
    run = cache[id] || (cache[id] = await getJSON(`data/${ep.file}`));
  } catch (e) {
    $("ep-title").innerHTML = `<span class="err">This run could not be loaded (${e.message}).</span>`;
    return;
  }
  frame = 0;
  $("scrub").max = run.frames.length - 1;
  $("ep-title").textContent = ep.title;
  $("prov").innerHTML = `${ep.summary}<br><br>Policy <b>${ep.policy}</b> on the packaged <b>${ep.config}</b> scenario, seed ${ep.seed}.
    Config hash <code>${ep.config_hash.slice(0, 12)}</code>, event-stream hash <code>${ep.event_stream_sha256.slice(0, 12)}</code>.
    Reproduce: <code>swarm run --config ${ep.config} --policy ${ep.policy} --seed ${ep.seed}</code>`;
  const a = run.scene3d.analysis;
  $("note3d").textContent = a.unavailable ? "3D view of the recorded episode." :
    `Kinematic 3D replay of the same episode (${a.engine}): ${a.contacts} robot–obstacle contacts; ${a.reports_occluded} of ${a.reports_checked} detection reports would be blocked for a line-of-sight sensor. Drone altitude is a convention, not a flight model.`;
  draw();
  if (autoplay) play();
}

function play() {
  if (!run) return;
  if (frame >= run.frames.length - 1) frame = 0;
  $("b-play").textContent = "Pause"; $("b-play").setAttribute("aria-label", "Pause");
  timer = setInterval(() => { if (frame >= run.frames.length - 1) return stop(); frame++; draw(); }, +$("speed").value);
}
function stop() {
  clearInterval(timer); timer = null;
  $("b-play").textContent = "Play"; $("b-play").setAttribute("aria-label", "Play");
}
$("b-play").onclick = () => (timer ? stop() : play());
$("b-back").onclick = () => { stop(); if (run && frame > 0) { frame--; draw(); } };
$("b-fwd").onclick = () => { stop(); if (run && frame < run.frames.length - 1) { frame++; draw(); } };
$("scrub").oninput = () => { stop(); frame = +$("scrub").value; draw(); };
$("speed").onchange = () => { if (timer) { stop(); play(); } };
["t-truth", "t-radius", "t-msgs", "t-trails"].forEach((id) => ($(id).onchange = () => run && draw()));
$("yaw").oninput = () => run && draw();
function setView(v) {
  view = v;
  $("v2d").classList.toggle("on", v === "2d"); $("v3d").classList.toggle("on", v === "3d");
  $("v2d").setAttribute("aria-pressed", String(v === "2d")); $("v3d").setAttribute("aria-pressed", String(v === "3d"));
  $("ctl3d").hidden = v !== "3d"; $("note3d").hidden = v !== "3d"; $("legend").hidden = v === "3d";
  if (run) draw();
}
$("v2d").onclick = () => setView("2d");
$("v3d").onclick = () => setView("3d");

function colorsFor(f) { const c = {}; f.agents.forEach((a, k) => (c[a.id] = COLORS[k % COLORS.length])); return c; }

function draw() {
  if (view === "3d") draw3d(); else draw2d();
  panels();
}

function draw2d() {
  const f = run.frames[frame], sc = run.scenario, cv = $("map"), g = cv.getContext("2d");
  const [h, w] = sc.shape, margin = 22, cell = Math.floor((720 - margin) / Math.max(w, h));
  cv.width = w * cell + margin; cv.height = h * cell + margin;
  g.fillStyle = "#fff"; g.fillRect(0, 0, cv.width, cv.height);
  // map-sheet frame: grid references along the edges
  g.fillStyle = "#4a5866"; g.font = "11px Archivo, system-ui"; g.textAlign = "center";
  for (let c = 0; c < w; c++) g.fillText(String.fromCharCode(65 + (c % 26)), margin + c * cell + cell / 2, 14);
  g.textAlign = "right";
  for (let r = 0; r < h; r++) g.fillText(String(r + 1), margin - 5, margin + r * cell + cell / 2 + 4);
  g.textAlign = "start";
  g.save(); g.translate(margin, margin);
  g.strokeStyle = "#e3e8ec"; g.lineWidth = 1;
  for (let i = 0; i <= w; i++) { g.beginPath(); g.moveTo(i * cell + 0.5, 0); g.lineTo(i * cell + 0.5, h * cell); g.stroke(); }
  for (let i = 0; i <= h; i++) { g.beginPath(); g.moveTo(0, i * cell + 0.5); g.lineTo(w * cell, i * cell + 0.5); g.stroke(); }
  const cx = (c) => c * cell + cell / 2, cy = (r) => r * cell + cell / 2;
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    if (sc.hazard[r][c]) { g.fillStyle = "#f1c9bf"; g.fillRect(c * cell, r * cell, cell, cell); }
    const col = TERRAIN[sc.terrain[r][c]];
    if (col) { g.fillStyle = col; g.fillRect(c * cell + 1, r * cell + 1, cell - 2, cell - 2); }
  }
  const color = colorsFor(f);
  if ($("t-trails").checked) f.agents.forEach((a) => {
    g.strokeStyle = color[a.id] + "80"; g.lineWidth = 2; g.beginPath();
    run.frames.slice(0, frame + 1).forEach((fr, k) => { const p = fr.agents.find((x) => x.id === a.id).pos;
      k ? g.lineTo(cx(p[1]), cy(p[0])) : g.moveTo(cx(p[1]), cy(p[0])); });
    g.stroke();
  });
  if ($("t-truth").checked) { g.strokeStyle = "#1c2630"; g.lineWidth = 1.2;
    sc.survivors_ground_truth.forEach(([r, c]) => star(g, cx(c), cy(r), cell * 0.38)); }
  if ($("t-radius").checked) f.agents.filter((a) => a.active).forEach((a) => {
    g.strokeStyle = color[a.id] + "99"; g.setLineDash([3, 3]); g.beginPath();
    g.arc(cx(a.pos[1]), cy(a.pos[0]), a.radius * cell, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]); });
  if ($("t-msgs").checked) f.messages.forEach((m) => {
    const s = f.agents.find((a) => a.id === m.sender).pos;
    m.recipients.forEach((rid) => { const t = f.agents.find((a) => a.id === rid).pos;
      g.strokeStyle = "#d9a21bcc"; g.lineWidth = 1.6; g.beginPath(); g.moveTo(cx(s[1]), cy(s[0])); g.lineTo(cx(t[1]), cy(t[0])); g.stroke(); }); });
  f.detections.forEach((d) => { g.fillStyle = d.true ? "#2e7d4f" : "#b4412c";
    g.beginPath(); g.arc(cx(d.cell[1]), cy(d.cell[0]), cell * 0.22, 0, 2 * Math.PI); g.fill(); });
  f.agents.forEach((a) => {
    const x = cx(a.pos[1]), y = cy(a.pos[0]), s = cell * 0.4;
    g.fillStyle = color[a.id]; g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.beginPath();
    if (a.layer === "air") { g.moveTo(x, y - s); g.lineTo(x + s, y + s * 0.85); g.lineTo(x - s, y + s * 0.85); g.closePath(); }
    else g.rect(x - s * 0.8, y - s * 0.8, s * 1.6, s * 1.6);
    g.fill(); g.stroke();
    if (!a.active) { g.strokeStyle = "#b4412c"; g.lineWidth = 3; g.beginPath(); g.moveTo(x - s, y - s); g.lineTo(x + s, y + s); g.moveTo(x + s, y - s); g.lineTo(x - s, y + s); g.stroke(); }
  });
  g.restore();
}

function draw3d() {
  const f = run.frames[frame], s = run.scene3d.scene, cv = $("map"), g = cv.getContext("2d");
  const [h, w] = s.shape, yaw = (+$("yaw").value * Math.PI) / 180, pitch = 0.72;
  cv.width = 720; cv.height = 600;
  g.fillStyle = "#fff"; g.fillRect(0, 0, cv.width, cv.height);
  const scale = (1.3 * Math.min(cv.width, cv.height)) / Math.hypot(w, h), cx0 = w / 2, cy0 = h / 2;
  const P = (x, y, z) => { const dx = x - cx0, dy = y - cy0, rx = dx * Math.cos(yaw) - dy * Math.sin(yaw), ry = dx * Math.sin(yaw) + dy * Math.cos(yaw);
    return [cv.width / 2 + rx * scale, cv.height * 0.56 + (ry * Math.sin(pitch) - z * Math.cos(pitch)) * scale, ry * Math.cos(pitch) + z * Math.sin(pitch)]; };
  const poly = (pts, fill, stroke) => { g.beginPath(); pts.forEach(([x, y], k) => (k ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath();
    g.fillStyle = fill; g.fill(); if (stroke) { g.strokeStyle = stroke; g.lineWidth = 0.6; g.stroke(); } };
  const quad = (x, y, z) => [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(([a, b]) => P(x + a, y + b, z));
  poly([P(-0.5, -0.5, 0), P(w - 0.5, -0.5, 0), P(w - 0.5, h - 0.5, 0), P(-0.5, h - 0.5, 0)], "#eef1f3", "#c9d2d9");
  s.hazards.forEach(([x, y]) => poly(quad(x, y, 0), "#f1c9bfdd"));
  poly(quad(s.base[0], s.base[1], 0), "#bcd3ea");
  const items = [];
  s.boxes.forEach((b) => items.push({ d: P(b.x, b.y, b.h / 2)[2], f: () => {
    const c = b.kind === "structure" ? ["#56636f", "#3d4a56", "#6f7c88"] : ["#d6dbdf", "#c3cad0", "#e1e5e8"];
    const v = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(([dx, dy]) => [b.x + dx, b.y + dy]);
    for (let k = 0; k < 4; k++) { const a = v[k], q = v[(k + 1) % 4];
      poly([P(a[0], a[1], 0), P(q[0], q[1], 0), P(q[0], q[1], b.h), P(a[0], a[1], b.h)], c[k % 2], "#0000001f"); }
    poly(v.map(([x, y]) => P(x, y, b.h)), c[2], "#00000026"); } }));
  if ($("t-truth").checked) s.survivors_ground_truth.forEach(([x, y]) => items.push({ d: P(x, y, 0.1)[2], f: () => {
    const [px, py] = P(x, y, 0.1); g.strokeStyle = "#1c2630"; g.lineWidth = 1.2; star(g, px, py, 6); } }));
  const color = colorsFor(f);
  Object.entries(s.tracks).forEach(([aid, tr]) => {
    const pts = tr.points.slice(0, frame + 1), [x, y, z] = pts[pts.length - 1];
    if ($("t-trails").checked) items.push({ d: -1e9, f: () => { g.strokeStyle = color[aid] + "88"; g.lineWidth = 2; g.beginPath();
      pts.forEach((p, k) => { const [px, py] = P(...p); k ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke(); } });
    items.push({ d: P(x, y, z)[2] + 0.01, f: () => {
      const [px, py] = P(x, y, z), [gx, gy] = P(x, y, 0), dead = tr.disabled_at !== null && frame >= tr.disabled_at;
      g.strokeStyle = "#1c263055"; g.setLineDash([2, 2]); g.beginPath(); g.moveTo(px, py); g.lineTo(gx, gy); g.stroke(); g.setLineDash([]);
      g.fillStyle = dead ? "#8a949c" : color[aid]; g.beginPath();
      if (tr.layer === "air") { g.moveTo(px, py - 9); g.lineTo(px + 9, py + 7); g.lineTo(px - 9, py + 7); } else g.rect(px - 7, py - 7, 14, 14);
      g.fill(); g.fillStyle = "#1c2630"; g.font = "12px Archivo, system-ui"; g.fillText(aid + (dead ? " (disabled)" : ""), px + 10, py - 7); } });
  });
  if ($("t-msgs").checked) f.messages.forEach((m) => { const a = s.tracks[m.sender].points[frame];
    m.recipients.forEach((rid) => { const b = s.tracks[rid].points[frame]; items.push({ d: 1e9, f: () => {
      const [x1, y1] = P(...a), [x2, y2] = P(...b); g.strokeStyle = "#d9a21bcc"; g.lineWidth = 1.6; g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke(); } }); }); });
  items.sort((p, q) => p.d - q.d).forEach((i) => i.f()); // far first
}

function star(g, x, y, r) {
  g.beginPath();
  for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? r * 0.45 : r; g.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); }
  g.closePath(); g.stroke();
}

function panels() {
  const f = run.frames[frame], color = colorsFor(f), total = run.scenario.survivors_ground_truth.length;
  $("scrub").value = frame;
  $("step-lbl").textContent = `Step ${f.step} of ${run.steps}`;
  $("mission").innerHTML = `Survivors found: <b>${f.found}</b> of ${total} (target ${Math.ceil(run.config.recall_target * total)}).
    <div class="bar" aria-hidden="true"><i style="width:${(100 * f.found) / total}%"></i></div>
    Episode ends at step ${run.steps}: ${run.end_reason.replace("_", " ")}. Active robots: ${f.agents.filter((a) => a.active).length} of ${f.agents.length}.`;
  $("robots").innerHTML = "<tr><th>Robot</th><th>Battery</th><th>Status</th></tr>" + f.agents.map((a) =>
    `<tr><td><span class="sw" style="background:${color[a.id]}"></span>${a.id.replace("_", " ")}</td>
      <td>${a.energy} / ${a.capacity}<div class="bar" aria-hidden="true"><i style="width:${(100 * a.energy) / a.capacity}%;background:${a.energy / a.capacity < 0.25 ? "#d9a21b" : "#2e7d4f"}"></i></div></td>
      <td>${a.active ? "Active" : '<span class="dead">Disabled</span>'}</td></tr>`).join("");
  const failures = run.frames.slice(1, frame + 1).flatMap((fr) => fr.disabled.map((d) => `${d.agent.replace("_", " ")} at step ${fr.step} (${d.reason.replace("_", " ")})`));
  $("stepstats").innerHTML = [["Messages sent", f.messages.length], ["Delivered", f.delivered], ["Dropped", f.dropped],
    ["Detection reports", f.detections.length], ["False reports", f.detections.filter((d) => !d.true).length],
    ["Perception errors", f.perception_errors], ["Failures so far", failures.join("; ") || "None"]]
    .map(([k, v]) => `<tr><td>${k}</td><td class="n">${v}</td></tr>`).join("");
}

// ------------------------------------------------------------------ archived results
async function results() {
  let r;
  try { r = await getJSON("data/results.json"); } catch (e) {
    $("r-nominal-note").innerHTML = `<span class="err">Results could not be loaded (${e.message}).</span>`; return;
  }
  const p7 = r.phase7, src = (p) => `<a href="${REPO}${p}">${p}</a>`;
  // search time chart
  const rows = p7.nominal.filter((x) => x.time_censored);
  const W = 560, L = 220, rowH = 36, H = rows.length * rowH + 40, hi = Math.max(...rows.map((x) => x.time_censored.ci_high));
  const X = (v) => L + ((W - L - 60) * v) / hi;
  const svg = $("r-nominal"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = rows.map((x, k) => { const t = x.time_censored, y = 8 + k * rowH, ref = x.policy === "assignment";
    return `<text x="${L - 8}" y="${y + 17}" text-anchor="end" font-size="13" fill="#1c2630">${label[x.policy] || x.policy}</text>
      <rect x="${L}" y="${y + 4}" width="${X(t.mean) - L}" height="18" fill="${ref ? "#9aa6b1" : "#2f5d8a"}"><title>${fmt(t.mean)} steps (95% CI ${fmt(t.ci_low)}–${fmt(t.ci_high)})</title></rect>
      <line x1="${X(t.ci_low)}" x2="${X(t.ci_high)}" y1="${y + 13}" y2="${y + 13}" stroke="#1c2630" stroke-width="1.5"/>
      <text x="${X(t.ci_high) + 6}" y="${y + 17}" font-size="12" fill="#4a5866">${fmt(t.mean)}</text>`; }).join("") +
    `<text x="${L}" y="${H - 6}" font-size="12" fill="#4a5866">Mean steps to find 75% of survivors, 95% interval (lower is faster)</text>`;
  $("r-nominal-note").innerHTML = `Final validation, nominal scenario, evaluation seeds ${p7.seeds} (n = 50 each). Source: ${src(p7.source)}.`;
  // verdict families
  const fam = { C: "Shared belief maps faster than independent search", A: "Auction faster than shared belief maps alone",
    L: "Learned (MAPPO) differs from shared belief maps", R: "Reassignment leaves fewer unresolved sectors", S: "Auction raises success at full recall" };
  $("r-families").innerHTML = "<tr><th>Hypothesis family</th><th class='n'>Supported</th><th class='n'>Inconclusive</th></tr>" +
    Object.entries(p7.families).map(([k, v]) => { const total = Object.values(v).reduce((a, b) => a + b, 0);
      const yes = (v.supported || 0) + (v["difference detected"] || 0);
      return `<tr><td>${fam[k] || k}</td><td class="n">${yes} of ${total}</td><td class="n">${v.inconclusive || 0}</td></tr>`; }).join("");
  $("r-families-note").innerHTML = `54 tests, Holm-corrected; no hypothesis was contradicted. "Inconclusive" is not evidence of no effect. Source: ${src(p7.source)}.`;
  // map sizes
  const maps = { map_15: "15 × 15", nominal: "20 × 20", map_30: "30 × 30", map_40: "40 × 40" };
  const pol = ["frontier", "frontier_shared", "auction_comm", "assignment"];
  $("r-maps").innerHTML = `<tr><th>Map</th>${pol.map((p) => `<th class="n">${(label[p] || p).replace(" (reference)", "")}</th>`).join("")}</tr>` +
    p7.maps.map((m) => `<tr><td class="nowrap">${maps[m.condition]}</td>${pol.map((p) => `<td class="n">${m.success[p] ? Math.round(100 * m.success[p].mean) + "%" : "—"}</td>`).join("")}</tr>`).join("");
  $("r-maps-src").href = REPO + p7.source;
  // benchmark
  const b = r.benchmark.robots, BW = 560, BH = b.length * 34 + 40, bhi = Math.max(...b.map((x) => x.ms_per_step)), BX = (v) => 110 + ((BW - 170) * v) / bhi;
  const bs = $("r-bench"); bs.setAttribute("viewBox", `0 0 ${BW} ${BH}`);
  bs.innerHTML = b.map((x, k) => { const y = 8 + k * 34; return `<text x="102" y="${y + 17}" text-anchor="end" font-size="13" fill="#1c2630">${x.case}</text>
    <rect x="110" y="${y + 4}" width="${BX(x.ms_per_step) - 110}" height="18" fill="#2f5d8a"/>
    <text x="${BX(x.ms_per_step) + 6}" y="${y + 17}" font-size="12" fill="#4a5866">${fmt(x.ms_per_step)} ms</text>`; }).join("") +
    `<text x="110" y="${BH - 6}" font-size="12" fill="#4a5866">Milliseconds per simulation step, auction policy</text>`;
  const env = r.benchmark.environment;
  $("r-bench-note").innerHTML = `Exploratory timing on one machine (Python ${env.python}, ${env.cpu_count} CPUs), not a validated result. Per-step cost grows faster than team size because robots exchange records pairwise. Source: ${src(r.benchmark.source)}.`;
}

boot();
results();
