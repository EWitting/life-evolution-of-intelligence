// Page components of the book, shared by every stage page. Plain JavaScript on the static data files that
// `python -m life.book export <stage>` writes (book/data/<stage>.json); nothing here contains a number.
//
// A page places a component with      <div data-life="curves" data-stage="1.1"></div>
// Components: status, summary, curves, lesions, brain, runs, settings, strip (with data-example="<extras name>").
// The brain diagram takes its layout from data/brain_layout.json (tools/brain_layout) when that has the stage, else
// columns by depth as in life/dashboard.html; the edge routing follows life/dashboard.html (the tool scores it too).
(function () {
  'use strict';
  const cache = {};
  const load = stage => cache[stage] || (cache[stage] = fetch(`data/${stage}.json`).then(r => {
    if (!r.ok) throw new Error(`no data file for stage ${stage} (python -m life.book export ${stage})`);
    return r.json();
  }));
  const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const fmt = (v, d) => {
    if (v === null || v === undefined || Number.isNaN(v)) return '–';
    const a = Math.abs(v), n = d !== undefined ? d : a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 3;
    return v.toFixed(n);
  };
  const pm = s => s ? `${fmt(s.mean)}${s.se !== null && s.se !== undefined ? ` <span class="lb-se">± ${fmt(s.se)}</span>` : ''}` : '–';
  const signed = s => s ? `${s.mean > 0 ? '+' : ''}${fmt(s.mean)}${s.se !== null && s.se !== undefined ? ` <span class="lb-se">± ${fmt(s.se)}</span>` : ''}` : '–';
  const seeds = s => s ? `<div class="lb-seeds">${s.seeds.map(v => fmt(v)).join(' · ')}</div>` : '';
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };

  // one tooltip for the whole page
  let tip;
  function showTip(ev, html) {
    if (!tip) { tip = el('div', 'lb-tip'); document.body.appendChild(tip); }
    tip.innerHTML = html; tip.style.display = 'block';
    const w = tip.offsetWidth, x = Math.min(ev.clientX + 14, window.innerWidth - w - 8);
    const y = Math.min(ev.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
    tip.style.left = Math.max(4, x) + 'px'; tip.style.top = Math.max(4, y) + 'px';
  }
  function hideTip() { if (tip) tip.style.display = 'none'; }
  // details on hover, and on a tap on a phone; a tap anywhere else closes them
  let leaveTip = hideTip;
  function hover(node, show, leave = hideTip) {
    node.addEventListener('mousemove', show);
    node.addEventListener('mouseleave', leave);
    node.addEventListener('click', e => { e.stopPropagation(); if (leaveTip !== leave) leaveTip(); leaveTip = leave; show(e); });
  }
  document.addEventListener('click', () => { leaveTip(); leaveTip = hideTip; });

  const METRICS = {fit_mean: 'fitness (well-fed lifetime)', alive_ticks: 'lifetime (ticks)', eaten: 'food eaten',
    pain: 'pain received', poison_frac: 'share of berries eaten that were poison', temp_mean: 'body temperature',
    survivors: 'alive at the end of life'};
  const nice = (lo, hi, n) => {   // round tick values covering [lo, hi]
    const span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10));
    return out;
  };

  // ------------------------------------------------------------------ status: what the numbers on the page rest on
  function status(root, D) {
    const ev = D.evaluation, n = D.runs.main.length, notes = [];
    if (D.provisional && D.provisional.length) notes.push(`<b>Provisional.</b> These runs were made with settings that differ from the stage as it is defined now (${D.provisional.map(esc).join('; ')}). The numbers will change when the stage is rerun.`);
    const founder = !(D.runs.control && D.runs.control.length);   // run once on purpose; the page says so in its text
    if (n < 3 && !founder) notes.push(`<b>${n} seed${n === 1 ? '' : 's'}.</b> A single run is not evidence: a stage is judged over three seeds.`);
    if (!ev) notes.push('Lesions, head to head and the re-evaluation have not been exported for these runs yet.');
    else if (ev.stale) notes.push('Lesions, head to head and the re-evaluation below were measured on earlier runs than the curves.');
    root.innerHTML = notes.length ? `<div class="lb-note">${notes.map(t => `<p>${t}</p>`).join('')}</div>` : '';
  }

  // ------------------------------------------------------------------ results table
  function summary(root, D) {
    const S = D.summary, ev = D.evaluation || {}, hasC = !!S.control;
    const rows = [[`Mean of the last ${S.window} generations of evolution`, S.main && S.main.fit_mean, S.control && S.control.fit_mean, S.difference]];
    if (ev.reevaluated) rows.push([`Final population, re-evaluated in ${ev.worlds} shared worlds`, ev.reevaluated.main, ev.reevaluated.control, ev.reevaluated.difference]);
    if (ev.versus) rows.push([`Head to head: half main, half control, in the same ${ev.worlds} worlds`, null, null, ev.versus]);
    let h = `<table class="lb-table"><thead><tr><th>Fitness</th><th>main</th>${hasC ? '<th>control</th><th>main − control</th>' : ''}</tr></thead><tbody>`;
    for (const [name, m, c, d] of rows) {
      h += `<tr><td>${name}</td><td>${m ? pm(m) + seeds(m) : ''}</td>` + (hasC ? `<td>${c ? pm(c) + seeds(c) : ''}</td><td>${d ? signed(d) + seeds(d) : ''}</td>` : '') + '</tr>';
    }
    const other = ['alive_ticks', 'eaten', 'pain', 'poison_frac'].filter(k => S.main && S.main[k]);
    for (const k of other) h += `<tr class="lb-minor"><td>${METRICS[k]}, last ${S.window} generations</td><td>${pm(S.main[k])}</td>${hasC ? `<td>${S.control[k] ? pm(S.control[k]) : ''}</td><td></td>` : ''}</tr>`;
    h += `</tbody></table><div class="lb-caption">Mean ± standard error over ${D.runs.main.length} seed${D.runs.main.length === 1 ? '' : 's'}; the small figures are the seeds. Main = the brain of this stage${hasC ? ', control = the previous stage\'s brain in the same world, from the same starting population' : ''}.</div>`;
    root.innerHTML = h;
  }

  // ------------------------------------------------------------------ fitness curves: every seed thin, the mean thick
  function curves(root, D) {
    const groups = [['main', 'lb-s1'], ['control', 'lb-s2']].filter(([g]) => D.curves[g] && D.curves[g].length);
    const keys = Object.keys(METRICS).filter(k => groups.some(([g]) => D.curves[g][0][k]));
    let metric = root.dataset.metric || 'fit_mean';
    const head = el('div', 'lb-controls'), plot = el('div', 'lb-plot');
    head.innerHTML = `<label>Show <select>${keys.map(k => `<option value="${k}"${k === metric ? ' selected' : ''}>${METRICS[k]}</option>`).join('')}</select></label>`
      + `<span class="lb-legend">${groups.map(([g, c]) => `<span><i class="lb-key ${c}"></i>${g}</span>`).join('')}<span class="lb-muted">thin = one seed, thick = mean of the seeds</span></span>`;
    root.append(head, plot);
    head.querySelector('select').addEventListener('change', e => { metric = e.target.value; draw(); });
    const smooth = (a, w) => a.map((_, i) => mean(a.slice(Math.max(0, i - w), i + w + 1)));
    function draw() {
      const W = 720, H = 300, m = {l: 48, r: 64, t: 12, b: 34};
      const series = groups.map(([g, c]) => {
        const runs = D.curves[g].filter(r => r[metric]).map(r => smooth(r[metric], 2));
        const n = Math.min(...runs.map(r => r.length));
        return {g, c, runs, mean: Array.from({length: n}, (_, i) => mean(runs.map(r => r[i])))};
      }).filter(s => s.runs.length);
      const all = series.flatMap(s => s.runs.flat()), G = Math.max(...series.flatMap(s => s.runs.map(r => r.length)));
      const lo = Math.min(0, ...all), hi = Math.max(...all) * 1.04 || 1;
      const x = g => m.l + g * (W - m.l - m.r) / Math.max(1, G - 1), y = v => H - m.b - (v - lo) * (H - m.t - m.b) / (hi - lo);
      const path = a => a.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
      let s = `<svg viewBox="0 0 ${W} ${H}" class="lb-svg" role="img" aria-label="${esc(METRICS[metric])} over generations">`;
      for (const v of nice(lo, hi, 5)) s += `<line class="lb-grid" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text class="lb-tick" x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
      for (const g of nice(0, G - 1, 6)) s += `<text class="lb-tick" x="${x(g)}" y="${H - m.b + 16}" text-anchor="middle">${g}</text>`;
      s += `<line class="lb-axis" x1="${m.l}" x2="${W - m.r}" y1="${y(lo)}" y2="${y(lo)}"/><text class="lb-tick" x="${(m.l + W - m.r) / 2}" y="${H - 2}" text-anchor="middle">generation</text>`;
      for (const q of series) for (const r of q.runs) s += `<path class="lb-line lb-thin ${q.c}" d="${path(r)}"/>`;
      // direct labels at the line ends, pushed apart when the means end close together
      const ends = series.map(q => ({q, y: y(q.mean[q.mean.length - 1])})).sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
      for (const q of series) s += `<path class="lb-line ${q.c}" d="${path(q.mean)}"/>`;
      for (const e of ends) s += `<text class="lb-label" x="${x(e.q.mean.length - 1) + 8}" y="${e.y + 4}">${e.q.g}</text>`;
      s += `<g class="lb-cross" style="display:none"><line class="lb-axis" y1="${m.t}" y2="${H - m.b}"/>${series.map(q => `<circle class="lb-dot ${q.c}" r="4.5"/>`).join('')}</g>`;
      s += `<rect class="lb-hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}"/></svg>`;
      plot.innerHTML = s;
      const svg = plot.firstChild, cross = svg.querySelector('.lb-cross'), hit = svg.querySelector('.lb-hit');
      hover(hit, ev => {
        const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
        const g = Math.max(0, Math.min(G - 1, Math.round((px - m.l) * (G - 1) / (W - m.l - m.r))));
        cross.style.display = ''; cross.querySelector('line').setAttribute('x1', x(g)); cross.querySelector('line').setAttribute('x2', x(g));
        cross.querySelectorAll('circle').forEach((c, i) => { const v = series[i].mean[Math.min(g, series[i].mean.length - 1)]; c.setAttribute('cx', x(g)); c.setAttribute('cy', y(v)); });
        showTip(ev, `<b>generation ${g}</b>` + series.map(q => `<div><i class="lb-key ${q.c}"></i>${q.g} <b>${fmt(q.mean[Math.min(g, q.mean.length - 1)])}</b> <span class="lb-muted">${q.runs.map(r => fmt(r[Math.min(g, r.length - 1)])).join(' · ')}</span></div>`).join(''));
      }, () => { cross.style.display = 'none'; hideTip(); });
    }
    draw();
  }

  // ------------------------------------------------------------------ lesions: is the circuit used?
  function lesions(root, D) {
    const ev = D.evaluation;
    if (!ev || !ev.lesions) { root.innerHTML = '<div class="lb-note"><p>No lesion data exported for this stage yet.</p></div>'; return; }
    const isNew = Object.fromEntries(D.brain.regions.map(r => [r.name, r.new]));
    const LABEL = {no_plasticity: 'learning switched off', no_depression: 'depression switched off'};
    const rows = Object.entries(ev.lesions).map(([name, v]) => ({name, p: v.percent}));
    const hi = Math.max(120, ...rows.flatMap(r => r.p.seeds)) * 1.02;
    const W = 720, RH = 30, m = {l: 150, r: 150, t: 22, b: 30}, H = m.t + rows.length * RH + m.b;
    const x = v => m.l + Math.max(0, v) * (W - m.l - m.r) / hi;
    let s = `<svg viewBox="0 0 ${W} ${H}" class="lb-svg" role="img" aria-label="fitness with each region silenced, in percent of the intact population">`;
    for (const v of nice(0, hi, 6)) s += `<line class="lb-grid" x1="${x(v)}" x2="${x(v)}" y1="${m.t}" y2="${H - m.b}"/><text class="lb-tick" x="${x(v)}" y="${H - m.b + 15}" text-anchor="middle">${v}</text>`;
    s += `<line class="lb-ref" x1="${x(100)}" x2="${x(100)}" y1="${m.t - 6}" y2="${H - m.b}"/><text class="lb-tick" x="${x(100)}" y="${m.t - 10}" text-anchor="middle">intact = 100</text>`;
    s += `<text class="lb-tick" x="${(m.l + W - m.r) / 2}" y="${H - 1}" text-anchor="middle">fitness with the region silenced, % of intact</text>`;
    rows.forEach((r, i) => {
      const cy = m.t + i * RH + RH / 2, w = x(r.p.mean) - m.l;
      s += `<g class="lb-row" data-i="${i}"><rect class="lb-hit" x="0" y="${cy - RH / 2}" width="${W}" height="${RH}"/>`
        + `<text class="lb-rowlabel" x="${m.l - 8}" y="${cy + 4}" text-anchor="end">${esc(LABEL[r.name] || r.name)}${isNew[r.name] ? ' <tspan class="lb-new">new</tspan>' : ''}</text>`
        + `<path class="lb-bar lb-s1" d="M${m.l},${cy - 7} h${Math.max(0, w - 4).toFixed(1)} a4,4 0 0 1 4,4 v6 a4,4 0 0 1 -4,4 h${(-Math.max(0, w - 4)).toFixed(1)} z"/>`
        + r.p.seeds.map(v => `<circle class="lb-seed" cx="${x(v).toFixed(1)}" cy="${cy}" r="4"/>`).join('')
        + `<text class="lb-value" x="${W - m.r + 10}" y="${cy + 4}">${fmt(r.p.mean, 0)}${r.p.se !== null ? ` ± ${fmt(r.p.se, 0)}` : ''}</text></g>`;
    });
    s += `<line class="lb-axis" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}"/></svg>`;
    root.innerHTML = s + `<div class="lb-caption">Each region is silenced in turn (no activity, no outgoing synapses) and the final population lives again in ${ev.worlds} shared worlds. Bar = mean over the seeds, dots = the seeds. A region that the animals use costs fitness when silenced; a bar at 100 means the population does as well without it.</div>`;
    root.querySelectorAll('.lb-row').forEach(g => {
      const r = rows[+g.dataset.i];
      hover(g, e => showTip(e, `<b>${esc(r.name)}</b> silenced<div>${fmt(r.p.mean, 1)}% of intact (± ${fmt(r.p.se, 1)})</div><div class="lb-muted">per seed: ${r.p.seeds.map(v => fmt(v, 0)).join(' · ')}</div>`));
    });
  }

  // ------------------------------------------------------------------ brain diagram
  // the layout shared by all stages (tools/brain_layout writes data/brain_layout.json); without it, or for a stage it
  // does not have, the diagram is laid out in columns by depth as before
  let layoutP;
  const loadLayout = () => layoutP || (layoutP = fetch('data/brain_layout.json').then(r => r.ok ? r.json() : null).catch(() => null));
  function brain(root, D) { return loadLayout().then(L => drawBrain(root, D, L)); }
  function drawBrain(root, D, L) {
    const REG = D.brain.regions, PROJ = D.brain.projections, BW = 122, BH = 38, pos = {};
    const A = L && L.stages && L.stages[D.stage.key], atlas = !!A && REG.every(r => A.pos[r.name]);
    // cells and group boxes take the colour of their top-level group: the layout file's hues (the same in every
    // chapter), else one from the group's name
    const hues = (L && L.hues) || {};
    const hueOf = g => hues[g] !== undefined ? hues[g] : [...g].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 0);
    const blk = {}, sock = {};   // with the layout: the senses / motor blocks' extent [y0, y1] and their sockets, in px
    let AH, SVGW;
    if (atlas) {
      const CW = BW + 110, STEP = BH + 26, rows = [...Object.values(A.pos).map(p => p[1]), ...Object.values(A.blocks).flat()];
      const W0 = 40 + BW + (A.cols - 1) * CW;   // at least as wide as the depth layout: small brains are not blown up
      SVGW = Math.max(720, W0);
      const top = Math.min(...rows) - 0.5, X = c => (SVGW - W0) / 2 + 20 + BW / 2 + c * CW, Y = r => 30 + (r - top) * STEP;
      // a socket sits on its block's inner edge (the right edge of the senses, the left edge of the motor neurons)
      for (const [n, [c, r]] of Object.entries(A.pos)) if (n.includes(':')) sock[n] = [X(c) + (n.startsWith('in:') ? BW / 2 : -BW / 2), Y(r)]; else pos[n] = [X(c), Y(r)];
      for (const [n, [b0, b1]] of Object.entries(A.blocks)) blk[n] = [Y(b0), Y(b1)];
      AH = Y(Math.max(...rows) + 0.5);
    } else {
    const depth = {in: 0}, q = ['in'];
    while (q.length) { const u = q.shift(); for (const p of PROJ) if (p.src === u && depth[p.dst] === undefined) { depth[p.dst] = depth[u] + 1; q.push(p.dst); } }
    const inner = REG.filter(r => r.name !== 'in' && r.name !== 'out');
    inner.forEach(r => { if (!(depth[r.name] >= 1)) depth[r.name] = 1; });
    const top = r => (r.name === 'in' || r.name === 'out') ? '#' + r.name : (r.group ? r.group.split('/')[0] : '#' + r.name);
    const units = new Map();
    REG.forEach(r => { const k = top(r); if (!units.has(k)) units.set(k, []); units.get(k).push(r); });
    for (const mem of units.values()) mem.sort((a, b) => (a.group || '').localeCompare(b.group || ''));
    const maxd = Math.max(1, ...inner.map(r => depth[r.name]));
    const ud = k => k === '#in' ? 0 : k === '#out' ? maxd + 1 : Math.round(mean(units.get(k).map(r => depth[r.name])));
    const colsOf = {};
    for (const k of units.keys()) (colsOf[ud(k)] = colsOf[ud(k)] || []).push(k);
    const colKeys = Object.keys(colsOf).map(Number).sort((a, b) => a - b);
    const STEP = BH + 26, GAP = 40, SUB = 26;   // SUB: extra room where a sub-group starts (its box and label)
    const subs = mem => mem.filter((r, i) => i && r.group !== mem[i - 1].group).length;
    const colH = colKeys.map(c => colsOf[c].reduce((s, k) => s + units.get(k).length * STEP + subs(units.get(k)) * SUB + GAP, -GAP));
    AH = Math.max(160, Math.max(...colH) + 70); SVGW = Math.max(720, 40 + colKeys.length * (BW + 110) - 110);
    colKeys.forEach((c, ci) => {
      const x = colKeys.length > 1 ? 20 + BW / 2 + ci * (SVGW - 40 - BW) / (colKeys.length - 1) : SVGW / 2;
      let y = (AH - colH[ci]) / 2 + BH / 2 + 16;
      for (const k of colsOf[c]) { units.get(k).forEach((r, i, mem) => { if (i && r.group !== mem[i - 1].group) y += SUB; pos[r.name] = [x, y]; y += STEP; }); y += GAP; }
    });
    }
    const clip = (cx, cy, dx, dy) => { const s = Math.min(dx ? (BW / 2 + 4) / Math.abs(dx) : 1e9, dy ? (BH / 2 + 4) / Math.abs(dy) : 1e9); return [cx + dx * s, cy + dy * s]; };
    const inBox = (x, y, n) => Math.abs(x - pos[n][0]) < BW / 2 + 6 && Math.abs(y - pos[n][1]) < BH / 2 + 6;
    const f1 = v => v.toFixed(1);
    // with the layout, the lines of a projection from the senses start at a socket per sense it reads, and a projection
    // onto some of the actions ends at that group's socket; a projection onto all of out ends on the motor block
    const actions = D.brain.actions || [];
    function ends(p) {
      if (!atlas) return [[p.src, p.dst]];
      let d = p.dst;
      const [lo, hi] = p.dst_range;
      if (p.dst === 'out' && p.dst_range.length && !(lo === 0 && hi >= actions.length)) {
        const s = 'out:' + actions.slice(lo, hi).map(a => a.toLowerCase()).join('/');
        if (sock[s]) d = s;
      }
      const ss = p.src === 'in' && p.src_select.length ? p.src_select.map(s => 'in:' + (/^[a-z_]+/i.exec(s) || ['?'])[0]).map(s => sock[s] ? s : 'in') : [p.src];
      return [...new Set(ss)].map(s => [s, d]);
    }
    const ENDS = PROJ.map(ends);
    const centre = n => sock[n] || pos[n];
    // where a line attaches: a region's centre (clipped to its box later), a socket, or a block's inner edge at the
    // height of the line's other end
    // the senses / motor neurons drawn as one small box (layout mode 'point') are a box like any region
    const modes = (L && L.modes) || {}, isBlock = n => atlas && !!blk[n] && modes[n] !== 'point';
    const attach = (n, other) => sock[n] || (isBlock(n)
      ? [pos[n][0] + (n === 'in' ? BW / 2 : -BW / 2), Math.min(blk[n][1], Math.max(blk[n][0], other[1]))] : pos[n]);
    const isBox = n => !sock[n] && !isBlock(n);
    const OBST = atlas ? REG.filter(r => r.name !== 'in' && r.name !== 'out') : REG;   // boxes a line bends around
    function geom(p, a, b) {
      if (a === b) { const [x1, y1] = pos[a], x = x1 + BW / 2 - 30, y = y1 - BH / 2; return {d: `M${f1(x)},${f1(y)} C${f1(x)},${f1(y - 32)} ${f1(x + 46)},${f1(y - 32)} ${f1(x + 30)},${f1(y - 3)}`, end: [x + 30, y - 3], dir: [-0.45, 0.9]}; }
      const [x1, y1] = attach(a, centre(b)), [x2, y2] = attach(b, centre(a));
      const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1, nx = -dy / len, ny = dx / len;
      const has = (s, d) => o => ENDS[PROJ.indexOf(o)].some(e => e[0] === s && e[1] === d);
      const twins = PROJ.filter(has(a, b)), ti = twins.indexOf(p);
      const base = (PROJ.some(has(b, a)) ? 14 : 0) + (twins.length > 1 ? (ti - (twins.length - 1) / 2) * 16 : 0);
      const hits = bb => OBST.some(r => r.name !== a && r.name !== b && [0.2, 0.35, 0.5, 0.65, 0.8].some(f => inBox(x1 + dx * f + nx * 4 * f * (1 - f) * bb, y1 + dy * f + ny * 4 * f * (1 - f) * bb, r.name)));
      let bend = base;
      for (let k = 1; k <= 10 && hits(bend); k++) { const sg = base ? Math.sign(base) : (k % 2 ? 1 : -1); bend = base + sg * Math.ceil(k / (base ? 1 : 2)) * 0.4 * BH; }
      const cx = (x1 + x2) / 2 + nx * 2 * bend, cy = (y1 + y2) / 2 + ny * 2 * bend;
      const [sx, sy] = isBox(a) ? clip(x1, y1, cx - x1, cy - y1) : [x1, y1], [ex, ey] = isBox(b) ? clip(x2, y2, cx - x2, cy - y2) : [x2, y2];
      const l = Math.hypot(ex - cx, ey - cy) || 1;
      return {d: `M${f1(sx)},${f1(sy)} Q${f1(cx)},${f1(cy)} ${f1(ex)},${f1(ey)}`, end: [ex, ey], dir: [(ex - cx) / l, (ey - cy) / l]};
    }
    const maxDrive = Math.max(1e-6, ...PROJ.map(p => p.drive));
    const H = Math.max(AH, ...Object.values(pos).map(p => p[1] + BH / 2 + 20));
    let s = `<svg viewBox="0 0 ${SVGW} ${H}" class="lb-svg lb-brain" role="img" aria-label="brain diagram: regions and projections">`;
    // groups: labelled boxes around their members
    const paths = new Set();
    let labels = '';   // drawn after the arrows, so that an arrow never hides a group's name
    REG.forEach(r => { if (r.group) r.group.split('/').forEach((_, i, a) => paths.add(a.slice(0, i + 1).join('/'))); });
    [...paths].sort((a, b) => a.split('/').length - b.split('/').length).forEach(gp => {
      const mem = REG.filter(r => r.group === gp || r.group.startsWith(gp + '/'));
      const pad = 7 + 9 * (Math.max(...mem.map(r => r.group.split('/').length)) - gp.split('/').length);
      const xs = mem.map(r => pos[r.name][0]), ys = mem.map(r => pos[r.name][1]);
      const x0 = Math.min(...xs) - BW / 2 - pad, x1 = Math.max(...xs) + BW / 2 + pad, y0 = Math.min(...ys) - BH / 2 - pad - 12, y1 = Math.max(...ys) + BH / 2 + pad;
      s += `<rect class="lb-group${gp.includes('/') ? '' : ' lb-gtop'}" style="--h:${hueOf(gp)}" x="${f1(x0)}" y="${f1(y0)}" width="${f1(x1 - x0)}" height="${f1(y1 - y0)}" rx="9"/>`;
      labels += `<text class="lb-grouplabel" x="${f1(x0 + 7)}" y="${f1(y0 + 12)}">${esc(gp.split('/').pop())}</text>`;
    });
    // modulators: a dashed line from the cell that releases one to every cell with a receptor for it, and to every
    // projection whose learning it gates (the line ends on the middle of that projection)
    const mid = {}, MODS = [];
    PROJ.forEach((p, i) => { const m = /^M([-\d.]+),([-\d.]+) Q([-\d.]+),([-\d.]+) ([-\d.]+),([-\d.]+)$/.exec(geom(p, ...ENDS[i][0]).d); if (m) mid[i] = [(+m[1] + 2 * +m[3] + +m[5]) / 4, (+m[2] + 2 * +m[4] + +m[6]) / 4]; });
    for (const m of D.brain.modulators || []) {
      if (!pos[m.pos]) continue;
      for (const r of REG) for (const rc of r.receptors || []) if (rc[0] === m.name) MODS.push({m, src: m.pos, to: pos[r.name], dst: r.name, how: rc[1], sens: rc[2]});
      PROJ.forEach((p, i) => { if (p.modulator === m.name && mid[i]) MODS.push({m, src: m.pos, to: mid[i], proj: p}); });
    }
    function modGeom(l) {
      const [x1, y1] = pos[l.src], [x2, y2] = l.to, dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1, nx = -dy / len, ny = dx / len;
      if (l.dst && Math.abs(dx) < 1 && Math.abs(dy) < 2.5 * BH) {   // neighbours in one column: a loop around their right side
        const x = x1 + BW / 2, b = 44;
        return {d: `M${f1(x)},${f1(y1)} C${f1(x + b)},${f1(y1)} ${f1(x + b)},${f1(y2)} ${f1(x + 5)},${f1(y2)}`, end: [x + 5, y2], mid: [x + b * 0.75 + 4, (y1 + y2) / 2], anchor: 'start'};
      }
      const base = l.dst && PROJ.some(o => (o.src === l.src && o.dst === l.dst) || (o.src === l.dst && o.dst === l.src)) ? 24 : 0;
      const hits = b => OBST.some(r => r.name !== l.src && r.name !== l.dst && [0.2, 0.35, 0.5, 0.65, 0.8].some(f => inBox(x1 + dx * f + nx * 4 * f * (1 - f) * b, y1 + dy * f + ny * 4 * f * (1 - f) * b, r.name)));
      let bend = base;
      for (let k = 1; k <= 10 && hits(bend); k++) { const sg = base ? Math.sign(base) : (k % 2 ? 1 : -1); bend = base + sg * Math.ceil(k / (base ? 1 : 2)) * 0.4 * BH; }
      const cx = (x1 + x2) / 2 + nx * 2 * bend, cy = (y1 + y2) / 2 + ny * 2 * bend;
      const [sx, sy] = clip(x1, y1, cx - x1, cy - y1), [ex, ey] = l.dst ? clip(x2, y2, cx - x2, cy - y2) : [x2, y2];
      return {d: `M${f1(sx)},${f1(sy)} Q${f1(cx)},${f1(cy)} ${f1(ex)},${f1(ey)}`, end: [ex, ey], mid: [(sx + 2 * cx + ex) / 4, (sy + 2 * cy + ey) / 4]};
    }
    let modsvg = '';
    MODS.forEach((l, i) => {
      const g = modGeom(l);
      modsvg += `<g class="lb-mod" data-m="${i}"><path class="lb-modline" fill="none" d="${g.d}"/><circle class="lb-modend" cx="${f1(g.end[0])}" cy="${f1(g.end[1])}" r="4"/>`
        + `<text class="lb-modlabel" text-anchor="${g.anchor || 'middle'}" x="${f1(g.mid[0])}" y="${f1(g.mid[1] + 3.5)}">${esc(l.m.name)}</text><path d="${g.d}" fill="none" stroke="transparent" stroke-width="14"/></g>`;
    });
    PROJ.forEach((p, i) => ENDS[i].forEach(([from, to]) => {
      const g = geom(p, from, to), w = 1 + 4 * Math.sqrt(p.drive / maxDrive), cls = p.rule !== 'fixed' ? 'lb-learned' : p.hard ? 'lb-hard' : 'lb-evolved', [ex, ey] = g.end, [ux, uy] = g.dir;
      // excitatory synapses end in an arrowhead, inhibitory ones in a bar; sensory projections (mixed signs) in a dot
      const fromIn = p.src === 'in' && (!p.hard || p.sign === 0), a = 7 + w;
      const head = fromIn ? `<circle class="${cls}-fill" cx="${f1(ex)}" cy="${f1(ey)}" r="${f1(2.5 + w / 2)}"/>`
        : p.sign < 0 ? `<line class="${cls}" stroke-width="${f1(w + 1.5)}" x1="${f1(ex - uy * a)}" y1="${f1(ey + ux * a)}" x2="${f1(ex + uy * a)}" y2="${f1(ey - ux * a)}"/>`
        : `<path class="${cls}-fill" d="M${f1(ex)},${f1(ey)} L${f1(ex - ux * a * 1.5 - uy * a * 0.6)},${f1(ey - uy * a * 1.5 + ux * a * 0.6)} L${f1(ex - ux * a * 1.5 + uy * a * 0.6)},${f1(ey - uy * a * 1.5 - ux * a * 0.6)} z"/>`;
      s += `<g class="lb-edge${p.new ? ' lb-isnew' : ''}" data-p="${i}"><path class="${cls}" fill="none" stroke-width="${f1(w)}" d="${g.d}"/>${head}<path d="${g.d}" fill="none" stroke="transparent" stroke-width="14"/></g>`;
    }));
    s += modsvg + labels;
    for (const r of REG) {
      if (isBlock(r.name)) {   // the senses / motor block: name above, a labelled socket per sense or action group
        const [b0, b1] = blk[r.name], x0 = pos[r.name][0] - BW / 2, SW = 104, SH = 18;
        s += `<g class="lb-reg" data-r="${esc(r.name)}"><rect x="${f1(x0)}" y="${f1(b0)}" width="${BW}" height="${f1(b1 - b0)}" rx="6" class="lb-io"/>`
          + `<text class="lb-regname" x="${f1(x0 + 2)}" y="${f1(b0 - 6)}">${esc(r.name)} <tspan class="lb-regsub">${r.name === 'in' ? 'senses' : 'motor neurons'} · ${r.size}</tspan></text>`;
        for (const [n, [x, y]] of Object.entries(sock)) if (n.startsWith(r.name + ':')) {
          const sx = r.name === 'in' ? x - SW : x;
          s += `<rect class="lb-sock" x="${f1(sx)}" y="${f1(y - SH / 2)}" width="${SW}" height="${SH}" rx="4"/><text class="lb-regsub" x="${f1(sx + SW / 2)}" y="${f1(y + 3.5)}" text-anchor="middle">${esc(n.slice(r.name.length + 1))}</text>`;
        }
        s += '</g>';
        continue;
      }
      const [x, y] = pos[r.name];
      s += `<g class="lb-reg${r.new ? ' lb-isnew' : ''}" data-r="${esc(r.name)}" transform="translate(${f1(x - BW / 2)},${f1(y - BH / 2)})"><rect width="${BW}" height="${BH}" rx="6" ${r.group ? `class="lb-cell" style="--h:${hueOf(r.group.split('/')[0])}"` : 'class="lb-io"'}/>`
        + `<text class="lb-regname" x="8" y="16">${esc(r.name)}</text><text class="lb-regsub" x="8" y="30">${r.name === 'in' ? 'senses' : r.name === 'out' ? 'motor neurons' : (r.sign > 0 ? 'excitatory' : 'inhibitory')} · ${r.size}</text></g>`;
    }
    s += '</svg>';
    const anyNew = REG.some(r => r.new);
    root.innerHTML = s + `<div class="lb-legend lb-brainlegend"><span><i class="lb-key lb-hardkey"></i>hard-wired (designed; evolution tunes only the strength)</span><span><i class="lb-key lb-evokey"></i>evolved</span>${PROJ.some(p => p.rule !== 'fixed') ? '<span><i class="lb-key lb-learnkey"></i>learned within a life</span>' : ''}<span>▸ excites</span><span>⊣ inhibits</span><span>● senses, mixed signs</span>${MODS.length ? '<span><i class="lb-key lb-modkey"></i>modulator: released by one cell, acts on a cell or on the learning of a projection</span>' : ''}${anyNew ? '<span><i class="lb-key lb-newkey"></i>added in this stage</span>' : ''}<span class="lb-muted">line width = total synaptic strength arriving at one target neuron, mean of the final population</span></div>`;
    // "new circuit": only what this stage added and what it connects to directly: new projections, projections into
    // or out of a new region, and the modulator lines of a new modulator or of a new region; a modulator line onto a
    // projection brings that projection along. Regions stay in place; the ones none of it touches are faded.
    const newReg = new Set(REG.filter(r => r.new).map(r => r.name));
    const showP = new Set(PROJ.map((p, i) => p.new || newReg.has(p.src) || newReg.has(p.dst) ? i : -1).filter(i => i >= 0));
    const showM = new Set(MODS.map((l, i) => l.m.new || newReg.has(l.src) || newReg.has(l.dst) || (l.proj && showP.has(PROJ.indexOf(l.proj))) ? i : -1).filter(i => i >= 0));
    for (const i of showM) if (MODS[i].proj) showP.add(PROJ.indexOf(MODS[i].proj));
    if (showP.size || showM.size) {
      const used = new Set(newReg);
      for (const i of showP) used.add(PROJ[i].src).add(PROJ[i].dst);
      for (const i of showM) { used.add(MODS[i].src); if (MODS[i].dst) used.add(MODS[i].dst); }
      root.querySelectorAll('.lb-edge').forEach(g => g.classList.toggle('lb-ctx', !showP.has(+g.dataset.p)));
      root.querySelectorAll('.lb-mod').forEach(g => g.classList.toggle('lb-ctx', !showM.has(+g.dataset.m)));
      root.querySelectorAll('.lb-reg').forEach(g => g.classList.toggle('lb-ctx', !used.has(g.dataset.r)));
      const svg = root.querySelector('.lb-brain'), bar = el('div', 'lb-focusbar', '<span class="lb-muted">Show</span>'
        + '<button type="button" data-f="1" title="only what this stage added and what it connects to directly">new circuit</button>'
        + '<button type="button" data-f="0" title="every region and connection of this brain">everything</button>');
      const set = on => { svg.classList.toggle('lb-focus', on); bar.querySelectorAll('button').forEach(b => b.classList.toggle('lb-on', (b.dataset.f === '1') === on)); };
      bar.addEventListener('click', e => { if (e.target.dataset.f) set(e.target.dataset.f === '1'); });
      root.insertBefore(bar, svg); set(true);
    }
    const sel = p => p.src_select.length ? ` [${p.src_select.join(', ')}]` : '';
    root.querySelectorAll('.lb-edge').forEach(g => {
      const p = PROJ[+g.dataset.p];
      const range = p.dst === 'out' && p.dst_range.length ? ` (${D.brain.actions.slice(p.dst_range[0], p.dst_range[1]).join(', ')})` : '';
      hover(g, e => showTip(e, `<b>${esc(p.src + sel(p))} → ${esc(p.dst)}${esc(range)}</b><div>${p.rule !== 'fixed' ? `learned within a life (rule: ${esc(p.rule)}${p.modulator ? ', gated by ' + esc(p.modulator) : ''})` : p.hard ? 'hard-wired' : 'evolved'}, ${p.src === 'in' && (!p.hard || p.sign === 0) ? 'mixed signs' : p.sign < 0 ? 'inhibitory' : 'excitatory'}${p.new ? ', added in this stage' : ''}</div><div>${fmt(p.synapses, p.synapses < 10 ? 0 : 0)} synapses, mean strength ${fmt(p.mean_abs)}${p.designed !== null && p.designed !== undefined ? ` (designed at ${fmt(Math.abs(p.designed))})` : ''}</div>`));
    });
    const lasts = m => m.lasts ? `Released at once and cleared slowly: it stays for about ${fmt(m.lasts, 0)} ticks (designed at ${fmt(m.lasts_designed, 0)}).` : 'Follows the cell that releases it, without delay.';
    root.querySelectorAll('.lb-mod').forEach(g => {
      const l = MODS[+g.dataset.m];
      const acts = l.proj ? `gates the learning of ${esc(l.proj.src + sel(l.proj))} → ${esc(l.proj.dst)}` : `acts on ${esc(l.dst)} (${l.how === 'gain' ? 'scales its response' : 'adds to its drive'}, sensitivity ${fmt(l.sens)})`;
      hover(g, e => showTip(e, `<b>${esc(l.m.name)}</b> modulator${l.m.new ? ', added in this stage' : ''}<div>released by ${esc(l.src)}, ${acts}</div><div class="lb-muted">${lasts(l.m)}</div>`));
    });
    root.querySelectorAll('.lb-reg').forEach(g => {
      const r = REG.find(o => o.name === g.dataset.r), ins = PROJ.filter(p => p.dst === r.name), outs = PROJ.filter(p => p.src === r.name);
      const rel = (D.brain.modulators || []).filter(m => m.pos === r.name).map(m => m.name), rec = (r.receptors || []).map(rc => rc[0]);
      hover(g, e => showTip(e, `<b>${esc(r.name)}</b> ${r.size} neuron${r.size === 1 ? '' : 's'}${r.new ? ', added in this stage' : ''}<div class="lb-muted">from: ${esc([...new Set(ins.map(p => p.src))].join(', ') || '–')}</div><div class="lb-muted">to: ${esc([...new Set(outs.map(p => p.dst))].join(', ') || '–')}</div>${rel.length ? `<div class="lb-muted">releases: ${esc(rel.join(', '))}</div>` : ''}${rec.length ? `<div class="lb-muted">receptors for: ${esc(rec.join(', '))}</div>` : ''}`));
    });
  }

  // ------------------------------------------------------------------ runs and settings
  function runs(root, D) {
    const first = g => (D.runs[g] || []).find(r => r.dashboard);
    const btn = (g, label) => first(g) ? `<a class="lb-watch" href="${esc(first(g).dashboard)}" target="_blank">&#9654; ${label}</a>` : '';
    const watch = btn('main', 'Watch a main run') + btn('control', 'Watch a control run');
    let h = (watch ? `<div class="lb-watchrow">${watch}</div>` : '<div class="lb-caption">No run of this stage is published yet.</div>')
      + '<div class="lb-caption">A dashboard replays the last generation of a run: all animals in their world, and the brain of a few of them, tick by tick. It is a large page and opens in a new tab.</div>'
      + '<details class="lb-details"><summary>Run directories and dates</summary><table class="lb-table"><thead><tr><th></th><th>seed</th><th>run directory</th><th>generations</th><th>finished</th><th>dashboard</th></tr></thead><tbody>';
    for (const g of ['main', 'control']) for (const r of D.runs[g]) h += `<tr><td>${g}</td><td>${r.seed}</td><td><code>runs/${esc(r.dir)}</code></td><td>${r.generations}</td><td>${esc(r.finished.replace('T', ' '))}</td><td>${r.dashboard ? `<a href="${esc(r.dashboard)}">open</a>` : '<span class="lb-muted">not published</span>'}</td></tr>`;
    root.innerHTML = h + `</tbody></table><div class="lb-caption">One dashboard per stage is published (seed 0 of main and of control); the other seeds are in the run directories. Data exported ${esc(D.exported.replace('T', ' '))}.</div></details>`;
  }
  function settings(root, D) {
    const S = D.settings, w = S.world, e = S.evolution;
    const rows = [['World', `${w.height} × ${w.width} cells, ${w.num_agents} animals`], ['Objects', esc(S.objects.filter(n => !/Empty/.test(n) && /Bush|Onion|Spring/.test(n)).filter((n, i, a) => a.indexOf(n) === i).join(', '))],
      ['Stomach', `holds ${w.max_food}; born ${Math.round(100 * w.start_food)}% full; loses ${w.hunger_per_tick} per tick at rest, +${Math.round(100 * w.move_cost)}% when walking, +${Math.round(100 * w.turn_cost)}% when turning`],
      ['Eyes', `${S.vision.columns} directions over ${S.vision.fov_degrees}°, range ${S.vision.range} cells, ${S.vision.appearance_dim} numbers for the look of each object`],
      ['Life', `${e.ticks_per_generation} ticks, one life per individual`],
      ['Evolution', `${w.num_agents / e.siblings} genomes × ${e.siblings} siblings, ${e.generations} generations, ${e.crossover ? 'two parents per child' : 'one parent per child'}, a weight mutates with probability ${e.weight_mutation_prob}`],
      ['Learning within a life', e.plastic ? 'yes' : 'none: every weight is inherited']];
    // folded away by default: reference material, not part of the story (data-open on the placeholder opens it)
    root.innerHTML = `<details class="lb-details"${root.dataset.open !== undefined ? ' open' : ''}><summary>World and evolution settings of this stage</summary><table class="lb-table lb-kv"><tbody>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</tbody></table></details>`;
  }

  // ------------------------------------------------------------------ zoomed-in example: one animal, a few dozen ticks
  const GLYPH = {NOOP: '·', FORWARD: '↑', TURN_LEFT: '↶', TURN_RIGHT: '↷', USE: '◆', EAT: '◆', VOCALIZE: '♪'};
  const WORDS = {NOOP: 'waits', FORWARD: 'steps forward', TURN_LEFT: 'turns left', TURN_RIGHT: 'turns right', USE: 'bites', EAT: 'eats', VOCALIZE: 'calls'};
  function strip(root, D) {
    const X = D.extras && D.extras[root.dataset.example];
    if (!X) { root.innerHTML = '<div class="lb-note"><p>No example exported for this stage yet.</p></div>'; return; }
    const steps = X.steps, n = steps.length, R = X.radius, size = 2 * R + 1, CELL = 30;
    const rows = [];   // one row per input or neuron
    X.inputs.forEach((name, k) => rows.push({label: name, sub: 'sense', get: s => s.inputs[k]}));
    X.regions.forEach((name, r) => steps[0].cells[r].forEach((_, j, a) => rows.push({label: name + (a.length > 1 ? ' ' + (j + 1) : ''), sub: '', get: s => s.cells[r][j]})));
    let cur = Math.max(0, steps.findIndex(s => s.tick === X.event_tick) - 2), timer = null;
    const anyNeg = () => rows.some(r => steps.some(s => r.get(s) < 0));
    root.innerHTML = `<div class="lb-strip"><div class="lb-stripworld"><canvas width="${size * CELL}" height="${size * CELL}"></canvas></div>`
      + `<div class="lb-stripside"><div class="lb-controls"><button data-a="play">Play</button><button data-a="prev" aria-label="previous tick">‹</button><button data-a="next" aria-label="next tick">›</button><input type="range" min="0" max="${n - 1}" value="${cur}" aria-label="tick"><span class="lb-tickno"></span></div>`
      + `<div class="lb-stripcaption"></div><div class="lb-striplegend"></div></div></div><div class="lb-heat"></div>`
      + `<div class="lb-caption">One animal of the recorded last generation (run ${esc(X.run)}, agent ${X.agent}). Top: the world around it before it acts, the animal in the middle. Below: one column per tick, one row per sense or neuron, darker = more active${anyNeg() ? ' (orange: below zero, less than the sense is used to)' : ''}. The action row shows ↑ a step, ↶ ↷ a turn, ● a good berry eaten, ✕ a poison berry eaten; the last row is the stomach. Click a column or drag the slider.</div>`;
    const cv = root.querySelector('canvas'), ctx = cv.getContext('2d'), heat = root.querySelector('.lb-heat');
    const col = o => { const c = X.objects[o].colour; return `rgb(${c.map(v => Math.round(v * 255)).join(',')})`; };
    const css = name => getComputedStyle(root).getPropertyValue(name).trim();
    // the heat strip is an SVG built once; the cursor moves
    const CW = Math.max(14, Math.min(24, Math.floor(600 / n))), RH = 22, L = 104, HW = L + n * CW + 4, extra = 2;
    const HH = (rows.length + extra) * RH + 22;
    let h = `<svg viewBox="0 0 ${HW} ${HH}" class="lb-svg" role="img" aria-label="activity per tick">`;
    rows.forEach((r, i) => {
      h += `<text class="lb-rowlabel" x="${L - 8}" y="${i * RH + 14}" text-anchor="end">${esc(r.label)}</text>`;
      steps.forEach((s, t) => { const raw = r.get(s), v = Math.max(0, Math.min(1, Math.abs(raw))); h += `<rect class="lb-cell" data-t="${t}" data-i="${i}" x="${L + t * CW}" y="${i * RH}" width="${CW - 2}" height="${RH - 2}" rx="2" style="fill:color-mix(in srgb, var(${raw < 0 ? '--lb-s2' : '--lb-seq'}) ${Math.round(v * 100)}%, var(--lb-cell0))"/>`; });
    });
    const ya = rows.length * RH, yf = ya + RH;
    h += `<text class="lb-rowlabel" x="${L - 8}" y="${ya + 14}" text-anchor="end">action</text><text class="lb-rowlabel" x="${L - 8}" y="${yf + 14}" text-anchor="end">stomach</text>`;
    steps.forEach((s, t) => {
      const a = X.actions[s.action], ate = s.ate ? X.objects[s.ate] : null, f = Math.max(0, Math.min(1, s.food / X.max_food));
      h += `<text class="lb-glyph${ate ? (ate.effect === 'poison' ? ' lb-bad' : ' lb-good') : ''}" data-t="${t}" x="${L + t * CW + (CW - 2) / 2}" y="${ya + 15}" text-anchor="middle">${ate ? (ate.effect === 'poison' ? '✕' : '●') : GLYPH[a] || '?'}</text>`
        + `<rect class="lb-track" x="${L + t * CW}" y="${yf + 2}" width="${CW - 2}" height="${RH - 6}" rx="2"/><rect class="lb-food" x="${L + t * CW}" y="${yf + 2 + (RH - 6) * (1 - f)}" width="${CW - 2}" height="${(RH - 6) * f}" rx="2"/>`;
    });
    h += `<rect class="lb-cursor" x="0" y="-2" width="${CW + 2}" height="${(rows.length + extra) * RH + 2}" rx="3"/>`
      + steps.map((s, t) => `<rect class="lb-hit" data-t="${t}" x="${L + t * CW - 1}" y="0" width="${CW}" height="${(rows.length + extra) * RH}"/>`).join('')
      + `<text class="lb-tick" x="${L}" y="${HH - 4}">tick ${steps[0].tick}</text><text class="lb-tick" x="${L + n * CW - 2}" y="${HH - 4}" text-anchor="end">tick ${steps[n - 1].tick}</text></svg>`;
    heat.innerHTML = h;
    const cursor = heat.querySelector('.lb-cursor');
    function drawWorld(s) {
      ctx.fillStyle = css('--lb-ground') || '#eef0e6'; ctx.fillRect(0, 0, cv.width, cv.height);
      ctx.strokeStyle = css('--lb-gridline') || '#e1e0d9'; ctx.lineWidth = 1;
      for (let i = 1; i < size; i++) { ctx.beginPath(); ctx.moveTo(i * CELL + 0.5, 0); ctx.lineTo(i * CELL + 0.5, cv.height); ctx.moveTo(0, i * CELL + 0.5); ctx.lineTo(cv.width, i * CELL + 0.5); ctx.stroke(); }
      for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
        const o = s.patch[y * size + x], cx = x * CELL + CELL / 2, cy = y * CELL + CELL / 2;
        if (o < 0) { ctx.fillStyle = css('--lb-wall') || '#c3c2b7'; ctx.fillRect(x * CELL, y * CELL, CELL, CELL); continue; }
        if (!o) continue;
        const ob = X.objects[o];
        ctx.beginPath(); ctx.arc(cx, cy, CELL * 0.34, 0, 7);
        if (ob.kind === 'empty bush') { ctx.strokeStyle = col(o); ctx.lineWidth = 2.5; ctx.stroke(); }
        else { ctx.fillStyle = col(o); ctx.fill(); ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 1; ctx.stroke(); }
        if (ob.effect === 'poison') { ctx.strokeStyle = ob.name.startsWith('White') ? '#333' : '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(cx - 4, cy - 4); ctx.lineTo(cx + 4, cy + 4); ctx.moveTo(cx + 4, cy - 4); ctx.lineTo(cx - 4, cy + 4); ctx.stroke(); }
      }
      const tri = (cx, cy, d, fill, r) => { ctx.save(); ctx.translate(cx, cy); ctx.rotate(d * Math.PI / 2); ctx.beginPath(); ctx.moveTo(0, -r); ctx.lineTo(r * 0.8, r * 0.8); ctx.lineTo(-r * 0.8, r * 0.8); ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke(); ctx.restore(); };
      for (const [dy, dx, d] of s.others) tri((dx + R) * CELL + CELL / 2, (dy + R) * CELL + CELL / 2, d, css('--lb-other') || '#898781', CELL * 0.3);
      tri(R * CELL + CELL / 2, R * CELL + CELL / 2, s.dir, css('--lb-animal') || '#0b0b0b', CELL * 0.36);
    }
    function show(i) {
      cur = Math.max(0, Math.min(n - 1, i));
      const s = steps[cur], prev = steps[cur - 1], a = X.actions[s.action], ate = s.ate ? X.objects[s.ate] : null;
      drawWorld(s); cursor.setAttribute('x', L + cur * CW - 2);
      root.querySelector('input').value = cur; root.querySelector('.lb-tickno').textContent = `tick ${s.tick}`;
      let c = `The animal <b>${WORDS[a] || a}</b>`;
      if (ate) c += ` and eats a ${esc(ate.name)}: ${ate.effect === 'poison' ? '<b>poison</b>' : 'good food'}`;
      if (prev) c += `. Stomach ${fmt(prev.food, 1)} → ${fmt(s.food, 1)}`; else c += `. Stomach ${fmt(s.food, 1)}`;
      if (s.pain > 0.005) c += `, pain ${fmt(s.pain, 2)}`;
      root.querySelector('.lb-stripcaption').innerHTML = c + '.';
    }
    const seen = new Set(steps.flatMap(s => s.patch.filter(o => o > 0)));
    root.querySelector('.lb-striplegend').innerHTML = [...seen].sort((a, b) => a - b).map(o => { const ob = X.objects[o]; return `<span><i class="lb-objkey${ob.kind === 'empty bush' ? ' lb-hollow' : ''}" style="--c:${col(o)}"></i>${esc(ob.name)}${ob.effect === 'poison' ? ' (poison, ✕)' : ''}</span>`; }).join('')
      + '<span><i class="lb-objkey lb-tri"></i>the animal (points where it faces); grey = others</span>';
    heat.addEventListener('click', e => { const t = e.target.dataset.t; if (t !== undefined) show(+t); });
    hover(heat, e => {
      const t = e.target.dataset.t; if (t === undefined) return hideTip();
      const s = steps[+t]; showTip(e, `<b>tick ${s.tick}</b>: ${esc(X.actions[s.action])}` + rows.map(r => `<div>${esc(r.label)} <b>${fmt(r.get(s), 2)}</b></div>`).join(''));
    });
    root.querySelector('input').addEventListener('input', e => show(+e.target.value));
    root.querySelector('.lb-controls').addEventListener('click', e => {
      const a = e.target.dataset.a; if (!a) return;
      if (a === 'prev') show(cur - 1); else if (a === 'next') show(cur + 1);
      else if (timer) { clearInterval(timer); timer = null; e.target.textContent = 'Play'; }
      else { if (cur >= n - 1) show(0); e.target.textContent = 'Pause'; timer = setInterval(() => { if (cur >= n - 1) { clearInterval(timer); timer = null; e.target.textContent = 'Play'; } else show(cur + 1); }, 700); }
    });
    show(cur);
    // the world tile is a canvas painted with the theme's colours: paint it again when the theme is switched
    new MutationObserver(() => show(cur)).observe(document.body, {attributes: true, attributeFilter: ['class']});
  }

  // ------------------------------------------------------------------ a small table from a stage's extras
  function table(root, D) {
    const X = (D.extras || {})[root.dataset.example];
    if (!X) { root.innerHTML = '<div class="lb-caption">Not measured yet.</div>'; return; }
    const cell = v => (v && typeof v === 'object') ? pm(v) + seeds(v) : (v == null ? '' : v);
    let h = `<table class="lb-table"><thead><tr>${X.columns.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>`;
    for (const r of X.rows) h += `<tr>${r.map((v, i) => `<td>${i ? cell(v) : v}</td>`).join('')}</tr>`;
    root.innerHTML = h + `</tbody></table>` + (X.caption ? `<div class="lb-caption">${X.caption}</div>` : '');
  }

  const COMPONENTS = {status, summary, curves, lesions, brain, runs, settings, strip, table};
  function init() {
    document.querySelectorAll('[data-life]').forEach(root => {
      const fn = COMPONENTS[root.dataset.life];
      if (!fn || root.dataset.done) return;
      root.dataset.done = '1'; root.classList.add('lb-root');
      load(root.dataset.stage).then(D => fn(root, D)).catch(err => { root.innerHTML = `<div class="lb-note"><p>${esc(err.message)}</p></div>`; });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
