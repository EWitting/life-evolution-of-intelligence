// Page components of the book, shared by every stage page. Plain JavaScript on the static data files that
// `python -m life.book export <stage>` writes (book/data/<stage>.json); nothing here contains a number.
//
// A page places a component with      <div data-life="curves" data-stage="1.1"></div>
// Components: status, summary, curves, lesions, brain, runs, settings, verdict, strip and table and bars (with
// data-example="<extras name>").
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
    root.innerHTML = s + `<div class="lb-caption">Each region is silenced in turn (no activity, no outgoing synapses) and the final population lives again in ${ev.worlds} shared worlds. Bar = mean over the seeds, dots = the seeds. A region that the creatures use costs fitness when silenced; a bar at 100 means the population does as well without it.</div>`;
    root.querySelectorAll('.lb-row').forEach(g => {
      const r = rows[+g.dataset.i];
      hover(g, e => showTip(e, `<b>${esc(r.name)}</b> silenced<div>${fmt(r.p.mean, 1)}% of intact (± ${fmt(r.p.se, 1)})</div><div class="lb-muted">per seed: ${r.p.seeds.map(v => fmt(v, 0)).join(' · ')}</div>`));
    });
  }

  // ------------------------------------------------------------------ verdict: the two questions a stage answers
  // <div data-life="verdict" data-stage="1.1" data-used="valence_av,no_feed"></div>: per listed lesion (or a table
  // extra with a "% of" row), the fitness without it; then main against control, living apart and sharing a world.
  const LESION_LABEL = {no_plasticity: 'learning', no_depression: 'depression'};
  const stat = a => { const m = mean(a), se = a.length > 1 ? Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1) / a.length) : null; return {mean: m, se, seeds: a}; };
  function verdict(root, D) {
    const ev = D.evaluation || {}, X = D.extras || {};
    // an entry is a lesion name, or "<table extra>:<label>" for a table with a "% of" row
    const used = (root.dataset.used || '').split(',').map(s => s.trim()).filter(Boolean).map(entry => {
      const [name, label] = entry.split(':');
      if (ev.lesions && ev.lesions[name]) return {name: `without <code>${esc(LESION_LABEL[name] || name)}</code>`, p: ev.lesions[name].percent};
      const row = X[name] && X[name].rows.find(r => /% of/.test(r[0]));
      return row ? {name: esc(label || row[0]), p: row[1]} : null;
    }).filter(Boolean);
    const useTag = p => p.seeds.every(v => v <= 95) ? (p.mean < 80 ? ['yes', 'used in every seed'] : ['some', 'used a little, every seed']) : p.mean <= 95 ? ['some', 'used in some seeds'] : ['no', 'not used'];
    const pct = (diff, base) => diff && base ? stat(diff.seeds.map((d, i) => 100 * d / base.seeds[i])) : null;
    const reev = ev.reevaluated, base = reev ? reev.control : D.summary.control && D.summary.control.fit_mean;
    const cmp = [];
    if (reev) cmp.push({name: 'each among its own kind', p: pct(reev.difference, reev.control)});
    if (ev.versus && base) cmp.push({name: 'sharing one world', p: pct(ev.versus, base)});
    if (!reev && D.summary.difference && base) cmp.push({name: 'during evolution', p: pct(D.summary.difference, base)});
    const cmpTag = p => p.se === null ? (p.mean > 0 ? ['yes', 'better'] : ['no', 'worse'])
      : p.mean - 2 * p.se > 0 ? (p.seeds.every(v => v > 0) ? ['yes', 'better in every seed'] : ['yes', 'better on average'])
      : p.mean + 2 * p.se < 0 ? ['bad', 'worse'] : ['no', 'about level'];
    const dots = (vals, x) => vals.map(v => `<i class="lb-vdot" style="left:${x(v).toFixed(1)}%"></i>`).join('');
    const lesionRow = it => {   // 0 .. 120 %, a line at 100
      const x = v => Math.max(0, Math.min(120, v)) / 120 * 100, [cls, tag] = useTag(it.p);
      return `<div class="lb-vitem"><span class="lb-vname">${it.name}</span><span class="lb-vtrack"><i class="lb-vref" style="left:${x(100)}%"></i><i class="lb-vbar" style="width:${x(it.p.mean).toFixed(1)}%"></i>${dots(it.p.seeds, x)}</span>`
        + `<span class="lb-vval">${fmt(it.p.mean, 0)}%</span><span class="lb-vtag lb-v${cls}">${tag}</span></div>`;
    };
    const lim = Math.max(30, ...cmp.flatMap(c => c.p.seeds.map(Math.abs))) * 1.1;
    const cmpRow = it => {   // -lim .. +lim %, the middle is level
      const x = v => 50 + 50 * Math.max(-lim, Math.min(lim, v)) / lim, [cls, tag] = cmpTag(it.p), m = it.p.mean;
      return `<div class="lb-vitem"><span class="lb-vname">${it.name}</span><span class="lb-vtrack"><i class="lb-vref" style="left:50%"></i><i class="lb-vbar${m < 0 ? ' lb-vneg' : ''}" style="left:${Math.min(50, x(m)).toFixed(1)}%;width:${Math.abs(x(m) - 50).toFixed(1)}%"></i>${dots(it.p.seeds, x)}</span>`
        + `<span class="lb-vval">${m > 0 ? '+' : ''}${fmt(m, 0)}%</span><span class="lb-vtag lb-v${cls}">${tag}</span></div>`;
    };
    let h = '<div class="lb-verdict">';
    if (cmp.length) h += `<div class="lb-vq">Does the new brain do better than the old one?</div><div class="lb-vsub">fitness of the new brain minus that of the old, % of the old (the line is level)</div>${cmp.map(cmpRow).join('')}`;
    if (used.length) h += `<div class="lb-vq">Is the new circuit used?</div><div class="lb-vsub">fitness with it silenced, % of the intact creatures (the line is 100%)</div>${used.map(lesionRow).join('')}`;
    root.innerHTML = h + `</div><div class="lb-caption">Bars: mean of the ${D.runs.main.length} seeds; dots: the seeds.${cmp.length ? ' The old brain is the brain of the previous stage, evolved in the same new world from the same start.' : ''}${reev ? ' <i>Each among its own kind</i>: each population lives alone, in the same set of worlds.' : ''}${ev.versus ? ' <i>Sharing one world</i>: half new, half old creatures compete for the same food.' : ''} All measures are under <a href="#full-results">Full results</a>.</div>`;
  }

  // ------------------------------------------------------------------ a table extra as bars: one group per row
  // <div data-life="bars" data-stage="2.2" data-example="xor_lab" data-rows="0,2,3"></div>
  function bars(root, D) {
    const X = (D.extras || {})[root.dataset.example];
    if (!X) { root.innerHTML = '<div class="lb-caption">Not measured yet.</div>'; return; }
    const pick = root.dataset.rows ? root.dataset.rows.split(',').map(Number) : X.rows.map((_, i) => i);
    const rows = pick.map(i => X.rows[i]), cols = X.columns.slice(1), CLS = ['lb-s1', 'lb-s2', 'lb-s3'];
    const unit = root.dataset.unit || '', scale = +(root.dataset.scale || 1);
    const hi = Math.max(...rows.flatMap(r => r.slice(1).flatMap(v => v ? [v.mean, ...v.seeds] : []))) * scale * 1.08 || 1;
    let h = `<div class="lb-legend">${cols.map((c, j) => `<span><i class="lb-key ${CLS[j]}"></i>${esc(root.dataset[`col${j}`] || c)}</span>`).join('')}<span class="lb-muted">dots: the seeds</span></div><div class="lb-barsgrid">`;
    for (const r of rows) {
      h += `<div class="lb-barslabel">${esc(r[0])}</div><div class="lb-barsset">`;
      r.slice(1).forEach((v, j) => {
        if (!v) return;
        const x = u => (u * scale / hi * 100).toFixed(1);
        h += `<div class="lb-barsline"><span class="lb-vtrack"><i class="lb-vbar ${CLS[j]}" style="width:${x(v.mean)}%"></i>${v.seeds.map(s => `<i class="lb-vdot" style="left:${x(s)}%"></i>`).join('')}</span><span class="lb-vval">${fmt(v.mean * scale, scale > 1 ? 0 : undefined)}${unit}</span></div>`;
      });
      h += '</div>';
    }
    root.innerHTML = h + '</div>' + (root.dataset.caption ? `<div class="lb-caption">${root.dataset.caption}</div>` : '');
  }

  // ------------------------------------------------------------------ brain diagram
  // the layout shared by all stages (tools/brain_layout writes data/brain_layout.json); without it, or for a stage it
  // does not have, the diagram is laid out in columns by depth as before
  let layoutP;
  const loadLayout = () => layoutP || (layoutP = fetch('data/brain_layout.json').then(r => r.ok ? r.json() : null).catch(() => null));
  function brain(root, D) { return loadLayout().then(L => drawBrain(root, D, L)); }
  // The diagram is drawn in fixed units and scaled to the page (a brain diagram may use the page's full width,
  // book/wide-brain.lua). Its width is capped so that the region names are never larger on screen than NAME times
  // the text; where it is scaled down further, the text grows inside the boxes (CSS --k, up to what fits). A diagram
  // whose names are still smaller than SMALL times the text gets a button that opens it full screen, with zoom
  // (wheel, pinch, buttons) and pan (drag). show(crop) moves the view to the box crop() returns, or to all of it.
  const NAME = 0.85, SMALL = 0.7, DESIGN = 12.5;   // DESIGN: the names' font size in diagram units (life-book.css)
  function brainView(root, svg, full, BW) {
    const frame = svg.parentNode, wrap = frame.parentNode, btn = frame.querySelector('.lb-expand');
    const textPx = () => parseFloat(getComputedStyle(root.parentNode).fontSize) || 16;
    const setVB = b => svg.setAttribute('viewBox', b.map(v => v.toFixed(1)).join(' '));
    const kOf = scale => Math.min(kfit || 1, Math.max(1, NAME * textPx() / (DESIGN * scale)));
    let vb = full.slice(), kfit = 0, zoom = null, anim = 0, shown = false;
    const V = {bar: null};
    function measure() {   // how much larger the text may be drawn and still fit its box: a region's name and kind, a socket's name
      let k = 1.25, ok = false;
      svg.querySelectorAll('.lb-reg').forEach(g => {
        const inner = g.transform.baseVal.numberOfItems > 0;
        g.querySelectorAll(inner ? 'text' : 'text[text-anchor=middle]').forEach(t => {
          const len = t.getComputedTextLength();
          if (len > 0) { ok = true; k = Math.min(k, (inner ? BW - 14 : 98) / len); }
        });
      });
      return ok ? Math.max(1, k) : 0;
    }
    function size() {
      if (zoom) return;
      const avail = root.clientWidth; if (!avail) return;
      const cap = NAME * textPx() / DESIGN;   // the largest scale: names at NAME times the text
      const w = Math.min(avail, vb[2] * cap), scale = w / vb[2];
      svg.style.width = wrap.style.maxWidth = w + 'px';   // the view buttons and the legend line up with the diagram
      if (!kfit) kfit = measure();
      const k = kOf(scale);
      svg.style.setProperty('--k', k.toFixed(3));
      btn.hidden = DESIGN * k * scale >= SMALL * textPx();
    }
    // the bounding box of some elements of the diagram, with a margin, grown until it cuts none of the elements in
    // whole; all of the diagram when that is most of it anyway
    V.box = (els, whole = []) => {
      const bb = e => {
        const b = e.getBBox(), t = e.transform && e.transform.baseVal.numberOfItems ? e.transform.baseVal.consolidate().matrix : {e: 0, f: 0};
        return b.width || b.height ? [b.x + t.e, b.y + t.f, b.x + b.width + t.e, b.y + b.height + t.f] : null;
      };
      let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
      const add = b => { x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]); x1 = Math.max(x1, b[2]); y1 = Math.max(y1, b[3]); };
      els.map(bb).filter(Boolean).forEach(add);
      if (x0 > x1) return full;
      const P = 18;
      x0 -= P; y0 -= P; x1 += P; y1 += P;
      const rest = whole.map(bb).filter(Boolean);
      for (let grew = true; grew;) {
        grew = false;
        for (const b of rest) if (b[0] < x1 && b[2] > x0 && b[1] < y1 && b[3] > y0 && (b[0] < x0 || b[2] > x1 || b[1] < y0 || b[3] > y1)) { add([b[0] - P, b[1] - P, b[2] + P, b[3] + P]); grew = true; }
      }
      x0 = Math.max(full[0], x0); y0 = Math.max(full[1], y0); x1 = Math.min(full[0] + full[2], x1); y1 = Math.min(full[1] + full[3], y1);
      return (x1 - x0) * (y1 - y0) > 0.8 * full[2] * full[3] ? full : [x0, y0, x1 - x0, y1 - y0];
    };
    V.show = crop => {
      const b = crop ? crop() : full;
      if (zoom) { vb = b; zoom.fit(); return; }
      if (!shown) { shown = true; vb = b; setVB(vb); size(); return; }
      const from = vb.slice(), t0 = performance.now(), id = ++anim;
      const step = now => {
        if (id !== anim || zoom) return;
        const f = Math.min(1, (now - t0) / 350), e = f < 0.5 ? 2 * f * f : 1 - 2 * (1 - f) * (1 - f);
        vb = from.map((v, i) => v + (b[i] - v) * e); setVB(vb); size();
        if (f < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };
    function open() {
      hideTip();
      const ov = el('div', 'lb-zoom lb-root', '<div class="lb-zoombar"><span class="lb-zoomsp"></span>'
        + '<button type="button" data-z="in" title="zoom in">+</button><button type="button" data-z="out" title="zoom out">−</button>'
        + '<button type="button" data-z="fit" title="fit to the screen">fit</button><button type="button" data-z="close" title="close (Esc)">✕</button></div>'
        + '<div class="lb-zoomstage"></div><div class="lb-zoomhint">wheel or pinch to zoom, drag to pan</div>');
      const zbar = ov.firstChild, stage = ov.children[1];
      if (V.bar) zbar.insertBefore(V.bar, zbar.firstChild);
      stage.appendChild(svg); document.body.appendChild(ov);
      document.documentElement.style.overflow = 'hidden'; svg.style.width = '';
      let z = vb;
      const rect = () => stage.getBoundingClientRect();
      const fit = () => {
        const r = rect(), s = Math.min(r.width / vb[2], r.height / vb[3]) * 0.96;
        z = [vb[0] + vb[2] / 2 - r.width / s / 2, vb[1] + vb[3] / 2 - r.height / s / 2, r.width / s, r.height / s]; setVB(z);
        svg.style.setProperty('--k', kOf(s).toFixed(3));
      };
      const at = (cx, cy, f) => {   // zoom by f around a point on the screen
        const r = rect(), ux = z[0] + (cx - r.left) / r.width * z[2], uy = z[1] + (cy - r.top) / r.height * z[3];
        f = z[2] / Math.min(4 * Math.max(full[2], full[3] * r.width / r.height), Math.max(1.5 * BW, z[2] / f));
        z = [ux - (ux - z[0]) / f, uy - (uy - z[1]) / f, z[2] / f, z[3] / f]; setVB(z);
      };
      const pan = (dx, dy) => { const r = rect(); z = [z[0] - dx / r.width * z[2], z[1] - dy / r.height * z[3], z[2], z[3]]; setVB(z); };
      const pts = new Map();
      const two = () => { const [a, b] = [...pts.values()]; return {d: Math.hypot(a.x - b.x, a.y - b.y), x: (a.x + b.x) / 2, y: (a.y + b.y) / 2}; };
      stage.addEventListener('pointerdown', e => { pts.set(e.pointerId, {x: e.clientX, y: e.clientY}); stage.setPointerCapture(e.pointerId); stage.classList.add('lb-grab'); });
      stage.addEventListener('pointermove', e => {
        const p = pts.get(e.pointerId); if (!p) return;
        if (pts.size === 1) { pan(e.clientX - p.x, e.clientY - p.y); p.x = e.clientX; p.y = e.clientY; return; }
        const a = two(); p.x = e.clientX; p.y = e.clientY; const b = two();
        pan(b.x - a.x, b.y - a.y); if (a.d > 0) at(b.x, b.y, b.d / a.d);
      });
      const up = e => { pts.delete(e.pointerId); if (!pts.size) stage.classList.remove('lb-grab'); };
      stage.addEventListener('pointerup', up); stage.addEventListener('pointercancel', up);
      stage.addEventListener('wheel', e => { e.preventDefault(); at(e.clientX, e.clientY, Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.002))); }, {passive: false});
      const key = e => { if (e.key === 'Escape') close(); };
      function close() {
        hideTip();
        frame.insertBefore(svg, btn); if (V.bar) wrap.insertBefore(V.bar, frame);
        ov.remove(); document.documentElement.style.overflow = '';
        document.removeEventListener('keydown', key); window.removeEventListener('resize', fit);
        zoom = null; setVB(vb); size();
      }
      zbar.addEventListener('click', e => {
        const a = e.target.dataset.z, r = rect();
        if (a === 'in' || a === 'out') at(r.left + r.width / 2, r.top + r.height / 2, a === 'in' ? 1.4 : 1 / 1.4);
        else if (a === 'fit') fit(); else if (a === 'close') close();
      });
      document.addEventListener('keydown', key); window.addEventListener('resize', fit);
      zoom = {fit}; fit();
    }
    btn.addEventListener('click', open);
    new ResizeObserver(() => size()).observe(root);
    setVB(vb); size();
    return V;
  }
  function drawBrain(root, D, L) {
    const REG = D.brain.regions, PROJ = D.brain.projections, MODL = D.brain.modulators || [], BW = 122, BH = 38, GAPW = 80, pos = {};   // GAPW: the space between two columns (tools/brain_layout uses the same)
    const A = L && L.stages && L.stages[D.stage.key];
    const atlas = !!A && REG.every(r => A.pos[r.name]);
    // cells and group boxes take the colour of their top-level group: the layout file's hues (the same in every
    // chapter), else one from the group's name
    const hues = (L && L.hues) || {};
    const hueOf = g => hues[g] !== undefined ? hues[g] : [...g].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 0);
    const blk = {}, sock = {};   // with the layout: the senses / motor blocks' extent [y0, y1] and their sockets, in px
    let AH, SVGW;
    if (atlas) {
      const CW = BW + GAPW, STEP = BH + 26, rows = [...Object.values(A.pos).map(p => p[1]), ...Object.values(A.blocks).flat()];
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
    AH = Math.max(160, Math.max(...colH) + 70); SVGW = Math.max(720, 40 + colKeys.length * (BW + GAPW) - GAPW);
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
      s += `<rect class="lb-group${gp.includes('/') ? '' : ' lb-gtop'}" data-g="${esc(gp)}" style="--h:${hueOf(gp)}" x="${f1(x0)}" y="${f1(y0)}" width="${f1(x1 - x0)}" height="${f1(y1 - y0)}" rx="9"/>`;
      // the name in the corner, and in the middle for when the group is drawn as an empty outline (view "new circuit")
      labels += `<text class="lb-grouplabel" data-g="${esc(gp)}" x="${f1(x0 + 7)}" y="${f1(y0 + 12)}">${esc(gp.split('/').pop())}</text>`
        + `<text class="lb-gcentre" data-g="${esc(gp)}" x="${f1((x0 + x1) / 2)}" y="${f1((y0 + y1) / 2 + 5)}" text-anchor="middle">${esc(gp.split('/').pop())}</text>`;
    });
    // modulators: a dashed line from the cell that releases one to every cell with a receptor for it, and to every
    // projection whose learning it gates (the line ends on the middle of that projection)
    const mid = {}, MODS = [];
    PROJ.forEach((p, i) => { const m = /^M([-\d.]+),([-\d.]+) Q([-\d.]+),([-\d.]+) ([-\d.]+),([-\d.]+)$/.exec(geom(p, ...ENDS[i][0]).d); if (m) mid[i] = [(+m[1] + 2 * +m[3] + +m[5]) / 4, (+m[2] + 2 * +m[4] + +m[6]) / 4]; });
    for (const m of MODL) {
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
          s += `<rect class="lb-sock" data-s="${esc(n)}" x="${f1(sx)}" y="${f1(y - SH / 2)}" width="${SW}" height="${SH}" rx="4"/><text class="lb-regsub" x="${f1(sx + SW / 2)}" y="${f1(y + 3.5)}" text-anchor="middle">${esc(n.slice(r.name.length + 1))}</text>`;
        }
        s += '</g>';
        continue;
      }
      const [x, y] = pos[r.name];
      s += `<g class="lb-reg${r.new ? ' lb-isnew' : ''}" data-r="${esc(r.name)}" transform="translate(${f1(x - BW / 2)},${f1(y - BH / 2)})"><rect width="${BW}" height="${BH}" rx="6" ${r.group ? `class="lb-cell" style="--h:${hueOf(r.group.split('/')[0])}"` : 'class="lb-io"'}/>`
        + `<text class="lb-regname" x="8" y="16">${esc(r.name)}</text><text class="lb-regsub" x="8" y="30">${r.name === 'in' ? 'senses' : r.name === 'out' ? 'motor neurons' : (r.sign > 0 ? 'excitatory' : r.sign < 0 ? 'inhibitory' : 'exc. and inh.')} · ${r.size}</text></g>`;
    }
    s += '</svg>';
    const anyNew = REG.some(r => r.new);
    root.innerHTML = `<div class="lb-brainwrap"><div class="lb-brainframe">${s}<button type="button" class="lb-expand" title="open full screen, to zoom and pan" aria-label="open full screen">⤢</button></div>` + `<div class="lb-legend lb-brainlegend"><span><i class="lb-key lb-hardkey"></i>hard-wired (designed; evolution tunes only the strength)</span><span><i class="lb-key lb-evokey"></i>evolved</span>${PROJ.some(p => p.rule !== 'fixed') ? '<span><i class="lb-key lb-learnkey"></i>learned within a life</span>' : ''}<span>▸ excites</span><span>⊣ inhibits</span><span>● senses, mixed signs</span>${MODS.length ? '<span><i class="lb-key lb-modkey"></i>modulator: released by one cell, acts on a cell or on the learning of a projection</span>' : ''}${anyNew ? '<span><i class="lb-key lb-newkey"></i>added in this stage</span>' : ''}<span class="lb-muted">line width = total synaptic strength arriving at one target neuron, mean of the final population</span></div></div>`;
    const svg = root.querySelector('.lb-brain'), view = brainView(root, svg, [0, 0, SVGW, H], BW);
    // three views. "all cells": only what this stage added and what it connects to directly: new projections,
    // projections into or out of a new region, and the modulator lines of a new modulator or of a new region; a
    // modulator line onto a projection brings that projection along. Regions stay in place; the ones none of it
    // touches are faded. "new circuit": the same, but a group none of it touches is drawn as its empty outline with
    // its name in the middle (the outermost such group; its cells and inner groups are hidden). "everything": all.
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
      const shut = [...paths].filter(gp => { const mem = REG.filter(r => r.group === gp || r.group.startsWith(gp + '/')); return mem.length > 1 && !mem.some(r => used.has(r.name)); });   // a single cell is shown as itself
      const inShut = gp => shut.some(o => gp.startsWith(o + '/'));
      root.querySelectorAll('[data-g]').forEach(g => { g.classList.toggle('lb-shut', shut.includes(g.dataset.g) && !inShut(g.dataset.g)); g.classList.toggle('lb-inshut', inShut(g.dataset.g)); });
      root.querySelectorAll('.lb-reg').forEach(g => { const r = REG.find(o => o.name === g.dataset.r); g.classList.toggle('lb-inshut', !!r.group && (shut.includes(r.group) || inShut(r.group))); });
      const bar = el('div', 'lb-focusbar', '<span class="lb-muted">Show</span>'
        + '<button type="button" data-f="2" title="what this stage added and what it connects to directly; other groups as outlines">new circuit</button>'
        + '<button type="button" data-f="1" title="what this stage added and what it connects to directly, among all cells">all cells</button>'
        + '<button type="button" data-f="0" title="every region and connection of this brain">everything</button>');
      const set = f => { svg.classList.toggle('lb-focus', f > 0); svg.classList.toggle('lb-outline', f > 1); bar.querySelectorAll('button').forEach(b => b.classList.toggle('lb-on', +b.dataset.f === f)); view.show(f > 1 ? crop : null); };
      // the view "new circuit" is cropped to what it shows: the regions, lines and the senses / motor sockets it uses,
      // and every other region or group name the box would cut (group outlines may be cut)
      const usedS = new Set(); for (const i of showP) for (const [a, b] of ENDS[i]) usedS.add(a).add(b);
      const crop = () => view.box([...svg.querySelectorAll('.lb-edge:not(.lb-ctx), .lb-mod:not(.lb-ctx)'), ...[...svg.querySelectorAll('.lb-reg:not(.lb-ctx):not(.lb-inshut)')].filter(g => !isBlock(g.dataset.r)), ...[...svg.querySelectorAll('.lb-sock')].filter(r => usedS.has(r.dataset.s))],
        [...[...svg.querySelectorAll('.lb-reg:not(.lb-inshut)')].filter(g => !isBlock(g.dataset.r)), ...svg.querySelectorAll('.lb-grouplabel, .lb-gcentre')]);
      bar.addEventListener('click', e => { if (e.target.dataset.f) set(+e.target.dataset.f); });
      svg.parentNode.parentNode.insertBefore(bar, svg.parentNode); view.bar = bar; set(2);
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
      const rel = MODL.filter(m => m.pos === r.name).map(m => m.name), rec = (r.receptors || []).map(rc => rc[0]);
      hover(g, e => showTip(e, `<b>${esc(r.name)}</b> ${r.size} neuron${r.size === 1 ? '' : 's'}${r.new ? ', added in this stage' : ''}<div class="lb-muted">from: ${esc([...new Set(ins.map(p => p.src))].join(', ') || '–')}</div><div class="lb-muted">to: ${esc([...new Set(outs.map(p => p.dst))].join(', ') || '–')}</div>${rel.length ? `<div class="lb-muted">releases: ${esc(rel.join(', '))}</div>` : ''}${rec.length ? `<div class="lb-muted">receptors for: ${esc(rec.join(', '))}</div>` : ''}`));
    });
  }

  // ------------------------------------------------------------------ runs and settings
  function runs(root, D) {
    const first = g => (D.runs[g] || []).find(r => r.dashboard);
    const btn = (g, label) => first(g) ? `<a class="lb-watch" href="${esc(first(g).dashboard)}" target="_blank">&#9654; ${label}</a>` : '';
    const watch = btn('main', 'Watch a main run') + btn('control', 'Watch a control run');
    let h = (watch ? `<div class="lb-watchrow">${watch}</div>` : '<div class="lb-caption">No run of this stage is published yet.</div>')
      + '<div class="lb-caption">A dashboard replays the last generation of a run: all creatures in their world, and the brain of a few of them, tick by tick. It is a large page and opens in a new tab.</div>'
      + '<details class="lb-details"><summary>Run directories and dates</summary><table class="lb-table"><thead><tr><th></th><th>seed</th><th>run directory</th><th>generations</th><th>finished</th><th>dashboard</th></tr></thead><tbody>';
    for (const g of ['main', 'control']) for (const r of D.runs[g]) h += `<tr><td>${g}</td><td>${r.seed}</td><td><code>runs/${esc(r.dir)}</code></td><td>${r.generations}</td><td>${esc(r.finished.replace('T', ' '))}</td><td>${r.dashboard ? `<a href="${esc(r.dashboard)}">open</a>` : '<span class="lb-muted">not published</span>'}</td></tr>`;
    root.innerHTML = h + `</tbody></table><div class="lb-caption">One dashboard per stage is published (seed 0 of main and of control); the other seeds are in the run directories. Data exported ${esc(D.exported.replace('T', ' '))}.</div></details>`;
  }
  function settings(root, D) {
    const S = D.settings, w = S.world, e = S.evolution;
    const rows = [['World', `${w.height} × ${w.width} cells, ${w.num_agents} creatures`], ['Objects', esc(S.objects.filter(n => !/Empty/.test(n) && /Bush|Onion|Spring/.test(n)).filter((n, i, a) => a.indexOf(n) === i).join(', '))],
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
  // the OHOL sprites of the objects in the strips (python -m life.book sprites), as in the dashboards
  let spritesP;
  const loadSprites = () => spritesP || (spritesP = fetch('data/sprites.json').then(r => r.ok ? r.json() : null).catch(() => null).then(S => {
    if (!S) return null;
    const img = u => new Promise(res => { const i = new Image(); i.onload = () => res(i); i.onerror = () => res(null); i.src = u; });
    const keys = Object.keys(S.sprites), pk = Object.keys(S.people || {});
    return Promise.all([...keys.map(k => img(S.sprites[k])), S.ground ? img(S.ground) : null, ...pk.map(k => img(S.people[k]))])
      .then(a => ({sprites: Object.fromEntries(keys.map((k, i) => [k, a[i]])), ground: a[keys.length],
        people: Object.fromEntries(pk.map((k, i) => [k, a[keys.length + 1 + i]])), cycle: S.people_cycle || 64}));
  }));
  // a red disc with a white skull, centre (x, y), diameter d (as in life/dashboard.html)
  function skull(ctx, x, y, d) {
    const r = d / 2, k = r * 0.62;
    ctx.save(); ctx.translate(x, y);
    ctx.fillStyle = '#d0242b'; ctx.beginPath(); ctx.arc(0, 0, r, 0, 7); ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = d / 14; ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(0, -k * 0.18, k * 0.78, Math.PI, 0); ctx.lineTo(k * 0.78, k * 0.25); ctx.lineTo(k * 0.45, k * 0.4); ctx.lineTo(k * 0.45, k * 0.8);
    ctx.lineTo(-k * 0.45, k * 0.8); ctx.lineTo(-k * 0.45, k * 0.4); ctx.lineTo(-k * 0.78, k * 0.25); ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#d0242b'; ctx.beginPath(); ctx.arc(-k * 0.33, -k * 0.08, k * 0.24, 0, 7); ctx.arc(k * 0.33, -k * 0.08, k * 0.24, 0, 7); ctx.fill();
    ctx.restore();
  }
  function strip(root, D) {
    const X = D.extras && D.extras[root.dataset.example];
    if (!X) { root.innerHTML = '<div class="lb-note"><p>No example exported for this stage yet.</p></div>'; return; }
    const steps = X.steps, n = steps.length, R = X.radius, size = 2 * R + 1, CELL = 30;
    const rows = [];   // one row per input or neuron
    X.inputs.forEach((name, k) => rows.push({label: name, sub: 'sense', get: s => s.inputs[k]}));
    (X.mods || []).forEach((name, k) => rows.push({label: name, sub: 'modulator', get: s => s.mods[k]}));
    X.regions.forEach((name, r) => steps[0].cells[r].forEach((_, j, a) => rows.push({label: name + (a.length > 1 ? ' ' + (j + 1) : ''), sub: '', get: s => s.cells[r][j]})));
    let cur = Math.max(0, steps.findIndex(s => s.tick === X.event_tick) - 2), timer = null;
    const anyNeg = () => rows.some(r => steps.some(s => r.get(s) < 0));
    const SC = 2;   // the canvas is painted at twice its size, so that sprites stay sharp on a sharp screen
    root.innerHTML = `<div class="lb-strip"><div class="lb-stripworld"><canvas width="${size * CELL * SC}" height="${size * CELL * SC}"></canvas></div>`
      + `<div class="lb-stripside"><div class="lb-controls"><button data-a="play">Play</button><button data-a="prev" aria-label="previous tick">‹</button><button data-a="next" aria-label="next tick">›</button><input type="range" min="0" max="${n - 1}" value="${cur}" aria-label="tick"><span class="lb-tickno"></span></div>`
      + `<div class="lb-stripcaption"></div><details class="lb-details lb-striplegendbox"><summary>What is what</summary><div class="lb-striplegend"></div></details></div></div><div class="lb-heat"></div>`
      + `<div class="lb-caption">One creature of the recorded last generation (run ${esc(X.run)}, agent ${X.agent}). Top: the world around it before it acts, the creature in the middle; point at a cell for its name. ☠ marks a poison kind of bush, a mark that the creature cannot see. Below: one column per tick, one row per sense or neuron, darker = more active${anyNeg() ? ' (orange: below zero, less than the sense is used to)' : ''}. The action row shows ↑ a step, ↶ ↷ a turn, ● a good berry eaten, ✕ a poison berry eaten; the last row is the stomach. Click a column or drag the slider.</div>`;
    const cv = root.querySelector('canvas'), ctx = cv.getContext('2d'), heat = root.querySelector('.lb-heat');
    const col = o => { const c = X.objects[o].colour; return `rgb(${c.map(v => Math.round(v * 255)).join(',')})`; };
    // a bush is of a poison kind when its berry is (export), an empty bush when its full bush is
    const byName = Object.fromEntries(X.objects.map((ob, i) => [ob.name, i]));
    const poison = o => {
      const ob = X.objects[o]; if (ob.effect === 'poison') return true;
      const f = ob.kind === 'empty bush' ? byName[ob.name.replace('Empty ', '')] : undefined;
      return f !== undefined && X.objects[f].effect === 'poison';
    };
    let SP = null;   // the sprites, once loaded
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
      const span = size * CELL;
      ctx.setTransform(SC, 0, 0, SC, 0, 0);
      const pat = SP && SP.ground ? ctx.createPattern(SP.ground, 'repeat') : null;   // OHOL tiles its ground over 4 x 4 cells
      if (pat && pat.setTransform) { const k = 4 * CELL / SP.ground.naturalWidth; pat.setTransform(new DOMMatrix([k, 0, 0, k, 0, 0])); }
      ctx.fillStyle = pat || css('--lb-ground') || '#eef0e6'; ctx.fillRect(0, 0, span, span);
      if (!pat) {
        ctx.strokeStyle = css('--lb-gridline') || '#e1e0d9'; ctx.lineWidth = 1;
        for (let i = 1; i < size; i++) { ctx.beginPath(); ctx.moveTo(i * CELL + 0.5, 0); ctx.lineTo(i * CELL + 0.5, span); ctx.moveTo(0, i * CELL + 0.5); ctx.lineTo(span, i * CELL + 0.5); ctx.stroke(); }
      }
      for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
        const o = s.patch[y * size + x], cx = x * CELL + CELL / 2, cy = y * CELL + CELL / 2;
        if (o < 0) { ctx.fillStyle = css('--lb-wall') || '#c3c2b7'; ctx.fillRect(x * CELL, y * CELL, CELL, CELL); continue; }
        if (!o) continue;
        const ob = X.objects[o], im = SP && SP.sprites[String(ob.ohol)];
        if (im) {   // drawn as in the dashboard: standing on the bottom of its cell, with a shadow
          const r = Math.min(0.95 * CELL / im.naturalWidth, 0.95 * CELL / im.naturalHeight), dw = im.naturalWidth * r, dh = im.naturalHeight * r;
          ctx.fillStyle = 'rgba(0,0,0,0.18)'; ctx.beginPath(); ctx.ellipse(cx, y * CELL + CELL * 0.92, dw * 0.42, CELL * 0.08, 0, 0, 7); ctx.fill();
          ctx.drawImage(im, x * CELL + (CELL - dw) / 2, y * CELL + CELL * 0.97 - dh, dw, dh);
        } else {
          ctx.beginPath(); ctx.arc(cx, cy, CELL * 0.34, 0, 7);
          if (ob.kind === 'empty bush') { ctx.strokeStyle = col(o); ctx.lineWidth = 2.5; ctx.stroke(); }
          else { ctx.fillStyle = col(o); ctx.fill(); ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 1; ctx.stroke(); }
        }
        if (poison(o)) skull(ctx, x * CELL + CELL * 0.75, y * CELL + CELL * 0.25, CELL * 0.5);
      }
      // the animals as in the dashboard: a dressed OHOL body (facing left or right) and a chevron at the cell edge
      // that points where it faces; red for the animal followed, blue for the others; triangles without sprites
      const tri = (cx, cy, d, fill, r) => { ctx.save(); ctx.translate(cx, cy); ctx.rotate(d * Math.PI / 2); ctx.beginPath(); ctx.moveTo(0, -r); ctx.lineTo(r * 0.8, r * 0.8); ctx.lineTo(-r * 0.8, r * 0.8); ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke(); ctx.restore(); };
      const animal = (gx, gy, d, body, left, col, focus) => {
        const cx = gx * CELL + CELL / 2, cy = gy * CELL + CELL / 2;
        if (!body) return tri(cx, cy, d, focus ? '#0b0b0b' : css('--lb-other') || '#898781', CELL * (focus ? 0.36 : 0.3));
        const bh = 1.05 * CELL, bw = body.naturalWidth * bh / body.naturalHeight;
        ctx.save(); ctx.translate(cx, cy);
        ctx.fillStyle = 'rgba(0,0,0,0.18)'; ctx.beginPath(); ctx.ellipse(0, 0.42 * CELL, bw * 0.5, 0.08 * CELL, 0, 0, 7); ctx.fill();
        ctx.save(); if (left) ctx.scale(-1, 1); ctx.drawImage(body, -bw / 2, 0.5 * CELL - bh - 0.03 * CELL, bw, bh); ctx.restore();
        const a = 0.2 * CELL;
        ctx.rotate(d * Math.PI / 2); ctx.beginPath(); ctx.moveTo(0, -0.5 * CELL - a * 0.4); ctx.lineTo(a, -0.5 * CELL + a * 0.6); ctx.lineTo(-a, -0.5 * CELL + a * 0.6); ctx.closePath();
        ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 1; ctx.stroke();
        ctx.restore();
        if (focus) { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, 0.62 * CELL, 0, 7); ctx.stroke(); }
      };
      const P = SP && SP.people, body = k => P && P[String(k)];
      s.others.forEach(([dy, dx, d], k) => animal(dx + R, dy + R, d, body(k % 4), d === 3, '#1b3a6b', false));
      animal(R, R, s.dir, body(X.agent % ((SP && SP.cycle) || 64)), facingLeft(steps.indexOf(s)), '#c8102e', true);
    }
    // OHOL bodies face only left or right: keep the last sideways facing while the animal moves up or down
    function facingLeft(i) {
      for (let k = i; k >= 0; k--) { const d = steps[k].dir; if (d % 2) return d === 3; }
      return false;
    }
    function show(i) {
      cur = Math.max(0, Math.min(n - 1, i));
      const s = steps[cur], prev = steps[cur - 1], a = X.actions[s.action], ate = s.ate ? X.objects[s.ate] : null;
      drawWorld(s); cursor.setAttribute('x', L + cur * CW - 2);
      root.querySelector('input').value = cur; root.querySelector('.lb-tickno').textContent = `tick ${s.tick}`;
      let c = `The creature <b>${WORDS[a] || a}</b>`;
      if (ate) c += ` and eats a ${esc(ate.name)}: ${ate.effect === 'poison' ? '<b>poison</b>' : 'good food'}`;
      if (prev) c += `. Stomach ${fmt(prev.food, 1)} → ${fmt(s.food, 1)}`; else c += `. Stomach ${fmt(s.food, 1)}`;
      if (s.pain > 0.005) c += `, pain ${fmt(s.pain, 2)}`;
      root.querySelector('.lb-stripcaption').innerHTML = c + '.';
    }
    const seen = new Set(steps.flatMap(s => s.patch.filter(o => o > 0)));
    const legend = () => {
      root.querySelector('.lb-striplegend').innerHTML = [...seen].sort((a, b) => a - b).map(o => {
        const ob = X.objects[o], im = SP && SP.sprites[String(ob.ohol)];
        return `<span>${im ? `<img class="lb-objimg" src="${im.src}" alt="">` : `<i class="lb-objkey${ob.kind === 'empty bush' ? ' lb-hollow' : ''}" style="--c:${col(o)}"></i>`}${esc(ob.name)}${poison(o) ? ' <b class="lb-bad">☠ poison</b>' : ''}</span>`;
      }).join('') + '<span><i class="lb-objkey lb-tri"></i>the creature followed (red ring; the chevron points where it faces); others have a blue chevron</span>';
    };
    legend();
    hover(cv, e => {   // the name of what is in a cell
      const b = cv.getBoundingClientRect(), x = Math.floor((e.clientX - b.left) / b.width * size), y = Math.floor((e.clientY - b.top) / b.height * size);
      const s = steps[cur], o = s.patch[y * size + x];
      if (x === R && y === R) return showTip(e, '<b>the creature</b>');
      if (s.others.some(([dy, dx]) => dy + R === y && dx + R === x)) return showTip(e, 'another creature');
      if (!(o > 0)) return hideTip();
      showTip(e, `<b>${esc(X.objects[o].name)}</b>${poison(o) ? '<div>a poison kind</div>' : ''}`);
    });
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
    loadSprites().then(S => { if (S) { SP = S; legend(); show(cur); } });
    // the world tile is a canvas painted with the theme's colours: paint it again when the theme is switched
    new MutationObserver(() => show(cur)).observe(document.body, {attributes: true, attributeFilter: ['class']});
  }

  // ------------------------------------------------------------------ line panels from a stage's extras
  // <div data-life="lines" data-stage="2.3" data-example="reversal_life"></div>
  // extra: {x, xticks?, xlabel, ylabel, vline?, vlabel?, zero?, caption, panels: [{title, series: [{name, y, cls, dash?}]}]}
  // cls: s1 / s2 (main / control colours), good, bad, muted
  function lines(root, D) {
    const X = (D.extras || {})[root.dataset.example];
    if (!X) { root.innerHTML = '<div class="lb-caption">Not measured yet.</div>'; return; }
    const P = X.panels, n = P.length, W = 640, H = 250, m = {l: 46, r: 12, t: 26, b: 40}, gap = 18;
    const pw = (W - m.l - m.r - gap * (n - 1)) / n, ph = H - m.t - m.b;
    const all = P.flatMap(p => p.series.flatMap(s => s.y)).concat(X.zero ? [0] : []);
    let lo = Math.min(...all), hi = Math.max(...all);
    if (lo > 0) lo = 0;
    const pad = (hi - lo) * 0.06 || 1; hi += pad; if (lo < 0) lo -= pad;
    const x0 = X.x[0], x1 = X.x[X.x.length - 1];
    const COL = {s1: 'var(--lb-s1)', s2: 'var(--lb-s2)', good: 'var(--lb-good)', bad: 'var(--lb-bad)', muted: 'var(--lb-muted)'};
    const ticks = (a, b, k) => { const st = Math.pow(10, Math.floor(Math.log10((b - a) / k || 1))), s = [1, 2, 5, 10].map(f => f * st).find(f => (b - a) / f <= k);
                                 const out = []; for (let v = Math.ceil(a / s) * s; v <= b + 1e-9; v += s) out.push(+v.toFixed(6)); return out; };
    let s = `<svg viewBox="0 0 ${W} ${H}" class="lb-svg" role="img" aria-label="${esc(X.ylabel)} over ${esc(X.xlabel)}">`;
    P.forEach((p, i) => {
      const ox = m.l + i * (pw + gap), x = v => ox + (v - x0) / (x1 - x0 || 1) * pw, y = v => m.t + (hi - v) / (hi - lo) * ph;
      for (const v of ticks(lo, hi, 4)) s += `<line class="lb-grid" x1="${ox}" x2="${ox + pw}" y1="${y(v)}" y2="${y(v)}"/>` + (i ? '' : `<text class="lb-tick" x="${ox - 6}" y="${y(v) + 4}" text-anchor="end">${fmt(v)}</text>`);
      if (X.zero && lo < 0) s += `<line class="lb-axis" x1="${ox}" x2="${ox + pw}" y1="${y(0)}" y2="${y(0)}"/>`;
      if (X.xticks) X.x.forEach((v, j) => { s += `<text class="lb-tick" x="${x(v)}" y="${H - m.b + 15}" text-anchor="middle">${esc(X.xticks[j])}</text>`; });
      else for (const v of ticks(x0, x1, 4)) s += `<text class="lb-tick" x="${x(v)}" y="${H - m.b + 15}" text-anchor="middle">${fmt(v)}</text>`;
      if (X.vline != null) s += `<line class="lb-axis" stroke-dasharray="3 3" x1="${x(X.vline)}" x2="${x(X.vline)}" y1="${m.t}" y2="${m.t + ph}"/>`
                              + (X.vlabel ? `<text class="lb-tick" x="${x(X.vline) + 4}" y="${m.t + 10}">${esc(X.vlabel)}</text>` : '');
      if (p.title) s += `<text class="lb-label" x="${ox}" y="${m.t - 9}">${esc(p.title)}</text>`;
      for (const se of p.series)
        s += `<polyline fill="none" stroke="${COL[se.cls] || COL.s1}" stroke-width="${se.cls === 'muted' ? 1.2 : 2}"${se.dash ? ' stroke-dasharray="5 4"' : ''} points="${se.y.map((v, j) => `${x(X.x[j]).toFixed(1)},${y(v).toFixed(1)}`).join(' ')}"><title>${esc(se.name)}</title></polyline>`;
    });
    s += `<text class="lb-tick" x="${m.l + (W - m.l - m.r) / 2}" y="${H - 6}" text-anchor="middle">${esc(X.xlabel)}</text>`
       + `<text class="lb-tick" x="12" y="${m.t + ph / 2}" text-anchor="middle" transform="rotate(-90 12 ${m.t + ph / 2})">${esc(X.ylabel)}</text></svg>`;
    const names = [...new Map(P.flatMap(p => p.series).map(se => [se.name, se])).values()];
    root.innerHTML = `<div class="lb-legend">${names.map(se => `<span><i class="lb-key" style="background:${COL[se.cls] || COL.s1}${se.dash ? ';opacity:.6' : ''}"></i>${esc(se.name)}</span>`).join('')}</div>`
                   + s + (X.caption ? `<div class="lb-caption">${X.caption}</div>` : '');
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

  const COMPONENTS = {status, summary, curves, lesions, brain, runs, settings, strip, table, verdict, bars, lines};
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
