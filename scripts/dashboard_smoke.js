// Smoke test for life/dashboard.html: node scripts/dashboard_smoke.js runs/<exp>/<ts>/dashboard.html
// Runs the page script under a DOM stub and prints "ok" plus draw-call counts, or throws on a runtime error.
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
// the page script is the largest <script> (the first one only follows the book's light / dark choice); its work is
// in main(), which the page starts after the loader has painted: the checks go at the end of main() and main() is
// called directly
const page = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).sort((a, b) => b.length - a.length)[0];
const end = page.lastIndexOf('}');
const checks = `
; setTick(Math.floor(T/2)); setFocus(1);
for (const m of ['w', 'dw', 'w0', 'eta']) { document.getElementById('wmode').value = m; for (const p of Object.keys(DRAW)) DRAW[p](); }   // every panel, each weight mode
if (M.book_layout) { bookLayout(); renderArch(); }
console.log('ok', {T, N, NN, NIN, focus, t, book_layout: M.book_layout ? M.book_layout.stage : null, calls: Object.keys(calls).length, fillRect: calls.fillRect, putImageData: calls.putImageData});
`;
const js = page.slice(0, end) + checks + page.slice(end) + '\nmain();';
const calls = {};
const ctxProxy = new Proxy({}, { get: (_, k) => { if (k === 'createImageData') return (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }); return (...a) => { calls[k] = (calls[k] || 0) + 1; }; }, set: () => true });
const elems = {};
function el(id) { if (!elems[id]) elems[id] = { id, style: { setProperty: () => {}, removeProperty: () => {} }, dataset: {}, value: '0', textContent: '', innerHTML: '', checked: true, width: 100, height: 100, max: 0,
  clientWidth: 100, clientHeight: 100, offsetWidth: 100, offsetHeight: 100,
  getContext: () => ctxProxy, addEventListener: () => {}, setAttribute: () => {}, getAttribute: () => null, querySelectorAll: () => [], querySelector: () => el('_q' + nEl++),
  classList: { toggle: () => {}, add: () => {}, remove: () => {}, contains: () => false }, appendChild: () => {}, append: () => {}, after: () => {}, remove: () => {},
  closest: () => null, scrollIntoView: () => {}, getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }), click: () => {} }; return elems[id]; }
let nEl = 0;
global.document = { getElementById: el, createElement: () => el('_new' + nEl++), createElementNS: () => el('_new' + nEl++), addEventListener: () => {},
  querySelectorAll: () => [], querySelector: () => null, documentElement: el('_html'), body: el('_body') };
global.Image = class { constructor() { this.complete = false; } set src(v) { this._src = v; } };
global.requestAnimationFrame = f => {};
global.ResizeObserver = class { observe() {} disconnect() {} };
global.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
global.getComputedStyle = () => ({ getPropertyValue: () => '#888888' });
global.matchMedia = () => ({ matches: false, addEventListener: () => {} });
global.window = global; global.addEventListener = () => {}; global.innerWidth = 1200; global.innerHeight = 800;
eval(js);
