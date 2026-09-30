// Smoke test for life/dashboard.html: node scripts/dashboard_smoke.js runs/<exp>/<ts>/dashboard.html
// Runs the page script under a DOM stub and prints "ok" plus draw-call counts, or throws on a runtime error.
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const js = html.match(/<script>([\s\S]*)<\/script>/)[1];
const calls = {};
const ctxProxy = new Proxy({}, { get: (_, k) => { if (k === 'createImageData') return (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }); return (...a) => { calls[k] = (calls[k] || 0) + 1; }; }, set: () => true });
const elems = {};
function el(id) { if (!elems[id]) elems[id] = { id, style: {}, value: '0', textContent: '', innerHTML: '', checked: true, width: 100, height: 100, max: 0,
  getContext: () => ctxProxy, addEventListener: () => {}, classList: { toggle: () => {} }, appendChild: () => {}, getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }), click: () => {} }; return elems[id]; }
let nEl = 0;
global.document = { getElementById: el, createElement: () => el('_new' + nEl++), addEventListener: () => {} };
global.Image = class { constructor() { this.complete = false; } set src(v) { this._src = v; } };
global.requestAnimationFrame = f => {};
global.window = global; global.addEventListener = () => {};
eval(js + `
; setTick(Math.floor(T/2)); setFocus(1); document.getElementById('wmode').value='dw'; drawW(); document.getElementById('wmode').value='eta'; drawW();
console.log('ok', {T, N, NN, NIN, focus, t, calls: Object.keys(calls).length, fillRect: calls.fillRect, putImageData: calls.putImageData});`);
