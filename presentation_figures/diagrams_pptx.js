// Native, editable PowerPoint versions of the block diagrams (rounded blocks, arrows, plate and ERP icons built from shapes, text as text).
// Slides: 1 forward model, 2 core architecture list, 3 general forward / inverse / invertible, 4 Invertible DeepONet, 5 iFNO family.
// Run: NODE_PATH=<dir with pptxgenjs> node diagrams_pptx.js   (10 x 5.625 in slides, like the thesis deck)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const ERP = JSON.parse(fs.readFileSync(path.join(__dirname, "erp_curve.json"), "utf8"));

const FONT = "Cambria";
const TITLE = "1C3550", SUB = "4A5A6A", FWD = "3D6A94", INV = "E8632B", GREY = "444444";
const C = {
  blue: ["D9E8F7", "8FB4D9"], green: ["E1F1DE", "97C791"], teal: ["D6EEF0", "86C3C8"], peach: ["FDE7D3", "EBA46F"],
  purple: ["EBE2F5", "B39AD1"], yellow: ["FDF3C9", "E0C25A"], icon: ["DBE8F6", "8FB4D9"],
};
const GREEK = { delta: "δ", Psi: "Ψ", psi: "ψ", beta: "β", rightarrow: "→", leftrightarrow: "↔", times: "×", cdot: "·", ldots: "…", top: "T" };

// ---- tiny LaTeX-like markup -> runs (italic letters, sub/superscripts, bold, greek) ------------------------------------
function parseMath(m, base) {
  const runs = []; let i = 0;
  const push = (text, extra = {}) => { if (text) runs.push({ text, options: { ...base, ...extra } }); };
  const group = () => { // {...} or single char
    if (m[i] === "{") { let d = 1, j = i + 1; while (d && j < m.length) { d += m[j] === "{" ? 1 : m[j] === "}" ? -1 : 0; j++; } const s = m.slice(i + 1, j - 1); i = j; return s; }
    if (m[i] === "\\") { let j = i + 1; while (j < m.length && /[A-Za-z]/.test(m[j])) j++; const s = m.slice(i, j); i = j; return s; }
    const s = m[i]; i += 1; return s;
  };
  const inner = (s, extra) => parseMath(s, { ...base, ...extra }).forEach((r) => runs.push(r));
  while (i < m.length) {
    const ch = m[i];
    if (ch === "\\") {
      let j = i + 1; while (j < m.length && /[A-Za-z]/.test(m[j])) j++;
      const cmd = m.slice(i + 1, j);
      if (cmd === "" ) { push(m[j] === "," || m[j] === " " ? " " : m[j] || ""); i = j + 1; continue; }
      i = j; while (m[i] === " ") i++;
      if (cmd === "mathbf") inner(group(), { bold: true });
      else if (cmd === "mathbb") { const s = group(); push(s === "R" ? "ℝ" : s); }
      else if (cmd === "hat") { let s = group(); let bold = false; const mb = /^\\mathbf\{(.)\}$/.exec(s); if (mb) { s = mb[1]; bold = true; } push(s.length === 1 ? s + "̂" : s, bold ? { bold: true } : { italic: true }); }
      else if (GREEK[cmd]) push(GREEK[cmd], cmd === "rightarrow" || cmd === "leftrightarrow" || cmd === "times" || cmd === "cdot" || cmd === "ldots" ? {} : { italic: !/^[A-Z]/.test(cmd) });
      else push(cmd);
      continue;
    }
    if (ch === "_") { i++; inner(group(), { subscript: true }); continue; }
    if (ch === "^") { i++; inner(group(), { superscript: true }); continue; }
    if (ch === "{" || ch === "}") { i++; continue; }
    if (ch === "'") { push("′"); i++; continue; }
    if (/[A-Za-z]/.test(ch)) { push(ch, { italic: true }); i++; continue; }
    push(ch); i++;
  }
  return runs;
}
function rich(str, base) {
  const lines = String(str).split("\n"); const out = [];
  lines.forEach((line, li) => {
    const parts = line.split("$"); const lr = [];
    parts.forEach((p, k) => (k % 2 ? parseMath(p, base).forEach((r) => lr.push(r)) : p && lr.push({ text: p, options: { ...base } })));
    if (!lr.length) lr.push({ text: " ", options: { ...base } });
    if (li < lines.length - 1) lr[lr.length - 1].options = { ...lr[lr.length - 1].options, breakLine: true };
    out.push(...lr);
  });
  return out;
}

// ---- canvas: figure coordinates (y up) -> slide inches ---------------------------------------------------------------
class Canvas {
  constructor(pres, slide, W, H, x0, y0, width, fk, minPt = 7) { Object.assign(this, { pres, slide, W, H, ox: x0, oy: y0, s: width / W, fk, minPt }); }
  X(x) { return this.ox + x * this.s; }
  Y(y) { return this.oy + (this.H - y) * this.s; }
  pt(p) { return Math.max(this.minPt, Math.round(p * this.fk * 2) / 2); }
  text(x, y, str, size, { color = "000000", align = "center", w = 40, h = 6, bold = false } = {}) {
    const wx = w * this.s, hx = h * this.s;
    const left = align === "center" ? this.X(x) - wx / 2 : align === "right" ? this.X(x) - wx : this.X(x);
    this.slide.addText(rich(str, { fontFace: FONT, fontSize: this.pt(size), color, bold }), {
      x: left, y: this.Y(y) - hx / 2, w: wx, h: hx, align, valign: "middle", margin: 0, isTextBox: true, fit: "none",
    });
  }
  block(x0, x1, y0, y1, title, sub, col, { tsize = 17, ssize = 10, dashed = false, accent = false, noText = false, subShift = 0, w3 = false } = {}) {
    const runs = [];
    if (!noText && title) runs.push(...rich(title, { fontFace: FONT, fontSize: this.pt(tsize), color: TITLE }).map((r, i, a) => (i === a.length - 1 && sub ? { ...r, options: { ...r.options, breakLine: true } } : r)));
    if (!noText && sub) runs.push(...rich(sub, { fontFace: FONT, fontSize: this.pt(ssize), color: SUB }));
    this.slide.addText(runs.length ? runs : " ", {
      shape: this.pres.ShapeType.roundRect, rectRadius: 0.08, x: this.X(x0), y: this.Y(y1), w: (x1 - x0) * this.s, h: (y1 - y0) * this.s,
      fill: { color: col[0] }, line: { color: accent ? INV : col[1], width: accent || dashed ? 2.25 : 1.25, dashType: dashed || accent ? "dash" : "solid" },
      align: "center", valign: "middle", margin: 0.03, fit: "none",
    });
  }
  frame(x0, x1, y0, y1, col) {
    this.slide.addShape(this.pres.ShapeType.roundRect, { rectRadius: 0.06, x: this.X(x0), y: this.Y(y1), w: (x1 - x0) * this.s, h: (y1 - y0) * this.s, fill: { type: "none" }, line: { color: col, width: 1.75, dashType: "dash" } });
  }
  line(p0, p1, color, { dash = false, head = false, width = 1.5 } = {}) {
    const [x0, y0] = [this.X(p0[0]), this.Y(p0[1])], [x1, y1] = [this.X(p1[0]), this.Y(p1[1])];
    this.slide.addShape(this.pres.ShapeType.line, {
      x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0), flipH: x1 < x0, flipV: y1 < y0,
      line: { color, width, dashType: dash ? "dash" : "solid", endArrowType: head ? "triangle" : undefined },
    });
  }
  arrow(pts, color = FWD, { dash = false, width = 1.5 } = {}) {
    for (let i = 0; i < pts.length - 1; i++) this.line(pts[i], pts[i + 1], color, { dash, head: i === pts.length - 2, width });
  }
  dot(x, y, d, color = "000000") { const r = d / 2; this.slide.addShape(this.pres.ShapeType.ellipse, { x: this.X(x) - r * this.s, y: this.Y(y) - r * this.s, w: d * this.s, h: d * this.s, fill: { color }, line: { color, width: 0.5 } }); }
  circle(x, y, r, color) { this.slide.addShape(this.pres.ShapeType.ellipse, { x: this.X(x - r), y: this.Y(y + r), w: 2 * r * this.s, h: 2 * r * this.s, fill: { color: "FFFFFF" }, line: { color, width: 1.75 } }); }
  iconPlate(x0, y0, w, h, pts) { // pts: [[fx, fy]] fractions of the plate
    this.block(x0, x0 + w, y0, y0 + h, "", "", C.icon, { noText: true });
    const pw = 0.8 * w, ph = Math.min(pw * 0.357, h * 0.6), px = x0 + (w - pw) / 2, py = y0 + (h - ph) / 2;
    this.slide.addShape(this.pres.ShapeType.rect, { x: this.X(px), y: this.Y(py + ph), w: pw * this.s, h: ph * this.s, fill: { color: "FFFFFF" }, line: { color: "000000", width: 1.75 } });
    pts.forEach(([fx, fy]) => this.dot(px + fx * pw, py + fy * ph, Math.max(0.9, w * 0.085)));
  }
  iconErp(x0, y0, w, h) {
    this.block(x0, x0 + w, y0, y0 + h, "", "", C.icon, { noText: true });
    const a0 = x0 + 0.12 * w, b0 = y0 + 0.14 * h, a1 = x0 + 0.92 * w, b1 = y0 + 0.9 * h;
    this.line([a0, b1], [a0, b0], "000000", { width: 1.25 }); this.line([a0, b0], [a1, b0], "000000", { width: 1.25 });
    const pts = ERP.map((v, i) => ({ x: (0.02 * w + (i / (ERP.length - 1)) * (0.96 * w - 0.12 * w + 0.12 * w - 0.12 * w - 0.04 * w + 0.04 * w)) , v }));
    const X0 = a0 + 0.02 * w, Wd = a1 - a0 - 0.04 * w, Y0 = b0 + 0.04 * h, Hd = b1 - b0 - 0.1 * h;
    const gx = this.X(X0), gw = Wd * this.s, gh = Hd * this.s, gy = this.Y(Y0 + Hd);
    const points = ERP.map((v, i) => ({ x: (i / (ERP.length - 1)) * gw, y: (1 - v) * gh, ...(i === 0 ? { moveTo: true } : {}) }));
    this.slide.addShape(this.pres.ShapeType.custGeom, { x: gx, y: gy, w: gw, h: gh, points, fill: { type: "none" }, line: { color: "000000", width: 2 } });
  }
}

const DESIGNS = [[[0.21, 0.30], [0.54, 0.76], [0.82, 0.44]], [[0.18, 0.76], [0.43, 0.24], [0.75, 0.60]], [[0.32, 0.50], [0.64, 0.84], [0.89, 0.24]]];
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; pres.title = "Block diagrams";
const newSlide = () => { const s = pres.addSlide(); s.background = { color: "FFFFFF" }; return s; };

// ============================== 1. forward model =========================================================================
{
  const s = newSlide(); const c = new Canvas(pres, s, 133, 45, 0.35, 1.0, 9.3, 0.72, 7);
  const Y_ENC = [33, 42.5], Y_Q = [19, 28.5], Y_F = [5, 14.5], ENC_X = [17, 46], CORE_X = [65, 90], DEC_X = [97, 118];
  const mid = (y) => (y[0] + y[1]) / 2;
  c.block(...ENC_X, ...Y_ENC, "Resonator encoder", "set + $f_t$-sorted encoders\n(GNO: graph nodes, STO: tokens)", C.blue, { tsize: 17, ssize: 9.6 });
  c.block(...ENC_X, ...Y_Q, "Resonance query", "$\\delta=f-f_t$, $|\\delta|$, $\\delta^2$ per resonator,\npooled over the resonators", C.green, { tsize: 17, ssize: 9.6 });
  c.block(...ENC_X, ...Y_F, "Frequency encoder", "MLP embedding of $f$\n(or the frequency itself)", C.teal, { tsize: 17, ssize: 9.6 });
  c.block(...CORE_X, 12, 36, "Core architecture", "architecture-specific:\nDON, DNO, FNO, DCO, GNO,\nSTO, SIREN, WNO, LNO\n(NN: plain MLP)", C.peach, { tsize: 18, ssize: 10.2, accent: true });
  c.block(...DEC_X, 15.5, 32.5, "Decoder", "local refinement along $f$,\noutput head", C.purple, { tsize: 17, ssize: 9.6 });
  c.text(77.5, 37.4, "the only block that changes", 10.5, { color: INV, w: 30, h: 3 });
  c.iconPlate(0.8, mid(Y_ENC) - 4, 10.6, 8, DESIGNS[0]);
  c.text(6.1, mid(Y_ENC) - 9.2, "Resonators\n$(m,k,f_t,x,y)$\n$i=1,2,3$", 12.5, { w: 14, h: 8 });
  c.text(0.8, mid(Y_F), "Frequency\n$f$\n301 points", 13, { align: "left", w: 13, h: 8 });
  c.arrow([[12.6, mid(Y_ENC)], [ENC_X[0], mid(Y_ENC)]]); c.arrow([[12.6, mid(Y_F)], [ENC_X[0], mid(Y_F)]]);
  c.arrow([[14.6, mid(Y_ENC)], [14.6, mid(Y_Q) + 1.8], [ENC_X[0], mid(Y_Q) + 1.8]], FWD, { width: 1.1 });
  c.arrow([[14.6, mid(Y_F)], [14.6, mid(Y_Q) - 1.8], [ENC_X[0], mid(Y_Q) - 1.8]], FWD, { width: 1.1 });
  c.dot(14.6, mid(Y_ENC), 1.1, FWD); c.dot(14.6, mid(Y_F), 1.1, FWD);
  const cy = 24;
  [[Y_ENC, cy + 4.5], [Y_Q, cy], [Y_F, cy - 4.5]].forEach(([y, ty]) => c.arrow([[ENC_X[1], mid(y)], [CORE_X[0], ty]]));
  c.arrow([[CORE_X[1], cy], [DEC_X[0], cy]]); c.arrow([[DEC_X[1], cy], [121, cy]]);
  c.iconErp(121.6, cy - 4.5, 10.6, 9);
  c.text(126.9, cy + 7.2, "ERP $\\hat y(f)$", 15, { w: 14, h: 4 });
  s.addText("Frequency encoder: MLP embedding in DON (trunk), DNO, DCO, GNO and STO; the raw frequency enters FNO, WNO, LNO, SIREN and NN.  Resonance query: inside the core for GNO and STO, absent in DON and NN.  Decoder: local refinement is part of the core blocks in FNO and WNO and absent in NN.  Each resonator also carries 120 plate-mode features; ERP values are normalised.", { x: 0.5, y: 4.3, w: 9.0, h: 0.75, fontFace: FONT, fontSize: 9, color: GREY, align: "center", valign: "top", margin: 0, isTextBox: true });
}

// ============================== 2. core architecture list ================================================================
{
  const s = newSlide();
  s.addText("Core architecture of each forward operator", { x: 0.35, y: 0.1, w: 9.3, h: 0.4, fontFace: FONT, fontSize: 20, color: TITLE, align: "center", margin: 0, isTextBox: true });
  const rows = [
    ["DeepONet (DON)", "Inner product", "branch coefficients · trunk basis", C.blue], ["Deep Neural Operator (DNO)", "Modulation (FiLM)", "scale and shift of residual blocks", C.blue],
    ["Deep Cat Operator (DCO)", "Concatenation", "branch, trunk, query → residual MLP", C.blue], ["Fourier Neural Operator (FNO)", "Fourier transform", "spectral convolution along f", C.purple],
    ["Wavelet Neural Operator (WNO)", "Wavelet transform", "3-level Haar wavelet blocks", C.purple], ["Laplace Neural Operator (LNO)", "Poles and residues", "Laplace-domain rational basis", C.purple],
    ["Graph Neural Operator (GNO)", "Graph network", "message passing, attention over resonators", C.peach], ["Set Transformer Operator (STO)", "Attention", "frequencies attend to resonator tokens", C.peach],
    ["SIREN Neural Operator (SIREN)", "Sine layers", "modulated by the design context", C.yellow], ["Plain Neural Network (NN)", "Dense layers", "one MLP on the flattened design and f", C.blue],
  ];
  const RH = 0.38, GAP = 0.07, Y0 = 0.62;
  rows.forEach(([a, core, detail, col], i) => {
    const y = Y0 + i * (RH + GAP);
    s.addText(a, { shape: pres.ShapeType.roundRect, rectRadius: 0.07, x: 0.6, y, w: 3.7, h: RH, fill: { color: col[0] }, line: { color: col[1], width: 1.25 }, fontFace: FONT, fontSize: 12, color: TITLE, align: "center", valign: "middle", margin: 0.02 });
    s.addShape(pres.ShapeType.line, { x: 4.35, y: y + RH / 2, w: 0.6, h: 0, line: { color: FWD, width: 1.5, endArrowType: "triangle" } });
    s.addText([{ text: core, options: { fontSize: 14, color: TITLE, breakLine: true, fontFace: FONT } }, { text: detail, options: { fontSize: 8.5, color: SUB, fontFace: FONT } }], { shape: pres.ShapeType.roundRect, rectRadius: 0.07, x: 5.0, y, w: 4.4, h: RH, fill: { color: col[0] }, line: { color: col[1], width: 1.25 }, align: "center", valign: "middle", margin: 0.01 });
  });
  [["combination", C.blue], ["transform", C.purple], ["relational", C.peach], ["periodic", C.yellow]].forEach(([l, col], i) => {
    const x = 1.2 + i * 2.1;
    s.addShape(pres.ShapeType.roundRect, { rectRadius: 0.05, x, y: 5.28, w: 0.3, h: 0.18, fill: { color: col[0] }, line: { color: col[1], width: 1 } });
    s.addText(l, { x: x + 0.38, y: 5.26, w: 1.4, h: 0.22, fontFace: FONT, fontSize: 10, color: SUB, align: "left", valign: "middle", margin: 0, isTextBox: true });
  });
}

// ============================== 3. general forward / inverse / invertible ==============================================
{
  const s = newSlide();
  const panel = (ox, W, H, title) => { const c = new Canvas(pres, s, W, H, ox, 0.3, 3.0, 0.52, 7); c.text(W / 2, H - 2.6, title, 18, { color: TITLE, w: 30, h: 5 }); return c; };
  const A = (c, cx, y0, y1, sub) => { c.block(cx - 14, cx + 14, y0, y1, "Architecture", sub, C.purple, { tsize: 18, ssize: 11 }); };
  // forward
  { const W = 50, H = 78, c = panel(0.2, W, H, "Forward"); const cx = 25;
    c.iconPlate(cx - 9, H - 19, 18, 13, DESIGNS[0]); c.text(cx, H - 22.2, "Resonator configuration", 13, { color: TITLE, w: 40, h: 4 }); c.text(cx + 15, H - 7.5, "Input", 12, { color: SUB, w: 10, h: 3 });
    A(c, cx, 28, 40, "design → ERP"); c.iconErp(cx - 9, 6, 18, 13); c.text(cx, 2.8, "ERP spectrum", 13, { color: TITLE, w: 40, h: 4 }); c.text(cx + 15, 17.5, "Output", 12, { color: SUB, w: 10, h: 3 });
    c.arrow([[cx, H - 25], [cx, 40.3]], FWD, { width: 2 }); c.arrow([[cx, 27.7], [cx, 19.4]], FWD, { width: 2 });
    c.text(cx, 0.4, "one design gives one ERP  (one-to-one)", 12, { color: FWD, w: 48, h: 2.5 }); }
  // inverse
  { const W = 50, H = 78, c = panel(3.4, W, H, "Inverse"); const cx = 25;
    c.iconErp(cx - 9, H - 19, 18, 13); c.text(cx, H - 22.2, "ERP spectrum", 13, { color: TITLE, w: 40, h: 4 }); c.text(cx + 15, H - 7.5, "Input", 12, { color: SUB, w: 10, h: 3 });
    A(c, cx, 28, 40, "ERP → design");
    for (let i = 0; i < 3; i++) c.iconPlate(cx - 22 + i * 15.5, 7, 13, 10, DESIGNS[i]);
    c.text(cx, 3.6, "Possible designs", 13, { color: TITLE, w: 40, h: 4 }); c.text(3, 20, "Output", 12, { color: SUB, w: 10, h: 3, align: "left" });
    c.arrow([[cx, H - 25], [cx, 40.3]], INV, { dash: true, width: 2 });
    for (let i = 0; i < 3; i++) c.arrow([[cx + (i - 1) * 8, 27.7], [cx - 22 + i * 15.5 + 6.5, 17.4]], INV, { dash: true, width: 2 });
    c.text(cx, 0.4, "many designs share one ERP  (many-to-one)", 12, { color: INV, w: 48, h: 2.5 }); }
  // invertible
  { const W = 50, H = 84, c = panel(6.6, W, H, "Invertible"); const cx = 25; const yTop = H - 22, ya0 = 32, ya1 = 44, yb = 7;
    for (let i = 0; i < 3; i++) c.iconPlate(cx - 22 + i * 15.5, yTop, 13, 10, DESIGNS[i]);
    c.text(cx, yTop + 12.8, "Resonator configurations", 13, { color: TITLE, w: 40, h: 4 });
    A(c, cx, ya0, ya1, "design ↔ ERP"); c.iconErp(cx - 9, yb, 18, 13); c.text(cx, yb - 3.2, "ERP spectrum", 13, { color: TITLE, w: 40, h: 4 });
    for (let i = 0; i < 3; i++) { const xp = cx + (i - 1) * 15.5, xa = cx + (i - 1) * 9;
      c.arrow([[xp - 1.6, yTop - 0.6], [xa - 1.6, ya1 + 0.3]], FWD, { width: 1.75 }); c.arrow([[xa + 1.6, ya1 + 0.3], [xp + 1.6, yTop - 0.6]], INV, { dash: true, width: 1.75 }); }
    c.arrow([[cx - 3, ya0 - 0.3], [cx - 3, yb + 13.4]], FWD, { width: 1.75 }); c.arrow([[cx + 3, yb + 14.2], [cx + 3, ya0 - 0.3]], INV, { dash: true, width: 1.75 });
    c.arrow([[33, 25], [38, 25]], FWD); c.text(39, 25, "forward", 11.5, { color: FWD, align: "left", w: 11, h: 3 });
    c.arrow([[38, 21], [33, 21]], INV, { dash: true }); c.text(39, 21, "inverse", 11.5, { color: INV, align: "left", w: 11, h: 3 });
    c.text(cx, 1.4, "forward: every design gives the same ERP;\ninverse: that ERP gives several designs", 10.5, { color: SUB, w: 50, h: 5 }); }
}

// ============================== 4. Invertible DeepONet ===================================================================
{
  const s = newSlide(); const c = new Canvas(pres, s, 124, 52, 0.35, 0.55, 9.3, 0.74, 7);
  const by = 34, ty = 11;
  c.text(62, 49, "Invertible DeepONet (Q8, Q64)", 19, { color: TITLE, w: 80, h: 4 });
  c.frame(20, 76, 24, 45, C.blue[1]); c.text(48, 42.5, "Branch network (invertible)", 14, { color: TITLE, w: 40, h: 3.5 });
  c.frame(20, 76, 2.5, 20, C.green[1]); c.text(48, 17.5, "Trunk network", 14, { color: TITLE, w: 40, h: 3.5 });
  c.iconPlate(2, by - 5, 13, 10, [[0.3, 0.5], [0.78, 0.5]]); c.text(8.5, by - 9.2, "Design $\\mathbf{a}$\n$(m,f_t,x,y)\\times2$", 11.5, { w: 16, h: 6 });
  c.block(25, 52, by - 6, by + 5, "RealNVP $T$", "bijection on $\\mathbb{R}^Q$, pad with zeros", C.blue, { tsize: 16, ssize: 9.5 });
  c.block(60, 71, by - 6, by + 5, "$\\mathbf{b}$", "", C.blue, { tsize: 17 });
  c.text(65.5, by - 8.2, "$Q$ coefficients", 10.5, { color: SUB, w: 18, h: 3 }); c.text(38.5, by - 8.2, "Q8: $Q=D=8$   |   Q64: $Q=64$", 10, { color: SUB, w: 30, h: 3 });
  c.text(8.5, ty, "Frequency\n$f$", 13, { w: 14, h: 6 });
  c.block(25, 52, ty - 5, ty + 4, "Trunk net", "dense layers", C.green, { tsize: 16, ssize: 9.5 });
  c.block(60, 71, ty - 5, ty + 4, "$\\Psi(f)$", "", C.green, { tsize: 17 });
  c.text(65.5, ty - 7.3, "basis, QR-orthonormal", 10, { color: SUB, w: 24, h: 3 });
  const cx = 88, cr = 3.3; c.circle(cx, by - 0.5, cr, TITLE); c.dot(cx, by - 0.5, 1.0, TITLE);
  c.text(cx, by + 5.2, "$\\Psi\\mathbf{b}+\\psi_0$", 13, { color: TITLE, w: 18, h: 3.5 });
  c.iconErp(108, by - 5.5, 13, 10); c.text(114.5, by + 6.4, "ERP $\\hat y(f)$", 13, { w: 16, h: 3.5 });
  const yf = by + 0.8, yi = by - 3;
  c.arrow([[15, yf], [25, yf]]); c.arrow([[25, yi], [15, yi]], INV, { dash: true });
  c.arrow([[52, yf], [60, yf]]); c.arrow([[60, yi], [52, yi]], INV, { dash: true });
  c.arrow([[71, by - 0.5], [cx - cr, by - 0.5]]); c.arrow([[cx - cr, by - 3.4], [71, by - 3.4]], INV, { dash: true });
  c.arrow([[cx + cr, by - 0.5], [108, by - 0.5]]); c.arrow([[108, by - 3.4], [cx + cr, by - 3.4]], INV, { dash: true });
  c.arrow([[15, ty], [25, ty]]); c.arrow([[52, ty], [60, ty]]); c.arrow([[71, ty], [cx, ty], [cx, by - 0.5 - cr]]);
  c.text(cx + 1.4, 22, "$\\Psi$ is shared by both directions", 10.5, { color: SUB, align: "left", w: 30, h: 3 });
  c.text(98, by - 9.2, "inverse: $\\mathbf{b}^*=\\Psi^\\top(y-\\psi_0)/F$", 11.5, { color: INV, w: 36, h: 3.5 });
  c.arrow([[88.5, 9], [94.5, 9]]); c.text(95.5, 9, "forward: design → ERP", 11, { color: FWD, align: "left", w: 28, h: 3 });
  c.arrow([[94.5, 5.8], [88.5, 5.8]], INV, { dash: true }); c.text(95.5, 5.8, "inverse: ERP → designs", 11, { color: INV, align: "left", w: 28, h: 3 });
}

// ============================== 5. iFNO family ===========================================================================
{
  const s = newSlide(); const c = new Canvas(pres, s, 124, 48, 0.35, 0.6, 9.3, 0.74, 7);
  const FY = 34, IY = 11.5;
  c.text(62, 45.4, "Invertible operators: iFNO and its variants", 19, { color: TITLE, w: 90, h: 4 });
  c.iconPlate(2, FY - 5, 13, 10, [[0.3, 0.5], [0.78, 0.5]]); c.text(8.5, FY + 7.4, "Design $\\mathbf{a}$", 12.5, { w: 16, h: 3.5 });
  c.iconErp(109, FY - 5, 13, 10); c.text(115.5, FY + 7.4, "ERP $\\hat y(f)$", 12.5, { w: 16, h: 3.5 });
  c.iconErp(109, IY - 5, 13, 10); c.text(115.5, IY - 8, "target ERP $y(f)$", 12, { w: 20, h: 3.5 });
  c.iconPlate(2, IY - 5, 13, 10, [[0.3, 0.5], [0.78, 0.5]]); c.text(8.5, IY - 8, "designs $\\hat{\\mathbf{a}}$", 12, { w: 16, h: 3.5 });
  c.block(24, 42, FY - 5.5, FY + 5.5, "Lift $P$", "design, $f$ → latent", C.blue, { tsize: 16, ssize: 10 });
  c.block(24, 42, IY - 5.5, IY + 5.5, "Readout $Q'$", "latent → design", C.peach, { tsize: 16, ssize: 10 });
  c.block(82, 100, FY - 5.5, FY + 5.5, "Readout $Q$", "latent → ERP", C.blue, { tsize: 16, ssize: 10 });
  c.block(82, 100, IY - 5.5, IY + 5.5, "Lift $P'$", "ERP → latent", C.peach, { tsize: 16, ssize: 10 });
  c.block(50, 74, 4, 42.5, "", "", C.purple, { noText: true });
  c.text(62, 40, "Coupling stack", 16, { color: TITLE, w: 22, h: 4 }); c.text(62, 37.2, "same weights both ways", 10, { color: SUB, w: 22, h: 3 });
  c.block(52, 72, 15.5, 33, "Gate $L$", "changes with the model:\niFNO, iDCO, iGNO, iDNO,\niWNO, iLNO, iSIREN, iSTO", C.peach, { tsize: 15, ssize: 10, dashed: true });
  c.arrow([[15, FY], [24, FY]]); c.arrow([[42, FY], [50, FY]]); c.arrow([[74, FY], [82, FY]]); c.arrow([[100, FY], [109, FY]]);
  c.arrow([[109, IY], [100, IY]], INV, { dash: true }); c.arrow([[82, IY], [74, IY]], INV, { dash: true }); c.arrow([[50, IY], [42, IY]], INV, { dash: true }); c.arrow([[24, IY], [15, IY]], INV, { dash: true });
  c.text(62, 1.7, "Only the coupling stack is an exact bijection; a β-VAE on the design estimate gives several candidate designs.", 10.5, { color: GREY, w: 110, h: 3 });
}
pres.writeFile({ fileName: path.join(__dirname, "block_diagrams_slides.pptx") }).then((f) => console.log("saved", f));
