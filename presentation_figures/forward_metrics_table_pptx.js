// One-slide PowerPoint with the forward-operator table built from native, editable shapes (rounded rectangles + text),
// not a picture. Values: erp_forward/plots/models/100k/ALL_MODELS/forward_models_metrics.csv. Run: node forward_metrics_table_pptx.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const csv = fs.readFileSync(path.join(__dirname, "..", "erp_forward", "plots", "models", "100k", "ALL_MODELS", "forward_models_metrics.csv"), "utf8")
  .trim().split("\n");
const header = csv[0].split(",");
const rowsCsv = csv.slice(1).map((l) => Object.fromEntries(l.split(",").map((v, i) => [header[i], v])));
const byModel = Object.fromEntries(rowsCsv.map((r) => [r["Model"], r]));

const COMB = ["DBE8F6", "8FB4D9"], TRANS = ["EBE2F5", "B39AD1"], REL = ["FDE7D3", "EBA46F"], PER = ["FDF3C9", "E0C25A"];
const GOOD = ["E1F1DE", "97C791"], BAD = ["F8C4C4", "D96B6B"], CELL = ["FFFFFF", "C3CCD6"];
const TITLE = "1C3550", SUB = "4A5A6A", FONT = "Cambria";

const ROWS = [
  ["DeepONet", "DeepONet (DON)", "Inner product", COMB],
  ["Deep Neural Operator", "Deep Neural Operator (DNO)", "Modulation (FiLM)", COMB],
  ["Deep Cat Operator", "Deep Cat Operator (DCO)", "Concatenation", COMB],
  ["Fourier Neural Operator", "Fourier Neural Operator (FNO)", "Fourier transform", TRANS],
  ["Wavelet Neural Operator", "Wavelet Neural Operator (WNO)", "Wavelet transform", TRANS],
  ["Laplace Neural Operator", "Laplace Neural Operator (LNO)", "Poles and residues", TRANS],
  ["Graph Neural Operator", "Graph Neural Operator (GNO)", "Graph network", REL],
  ["Set Transformer Operator", "Set Transformer Operator (STO)", "Attention", REL],
  ["SIREN Neural Operator", "SIREN Neural Operator (SIREN)", "Sine layers", PER],
  ["Plain Neural Network", "Plain Neural Network (NN)", "Dense layers", COMB],
];
const COLS = [ // csv column, header, decimals, lower is better
  ["RMSE (dB)", "RMSE (dB)", 2, true],
  ["R2", "R²", 3, false],
  ["Pearson r (all points)", "Correlation", 3, false],
  ["RMSE at all peaks (dB)", "Peak RMSE (dB)", 2, true],
];
const vals = (c) => ROWS.map((r) => parseFloat(byModel[r[0]][c]));
const stats = Object.fromEntries(COLS.map(([c, , , low]) => {
  const v = vals(c); const best = low ? Math.min(...v) : Math.max(...v); const worst = low ? Math.max(...v) : Math.min(...v);
  return [c, { best, worst }];
}));

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Forward operators: core operation and test error";
const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };
slide.addText("Forward operators: core operation and test error", {
  x: 0.35, y: 0.12, w: 9.3, h: 0.42, fontFace: FONT, fontSize: 20, color: TITLE, align: "center", margin: 0, isTextBox: true, objectName: "Title",
});

const X_ARCH = 0.35, W_ARCH = 2.5, X_CORE = 2.93, W_CORE = 1.85, X_MET = 4.86, W_MET = 1.07, G_MET = 0.085;
const Y_HEAD = 0.62, Y0 = 0.9, RH = 0.34, GAP = 0.06;
const block = (x, y, w, text, [fill, line], opts = {}) => slide.addText(text, {
  shape: pres.ShapeType.roundRect, rectRadius: 0.07, x, y, w, h: RH, fill: { color: fill }, line: { color: line, width: 1.25 },
  fontFace: FONT, fontSize: opts.size || 10.5, color: TITLE, align: "center", valign: "middle", margin: 0.02, objectName: opts.name,
});
const label = (x, w, text) => slide.addText(text, {
  x, y: Y_HEAD, w, h: 0.26, fontFace: FONT, fontSize: 11, color: SUB, align: "center", valign: "middle", margin: 0, isTextBox: true,
});
label(X_ARCH, W_ARCH, "Architecture");
label(X_CORE, W_CORE, "Core operation");
COLS.forEach(([, h], j) => label(X_MET + j * (W_MET + G_MET), W_MET, h));

ROWS.forEach(([key, name, core, color], i) => {
  const y = Y0 + i * (RH + GAP);
  block(X_ARCH, y, W_ARCH, name, color, { name: `Architecture ${i + 1}` });
  block(X_CORE, y, W_CORE, core, color, { size: 11, name: `Core ${i + 1}` });
  COLS.forEach(([c, , dec], j) => {
    const v = parseFloat(byModel[key][c]); const s = stats[c];
    const t = (v - s.best) / (s.worst - s.best);
    const col = Math.abs(v - s.best) < 1e-9 ? GOOD : (t > 0.5 ? BAD : CELL);
    block(X_MET + j * (W_MET + G_MET), y, W_MET, v.toFixed(dec), col, { size: 11.5, name: `Value ${i + 1}-${j + 1}` });
  });
});
slide.addText("Latest 200-epoch models, 100k dataset (10,000 test spectra). Correlation: Pearson r over all points. Peak RMSE: error at the resonance peaks of the true ERP.\nBest value of each column in green; red: worse than halfway between the best and the worst value of the column.", {
  x: 0.35, y: 5.0, w: 9.3, h: 0.5, fontFace: FONT, fontSize: 9, color: SUB, align: "center", valign: "top", margin: 0, isTextBox: true, objectName: "Footnote",
});
pres.writeFile({ fileName: path.join(__dirname, "forward_metrics_table_slide.pptx") }).then((f) => console.log("saved", f));
