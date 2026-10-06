"""Presentation figure: forward neural operator (final Deep Cat Operator, DCO_sorted_phys) and its loss.

Every number below is read from the code / checkpoint (see the notes in the final answer):
  hidden 67, branch 56 (set) + 56 (sorted), trunk 56, query 28, depth 4, SiLU; 155,240 parameters;
  AdamW lr 5e-4 (wd 1e-4), cosine to 1e-6, 200 epochs, batch 128, grad-clip 5; loss =
  MSE + 0.5 slope + 0.05 peak; 100k configurations, 80/10/10, seed 727.
"""
from diagram_kit import *  # noqa: F401,F403
from diagram_kit import BLUE, GREEN, ORANGE, YELLOW, GRAY, PURPLE

fig, ax = new_figure()

# ---- outer frames -----------------------------------------------------------------------
panel(ax, 1.0, 4.4, 63.8, 49.6, fc="#f6f6f6", ls="--", lw=1.2, r=2.0, z=0)
panel(ax, 66.0, 4.4, 33.2, 49.6, fc="#f6f6f6", ls="--", lw=1.2, r=2.0, z=0)
text(ax, 32.9, 52.4, "Forward neural operator: Deep Cat Operator (DCO)", size=12.5)
text(ax, 32.9, 50.6, r"resonator design and frequency $\rightarrow$ ERP spectrum;  155,240 parameters", size=8.5, color="#555555")
text(ax, 82.6, 52.4, "Loss functions", size=12.5)
text(ax, 2.4, 48.9, r"$\mathbf{r}_i=(m_i,k_i,f_{t,i},x_i,y_i)$, $i=1,2,3$ (z-scored)", size=7.8, color="#444444", ha="left")


def net_cols(c_top, c_bot, xs=(11.6, 16.6), size=7.5, r=0.85, with_dots=True):
    """Two columns of sigma neurons (top, dots, bottom) with full links between them."""
    mlp(ax, list(xs), [[c_top, None, c_bot], [c_top, None, c_bot]], size=size, r=r)
    if with_dots:
        for x in xs:
            dots(ax, x, (c_top + c_bot) / 2, size=8)


def outputs(y_top, y_bot, labels, x=23.7):
    for lab, yy in zip(labels, (y_top, y_bot)):
        neuron(ax, x, yy, lab, r=1.05, size=6.4)
    dots(ax, x, (y_top + y_bot) / 2, size=8)


# ---- Branch net: two encoders -------------------------------------------------------------
BR_Y0, BR_Y1 = 30.0, 47.5
panel(ax, 2.3, BR_Y0, 4.2, BR_Y1 - BR_Y0, fc=BLUE, r=1.4)
for yy, lab in zip((43.6, 38.8, 34.0), (r"$\mathbf{r}_1$", r"$\mathbf{r}_2$", r"$\mathbf{r}_3$")):
    neuron(ax, 4.4, yy, lab, r=1.05, size=7.5)
rows = [(39.0, 47.5, "Set encoder", "shared MLP per resonator,\nmean + max pool", (r"$c_1$", r"$c_{56}$")),
        (30.0, 38.4, "Sorted encoder", "resonators sorted by $f_t$,\nconcatenated, MLP", (r"$c'_1$", r"$c'_{56}$"))]
branch_out_y = []
for y0, y1, title, cap, labs in rows:
    panel(ax, 7.8, y0, 12.6, y1 - y0, fc=BLUE, r=1.2)
    text(ax, 14.1, y1 - 1.0, title, size=8.8)
    ymid = (y0 + y1) / 2
    net_cols(ymid + 0.9, ymid - 0.9, size=6.8, r=0.75, with_dots=False)
    text(ax, 14.1, y0 + 1.15, cap, size=6.6, color="#333333", linespacing=1.15)
    arrow(ax, [(6.5, ymid), (7.8, ymid)], lw=0.9, ms=6)
    arrow(ax, [(17.5, ymid + 0.5), (22.6, ymid + 0.5)], lw=0.7, ms=5) if False else None
panel(ax, 21.6, BR_Y0, 4.2, BR_Y1 - BR_Y0, fc=BLUE, r=1.4)
for (y0, y1, _, _, labs) in rows:
    ymid = (y0 + y1) / 2
    outputs(ymid + 1.9, ymid - 1.9, labs)
    arrow(ax, [(20.4, ymid), (21.6, ymid)], lw=0.9, ms=6)
    branch_out_y.append(ymid)


# ---- Resonance query and trunk ---------------------------------------------------------------
def simple_band(y0, h, title, fc, in_labels, out_labels, caption, out_dim):
    panel(ax, 2.3, y0, 4.2, h, fc=fc, r=1.4)
    panel(ax, 7.8, y0, 12.6, h, fc=fc, r=1.2)
    panel(ax, 21.6, y0, 4.2, h, fc=fc, r=1.4)
    text(ax, 14.1, y0 + h - 1.1, title, size=9.0)
    c = y0 + h / 2 + 0.4
    ys = [c] if len(in_labels) == 1 else [c + 1.7, c - 1.7]
    for lab, yy in zip(in_labels, ys):
        neuron(ax, 4.4, yy, lab, r=1.05, size=7.5)
    arrow(ax, [(6.5, c), (7.8, c)], lw=0.9, ms=6)
    net_cols(c + 1.6, c - 1.9, size=6.8, r=0.75)
    text(ax, 14.1, y0 + 1.3, caption, size=6.6, color="#333333", linespacing=1.15)
    outputs(c + 1.9, c - 1.9, out_labels)
    arrow(ax, [(20.4, c), (21.6, c)], lw=0.9, ms=6)
    return c


Q0, QH = 17.4, 11.6
T0, TH = 5.6, 11.0
c_q = simple_band(Q0, QH, "Resonance query", GREEN, [r"$\mathbf{r}_i$", r"$f$"], [r"$q_1$", r"$q_{28}$"],
                  "per resonator and frequency:\n$\\mathbf{r}_i, f,\\delta,|\\delta|,\\delta^2$; MLP; mean + max", 28)
c_t = simple_band(T0, TH, "Trunk net (frequency)", GREEN, [r"$f$"], [r"$t_1$", r"$t_{56}$"],
                  "MLP on the frequency\n$1\\rightarrow67\\rightarrow56$", 56)

# ---- concatenation ----------------------------------------------------------------------
CY, CX = c_q, 30.4
circle_op(ax, CX, CY, "cat", r=1.45, size=7.5)
top_y = branch_out_y[0]
arrow(ax, [(25.8, top_y), (CX, top_y), (CX, CY + 1.45)], lw=1.0)
ax.plot([25.8, CX], [branch_out_y[1], branch_out_y[1]], lw=1.0, color="black", zorder=6)
arrow(ax, [(25.8, CY), (CX - 1.45, CY)], lw=1.0)
arrow(ax, [(25.8, c_t), (CX, c_t), (CX, CY - 1.45)], lw=1.0)
text(ax, CX - 2.1, CY - 3.0, "$196$", size=7.4, color="#555555")
text(ax, CX + 3.2, CY + 2.4, "", size=6)

# ---- operator core ----------------------------------------------------------------------
panel(ax, 33.4, 7.4, 18.4, 41.0, fc=PURPLE, r=1.6, z=1)
text(ax, 42.6, 47.0, r"Operator core $\mathcal{G}_\theta$ (pointwise in $f$)", size=9.6)
core = [
    (38.0, "Lift\nLinear $196\\!\\rightarrow\\!67$, SiLU"),
    (28.6, "$4\\times$ residual block, width 67\n$h\\leftarrow\\mathrm{SiLU}(\\mathrm{LN}(h+\\mathrm{MLP}(h)))$"),
    (19.2, "Local refinement along $f$\n$h\\leftarrow\\mathrm{SiLU}(\\mathrm{GN}(h+\\mathrm{Conv}(h)))$\nConv: Conv1d, SiLU, Conv1d ($k{=}3$)"),
    (9.8, "Output head\nMLP $67\\!\\rightarrow\\!33\\!\\rightarrow\\!1$"),
]
for yy, s in core:
    box(ax, 34.6, yy, 16.0, 7.2, s, size=8.0, fc="white")
for yy_top, yy_bot in ((38.0, 35.8), (28.6, 26.4), (19.2, 17.0)):
    arrow(ax, [(42.6, yy_top), (42.6, yy_bot)], lw=1.0, ms=7)
arrow(ax, [(CX + 1.45, CY), (33.4, CY)], lw=1.1)

# ---- output -----------------------------------------------------------------------------
panel(ax, 53.2, 17.0, 10.4, 26.0, fc=ORANGE, r=1.6)
text(ax, 58.4, 41.5, "Output", size=10.5)
spectrum_inset(fig, ax, 54.6, 29.0, 8.0, 8.4, sample_erp())
text(ax, 58.4, 26.4, r"predicted ERP", size=8.2)
text(ax, 58.4, 24.4, r"$\hat y(f)$", size=10)
text(ax, 58.4, 21.5, "301 frequencies\n10 to 160 Hz", size=7.2, color="#444444", linespacing=1.3)
arrow(ax, [(51.8, 27.0), (53.2, 27.0)], lw=1.1)

# ---- loss panel -------------------------------------------------------------------------
box(ax, 70.0, 43.4, 20.0, 5.0, "Training data: true ERP $y(f)$", fc=YELLOW, size=9)
LOSS = [
    (34.2, "MSE loss", r"$\frac{1}{F}\sum_f(\hat y_f-y_f)^2$"),
    (25.4, r"Slope loss ($\times\,0.5$)", r"$\frac{1}{F-1}\sum_f(\Delta\hat y_f-\Delta y_f)^2$"),
    (16.6, r"Peak loss ($\times\,0.05$)", r"$\sum_{f\in\mathcal{P}(y)}(\hat y_f-y_f)^2$"),
]
mids = []
for yy, name, formula in LOSS:
    panel(ax, 70.0, yy, 22.0, 6.8, fc=GRAY, r=1.0, z=3)
    text(ax, 81.0, yy + 5.2, name, size=9.4)
    text(ax, 81.0, yy + 2.6, formula, size=9.6)
    mids.append(yy + 3.4)
ax.plot([90.0, 95.8, 95.8], [45.9, 45.9, mids[-1]], lw=1.1, color="black", zorder=6)
for m in mids:
    arrow(ax, [(95.8, m), (92.2, m)], lw=1.1, ms=8)
arrow(ax, [(63.6, 30.0), (67.8, 30.0)], lw=1.1, head=False)
ax.plot([67.8, 67.8], [mids[-1], mids[0]], lw=1.1, color="black", zorder=6)
for m in mids:
    arrow(ax, [(67.8, m), (69.9, m)], lw=1.1, ms=8)
text(ax, 65.4, 31.4, r"$\hat y$", size=9)
circle_op(ax, 81.0, 33.2, "+", r=0.85, size=9)
circle_op(ax, 81.0, 24.4, "+", r=0.85, size=9)
text(ax, 82.4, 14.1, "$\\mathcal{P}(y)$: the true spectrum's\nresonance peaks (local maxima)", size=7.0, color="#555555", ha="left", linespacing=1.2)
arrow(ax, [(81.0, 16.6), (81.0, 12.0)], lw=1.1)
diamond(ax, 81.0, 9.0, 15.0, 6.2, r"$\mathrm{arg\,min}_\theta\,\mathcal{L}(\theta)$", size=9.5)
arrow(ax, [(73.5, 9.0), (51.9, 9.0)], lw=1.1)
text(ax, 58.2, 11.0, "AdamW ($5\\times10^{-4}$), cosine decay,\n200 epochs, batch 128", size=7.0, color="#555555", linespacing=1.2)

# ---- footer -----------------------------------------------------------------------------
text(ax, 50.0, 2.9, r"Each $\mathbf{r}_i$ is extended by 120 plate-mode features: $\sin(a\pi x/L_x)$, $\sin(b\pi y/L_y)$ and their products, $a,b=1,\ldots,10$. "
                    "ERP values are normalised before the loss.", size=8.2, color="#444444")
text(ax, 50.0, 1.3, "Same data, output and loss for all forward operators (DON, DNO, FNO, GNO, STO, SIREN, WNO, LNO, NN); they differ in the operator core and in how the resonators are encoded.   "
                    "Data: 100k configurations, 3 resonators, 80/10/10 split.", size=8.2, color="#444444")
save(fig, "forward_operator")
