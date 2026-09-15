import os
"""Redraw Fig. 2 (primary performance) and Fig. 3 (validity) as vector PDFs.

Values are exactly those published in the manuscript / frozen artifacts.
Layout rule that avoids the earlier collisions: every panel reserves a
right-hand annotation band. All interval text is LEFT-aligned at the band
start, which lies strictly to the right of the largest interval end, so text
can never overlap a marker or a whisker.
"""
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7.2,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '#8a8a8a',
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.color': '#4a4a4a', 'ytick.color': '#4a4a4a',
    'text.color': '#1a1a1a', 'axes.labelcolor': '#1a1a1a',
    'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
MAIN, ROB, ACC = '#12695f', '#6b5b95', '#c2703a'
LIGHT, GREY = '#d9d9d9', '#7a7a7a'
OUT = os.path.dirname(os.path.abspath(__file__))
ANN_FS = 6.2


def badge(ax, letter, title):
    ax.text(0.0, 1.13, letter, transform=ax.transAxes, fontsize=8.0,
            fontweight='bold', color='white', va='center', ha='center',
            clip_on=False, bbox=dict(boxstyle='round,pad=0.30', fc='#1f3b57', ec='none'))
    ax.text(0.062, 1.13, title, transform=ax.transAxes, fontsize=8.0,
            fontweight='bold', color='#1a1a1a', va='center', ha='left', clip_on=False)


def frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=2)


def forest(ax, rows, xlim, band, xlabel, colors, xticks, xticklabels=None,
           vline=None, vlabel=None, fs=ANN_FS):
    """rows: (label, centre, lo, hi, annotation). Annotation drawn at `band`."""
    n = len(rows)
    ys = list(range(n))[::-1]
    if vline is not None:
        ax.axvline(vline, color=LIGHT, lw=0.8, zorder=0)
        if vlabel:
            ax.text(vline, n - 0.30, vlabel, fontsize=6.0, color=GREY, ha='center')
    for y, (lab, c, lo, hi, ann), col in zip(ys, rows, colors):
        if lo is None:                      # range bar
            ax.plot([c, hi], [y, y], color=col, lw=4.2, alpha=.5,
                    solid_capstyle='butt', zorder=2)
        else:
            ax.plot([lo, hi], [y, y], color=col, lw=1.5,
                    solid_capstyle='round', zorder=2)
            ax.plot([lo, lo], [y - .13, y + .13], color=col, lw=1.1, zorder=2)
            ax.plot([hi, hi], [y - .13, y + .13], color=col, lw=1.1, zorder=2)
            ax.plot([c], [y], 'o', ms=4.4, color=col, zorder=3)
        if ann:
            ax.text(band, y, ann, fontsize=fs, color=col, va='center', ha='left', zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.9)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    if xticklabels:
        ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel, fontsize=7.1, labelpad=2)
    frame(ax)


def dumbbell(ax, labels, left, right, xlim, xticks, xlabel, xticklabels=None,
             left_name='', right_name=''):
    """Two-point dumbbell with outward, collision-free value labels."""
    n = len(labels)
    ys = list(range(n))[::-1]
    span = xlim[1] - xlim[0]
    for y, a, b in zip(ys, left, right):
        ax.plot([a, b], [y, y], color=LIGHT, lw=1.1, zorder=1)
        ax.plot([a], [y], 'o', ms=4.4, mfc='white', mec=GREY, mew=1.1, zorder=3)
        ax.plot([b], [y], 'o', ms=5.0, color=MAIN, zorder=3)
        ax.text(a - 0.012 * span, y, f'{a:.3f}'.lstrip('0'), fontsize=6.2,
                color=GREY, ha='right', va='center')
        ax.text(b + 0.012 * span, y, f'{b:.3f}'.lstrip('0'), fontsize=6.2,
                color=MAIN, ha='left', va='center')
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=6.9)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    if xticklabels:
        ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel, fontsize=7.1, labelpad=2)
    frame(ax)
    ax.plot([], [], 'o', ms=4.4, mfc='white', mec=GREY, mew=1.1, label=left_name)
    ax.plot([], [], 'o', ms=5.0, color=MAIN, label=right_name)


# ================================================================= Fig. 2
fig = plt.figure(figsize=(7.16, 3.70))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.62)

methods = ['RA-SCOPE', 'Uniform pool', 'Length ordinal', 'Fine-tuned MacBERT',
           'Standard measurements', 'Character TF--IDF', 'Qwen2.5-3B rubric']
main_q = [.823, .790, .767, .763, .622, .577, .538]
rob_q = [.786, .786, .772, .672, .620, .487, .603]

axA = fig.add_subplot(gs[0, 0])
ys = list(range(len(methods)))[::-1]
for y, m, r in zip(ys, main_q, rob_q):
    axA.plot([m, r], [y, y], color=LIGHT, lw=1.1, zorder=1)
for y, m in zip(ys, main_q):
    axA.plot([m], [y], 'o', ms=5.0, color=MAIN, zorder=3)
for y, r in zip(ys, rob_q):
    axA.plot([r], [y], 'D', ms=4.4, color=ROB, zorder=3)
# reserved numeric read-out band, strictly right of every marker
B0, DX = 0.885, 0.105
for y, m, r in zip(ys, main_q, rob_q):
    axA.text(B0, y, f'{m:.3f}'.lstrip('0'), fontsize=6.2, color=MAIN,
             ha='center', va='center')
    axA.text(B0 + DX, y, f'{r:.3f}'.lstrip('0'), fontsize=6.2, color=ROB,
             ha='center', va='center')
axA.text(B0, len(methods) - 0.32, 'main', fontsize=6.2, color=MAIN, ha='center')
axA.text(B0 + DX, len(methods) - 0.32, 'rob.', fontsize=6.2, color=ROB, ha='center')
axA.set_yticks(ys)
axA.set_yticklabels(methods, fontsize=6.9)
axA.set_ylim(-0.8, len(methods) - 0.02)
axA.set_xlim(0.44, 1.03)
axA.set_xticks([0.45, 0.55, 0.65, 0.75, 0.85])
axA.set_xlabel('Quadratic weighted kappa', fontsize=7.1, labelpad=2)
frame(axA)
badge(axA, 'A', 'QWK across two evaluation sets')
axA.plot([], [], 'o', ms=5.0, color=MAIN, label='Main  $n{=}120$')
axA.plot([], [], 'D', ms=4.4, color=ROB, label='Robustness  $n{=}28$')
axA.legend(loc='upper left', fontsize=6.2, frameon=False, handletextpad=0.35,
           borderpad=0.18, labelspacing=0.25)

rowsB = [
    ('Main vs Qwen2.5-3B', .285, .173, .403, '.285 [.173,.403]'),
    ('Main vs MacBERT', .060, .008, .118, '.060 [.008,.118]'),
    ('Main vs Length ordinal', .056, .013, .114, '.056 [.013,.114]'),
    ('Main vs Uniform pool', .032, .005, .066, '.032 [.005,.066]'),
    ('Robust vs MacBERT', .114, .030, .227, '.114 [.030,.227]'),
    ('Robust vs Qwen2.5-3B', .183, .008, .313, '.183 [.008,.313]'),
]
axB = fig.add_subplot(gs[0, 1])
forest(axB, rowsB, (0.0, 0.88), 0.45, r'$\Delta$QWK = RA-SCOPE $-$ comparator',
       [MAIN] * 4 + [ROB] * 2, [0.0, 0.1, 0.2, 0.3, 0.4])
badge(axB, 'B', 'Paired $\\Delta$QWK (95% CI)')
fig.savefig(OUT + r'\fig2_performance.pdf')
fig.savefig(OUT + r'\preview2.png', dpi=170)
plt.close(fig)

# ================================================================= Fig. 3
fig = plt.figure(figsize=(7.16, 3.74))
gs = fig.add_gridspec(2, 2, hspace=0.88, wspace=0.60)

ax = fig.add_subplot(gs[0, 0])
forest(ax, [
    ('RA-SCOPE', .400, .258, .545, '[.258,.545]'),
    ('Length expert', -.034, -.091, .090, '$[-.091,.090]$'),
    ('paired $\\Delta$ partial $\\rho$', .398, .238, .573, '[.238,.573]'),
], (-0.12, 0.94), 0.63, r'Partial Spearman $\rho$', [MAIN, ROB, ACC],
    [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    ['$-.1$', '0', '.1', '.2', '.3', '.4', '.5', '.6'], vline=0.0)
badge(ax, 'A', 'Residual signal beyond length')

ax = fig.add_subplot(gs[0, 1])
forest(ax, [
    ('Counter-length  ($n{=}24$)', .708, .500, .875, '[.500,.875]'),
    ('Nearest level  ($n{=}56$)', .875, .786, .946, '[.786,.946]'),
], (0.0, 1.36), 0.98, 'Direction accuracy', [MAIN, ROB],
    [0.0, 0.25, 0.5, 0.75, 1.0], vline=0.5, vlabel='chance .50')
ax.text(0.0, -0.58, 'Length $=.000$ by construction', fontsize=6.0, color=GREY)
badge(ax, 'B', 'Disjoint hard-pair probes')

ax = fig.add_subplot(gs[1, 0])
forest(ax, [
    ('Mean of 10 criteria $\\leftrightarrow$ global', .935, .907, .952, '[.907,.952]'),
    ('After length / pages / source', .720, .589, .800, '[.589,.800]'),
    ('Leave-one-dimension-out', .9305, None, None, r'$\rho$ .920--.941'),
], (0.50, 1.26), 0.98, 'Spearman $\\rho$', [MAIN, ROB, ACC],
    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ['.50', '.60', '.70', '.80', '.90', '1.00'])
badge(ax, 'C', 'Standard-derived construct alignment')

ax = fig.add_subplot(gs[1, 1])
dumbbell(ax,
         ['10-criterion macro', 'Illustration', 'Pinyin'],
         [.456, .309, .260], [.532, .365, .966],
         (0.12, 1.14), [0.0, 0.25, 0.5, 0.75, 1.0], 'QWK',
         left_name='original RA-SCOPE', right_name='capability-routed')
ax.set_ylim(-0.8, 3.45)          # headroom so the legend clears the top row
ax.legend(loc='upper right', fontsize=6.2, frameon=False, handletextpad=0.35,
          borderpad=0.18, labelspacing=0.25)
badge(ax, 'D', 'Observable-modality routing')

fig.savefig(OUT + r'\fig3_validity.pdf')
fig.savefig(OUT + r'\preview3.png', dpi=170)
plt.close(fig)
print('wrote fig2_performance.pdf, fig3_validity.pdf, previews')
