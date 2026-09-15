# RA-SCOPE ImageGen figure-set contract v7.0

## Scope

Figures 1--3 retain the accepted assets and prompt contracts recorded in
`figure_set_imagegen_contract_v6.2.md`. Figure 4 is replaced by a high-density
statistical visualization. Its visual base is authored with ImageGen; exact marker,
interval, range, and dumbbell geometry is then locked on the same raster canvas from
the authoritative result files.

## Shared visual contract

- Wide, warm-white scientific canvas with four aligned panels and ample whitespace.
- Deep navy typography; teal primary estimates; slate baselines; violet adjusted or
  secondary estimates; amber sensitivity ranges.
- Subtle panel boundaries and fine gridlines; color is reinforced by position and
  marker shape.
- No large dark field, KPI cards, oversized numerals, gradients, 3-D treatment,
  decorative icons, or dashboard ornament.
- All critical labels remain legible at full two-column width.

## Figure 4 ImageGen redesign prompt

```text
Create a new publication-grade scientific composite figure for a top-tier NLP paper.

Image 1 is the authoritative content, data, label, axis, and geometry reference.
Image 2 is a style/composition reference only. Image 3 is the figure-set style
reference. Redesign Figure 4 as four true quantitative visualizations, not KPI cards
and not a dashboard made of large standalone numbers. Every important number must be
visually encoded by marker position, interval length, or paired displacement.

Use a wide 2x2 white scientific canvas with restrained navy typography, teal/slate/
violet/amber marks, subtle panels, fine pale-gray gridlines, generous whitespace,
and small A--D badges. Use no black background, gradient, oversized numeral,
decorative icon, large arrow, or card-like number box.

A "Residual signal beyond length": forest plot on Partial Spearman rho, axis -.10 to
.60. RA-SCOPE .400 [.258,.545]; Length expert -.034 [-.091,.090]; paired delta
partial rho interval [.238,.573].

B "Disjoint hard-pair probes": two interval tracks on Direction accuracy, axis 0 to
1, chance line .50. Counter-length n=24: .708 [.500,.875], Length=.000 by
construction. Nearest different level n=56: .875 [.786,.946], continuous direction
accuracy.

C "Standard-derived construct alignment": correlation forest/range plot on
Spearman rho, axis .50 to 1.00, guide .90. Mean of 10 criteria to global: .935
[.907,.952]. After length/pages/source control: .720 [.589,.800].
Leave-one-dimension-out range [.920,.941].

D "Observable-modality routing": proportional before-to-after dumbbell plot on QWK,
axis 0 to 1. Macro .456 to .532, delta .076 [95% CI .055,.096]; Illustration .309
to .365; Pinyin .260 to .966. Open slate circles denote original RA-SCOPE and filled
teal circles capability-routed.

Preserve every value, sign, interval endpoint, row order, axis, qualifier, and title.
Keep labels separate from plotted marks and do not invent values.
```

## Figure 4 geometry-lock contract

The accepted ImageGen visual base is
`exec-c7c9eda5-3e23-4690-a207-bb1bfe2aaebd.png`. The finalization script
`scripts/finalize_figure4_imagegen_geometry.py` retains the ImageGen canvas, titles,
panel system, palette, and raster finish while replacing chart interiors with marks
computed from the locked JSON values. The script refuses to render if the principal
source values no longer match the figure contract.

Exact scales and encodings:

- A: shared axis [-.10,.60], zero reference, two point intervals plus one paired
  difference interval.
- B: shared axis [0,1], chance reference .50, two point intervals.
- C: shared axis [.50,1], guide .90, two point intervals plus one range.
- D: shared axis [0,1], three proportional before/after dumbbells.

## Acceptance checks

- The length-expert diamond is left of zero at -.034 and its interval spans -.091 to
  .090.
- The controlled construct estimate is at .720 on the .50--1.00 axis, with interval
  .589--.800.
- All B intervals and all D endpoints are proportional to their common 0--1 axes.
- Numerical labels support the geometry rather than replacing it.
- The accepted manuscript asset is a single PNG raster and remains readable in the
  compiled anonymous and author PDFs.
