---
title: "Gait Analysis Methods"
subtitle: "DICE split-belt treadmill, military load carriage — how every metric is calculated, why each step is needed, and what a healthy value looks like"
date: "Version 1 — September 2026 — companion to gait_analysis.py and detect_gait_events.py"
---

# How to read this document

This document and the script `gait_analysis.py` are meant to be read side by side. Every section here has the **same number** as the section of the code that implements it. A section header in the code looks like this:

```
# %% ==========================================================================
# §10  MARGIN OF STABILITY                         (Word document §10, Fig. 17)
# =============================================================================
```

In Spyder, each `# %%` line starts a cell, so the outline pane lists the sections in the same order as this document. Sections 2–4 (equipment, boot meshes, gait events) are carried out by `build_foot_binding.py` and `detect_gait_events.py`, and every section from 5 onwards by `gait_analysis.py`.

Each metric section has the same parts, in the same order:

1. **What it measures**, in words.
2. **The equations.**
3. **The exact steps** the code takes, numbered.
4. **Why**: the reason each step is needed and the reason for each choice. Where another method exists, this explains why it was not used.
5. **A figure** drawn by the real code on a synthetic trial whose true answers are known (Section 20).
6. **The healthy range**: the values healthy, unloaded adults show in the literature, so you can see where your participants fall.

Three coloured boxes appear throughout:

::: {custom-style="Code Box"}
**In the code** — the function in `gait_analysis.py` (or `detect_gait_events.py`) that carries out the step, and the settings from §0 that control it.
:::

::: {custom-style="Why Box"}
**Why** — the reason for a step or a choice.
:::

::: {custom-style="Range Box"}
**Healthy range** — what to expect, with the source. The same ranges colour every trial in the "Healthy ranges" sheet of the results workbook (Section 19).
:::

::: {custom-style="Note Box"}
**Caution** — something that changes how a number should be read (for example, the metronome in these trials).
:::

## Two things to keep in mind when you compare with the healthy ranges

**The ranges are for healthy adults without a load, mostly walking overground at 1.2–1.4 m/s.** Your participants walk on a treadmill at 1.3 m/s, to a 108 beats/min metronome, carrying a military load. Values outside a range are therefore expected for some metrics, and are often the finding: for example, the first force peak per body weight rises with load. Appendix B lists the direction in which load is expected to move each metric. Divide by total weight (the `_tw` columns) to see whether the pattern changed beyond simply carrying more.

**Each range is marked "yes" (checked) or "verify".** "Yes" means the range was checked against the source's abstract or a review. "Verify" means it comes from the gait literature at large and should be checked against the full paper before it is quoted in a publication. (The publishers' websites could not be reached when this document was prepared, so the full texts could not be opened.) All ranges live in one place in the code, `REFERENCE_RANGES` in §19, so correcting one there corrects the workbook, and rebuilding this document corrects its tables.

## Notation

| Symbol | Meaning | Unit |
|:--|:--|:--|
| $t_{HS}, t_{TO}$ | time of a heel strike, of a toe-off | s |
| $L$, $R$ | left, right foot; "the other foot" is written $o$ | – |
| $\hat{f}$, $\hat{\ell}$ | unit vectors along the belt's direction of travel (forward), and across it (to the left) | – |
| $v_b$ | belt speed | m/s |
| $\mathbf{F}$ | ground reaction force on the walker | N |
| $m_b$, $m_{\ell}$, $m$ | body mass (from participants.csv), load mass, system mass $m=m_b+m_\ell$ | kg |
| $\mathbf{x}_{CoM}$, $\mathbf{v}_{CoM}$ | system centre of mass (body + load), and its velocity | m, m/s |
| $g$ | 9.81 m/s² | m/s² |
| BW, TW | body weight $m_b g$; total weight $m g$ | N |
| ML, AP, VT | medio-lateral (across the belt), antero-posterior (along it), vertical | – |
| $N$ | number of strides in a series | – |
| $\operatorname{sd}$, CV | standard deviation; coefficient of variation = 100 × sd / \|mean\| | –, % |

Units are metres, seconds, newtons, kilograms and degrees unless a column name says otherwise: `_mm`, `_pct` (%), `_bw` (per body weight), `_tw` (per total weight), `_j_kg` (joules per kg of system mass).
