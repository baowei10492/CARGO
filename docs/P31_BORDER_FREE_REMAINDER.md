# P31: the plug-in rows and the localizer ablation under the border-free protocol (2026-09-19)

> Written before any P31 run; not changed after seeing results. Runs before the advisor has chosen the main
> protocol because the GPUs are free; if the per-crop protocol stays the main one, these results form the
> border-free section instead.

## 0. Why

```text
P30 (test, record 192.45) measured only the DINOv3 row without a zero border. The same frozen system
(localizer v1 + P27 selector) gains at level 3: per-crop zero border +31.8 / +5.3 / +19.9, border-free
+14.9 / +1.2 / +10.1 (LEVIR / DSIFN / WHU): part of the P27 gains came from the border. The 12-cell generality
table and the localizer ablation are on the per-crop protocol; P31 repeats them border-free.
```

## 1. Plug-in rows

```text
Protocol: P30 exactly (configs/p30_border_free.yaml): crops keep 64 px to the tile edge, perturbation and
  correction act on the whole tile, P27 levels, per-crop draws and scoring pixels, no padding anywhere.
Localizer: v2, the P30 localizer, frozen. Selector: configs/p27_selector.yaml, retrained per detector on that
  detector's border-free training records (the P30 training-crop rule).
Detectors: the frozen P27 plug-in detectors: BIT, CD-Lamba, CDxLSTM; LEVIR-CD and DSIFN-CD from
  configs/p27_plugins/<row>_f1.yaml, WHU-CD from configs/p27_plugins/<row>_whu_f1.yaml.
Per row: features -> train -> validation -> test. Validation and test are both run and both reported;
  nothing is chosen on validation; the test runs once per row (markers keyed by the components).
Reported: without correction vs with localizer + selector, F1 at clean and levels 1-3; localization success;
  selection rates. The DINOv3 row is the P30 test run.
```

## 2. Localizer ablation

```text
DINOv3 detectors (configs/p27_f1_gate.yaml), test split, P30 protocol, reporting only (nothing is selected).
Variants: v2_full, v2_coarse_only, v2_one_fine_step, v2_no_reliability (inference-time switches of the frozen
  v2 localizer, as in P27), v1_full (the P27 localizer), roma_similarity, sift_similarity (P27 settings).
Paths: raw, oracle, estimated (every pair corrected), estimated_gated; localization success per level.
```

## 3. Reading

```text
Tables only; no verdicts. The raw path of the ablation must reproduce the P30 test raw row of the DINOv3
detector (same crops, draws and scoring pixels); a difference above 0.001 F1 stops the run as an instrument error.
```
