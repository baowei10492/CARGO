# P33: experiments requested by the two pre-submission reviews (2026-10-08)

Written before any P33 result existed. Protocol unchanged from P30/P31: border-free test crops, per-crop draws of
levels 1-3, scoring pixels, hold-out tiles and the selector recipe of `configs/p27_selector.yaml`. Every run that
re-evaluates a frozen setting must reproduce the frozen numbers first (anchors in the tools).

## Experiments, in order of importance

1. **Registrar + the paper's selector** (`tools/p33_selector_runs.py pipeline --localizer roma_ts_ransac`, queue A;
   `phasecorr_fm`, queue C). The registration module is replaced by RoMa (3 DoF, RANSAC 3 px, Table 7's fair
   configuration) or by Fourier-Mellin phase correlation; the selector is trained with the same recipe and the same
   19 features computed from that registrar (its misalignment flag is the displacement > 4 px rule of
   `SimilarityBaseline`). Twelve combinations of Table 6.
2. **Retrained registration modules** (`configs/p33/localizer_conf_only.yaml`, `localizer_no_change_label.yaml`,
   queue B): without change-aware weights (confidence only, no change supervision) and with the weights kept but no
   change labels. Always-registering rows (Table 7), per-crop errors by changed area (Figure 4a), and the selector
   recipe on DINOv3 and BIT.
3. **S2Looking** (queue B): never used to train the registration module. DINOv3 detector with the paper's recipe;
   selector recipe; test split levels 0-3. Level 0 are the original pairs with their own registration offsets.
4. **Interpolation control** (`interp`, queue C; `tools/p33_diffregcd_interp.py`, queue B): at level 0 the raw input
   is not resampled, at levels 1-3 it is. The control resamples T2 (for DiffRegCD: T1) by a per-crop sub-pixel
   transform (|t| <= 0.5 px, |s - 1| <= 0.5 %, `rs_cd/p33_protocol.py`), scored on the paper's pixels.
5. **Classical global registration** in the 3-DoF family: Fourier-Mellin phase correlation (Reddy and Chatterji,
   1996) and pyramid ECC (Evangelidis and Psarakis, 2008), Table 7 rows and Figure 4 errors (queue C).
6. **Analyses** (queue C): CARGO minus always correcting with tile-bootstrap intervals per cell and for the 18-cell
   mean (tiles drawn once per dataset); the WHU-CD mixed test with intervals; the registration module on identical
   pairs and on known level-1 transforms of T1 (signed errors); RoMa on identical pairs (subset).

## Reading rules (fixed now)

- **1.** Compare the twelve-combination means with the full selector of Table 6 (+11.1 at level 3, 0.0 at level 0,
  2 degraded). If RoMa + selector reaches at least the level-3 gain minus 0.5 pp with no more degraded combinations,
  the paper states that the selector carries the accuracy and that the registration module's advantage is cost
  (timing of both); otherwise it reports the registration module's advantage with the numbers. Phase correlation is
  reported the same way.
- **2.** If the confidence-only module comes within 1 pp of the full module's mean level-3 gain with the selector and
  is not worse at level 0, "change-aware" is reworded as learned reliability weighting supervised with change labels,
  and the contribution is narrowed. The no-change-label variant tells whether the supervision itself matters.
  *Amended 2026-10-09 (authors' decision, before any result):* if the full module's advantage is small, the
  retrained variants are not reported in the paper and the wording stays; Section 4.7.2 keeps stating that its rows
  are inference-time switches without retraining, and no sentence is added that credits change supervision. A clear
  advantage is reported as a result; if the confidence-only module is clearly better, the authors decide with the
  numbers. A reviewer who asks for the retrained ablation gets the numbers in the response.
- **3.** S2Looking is reported as measured: level-3 gain with interval, level-0 change, adoption rate at level 0.
  Level 0 on S2Looking is a real-misregistration reading only in the sense above; its labels mix both dates
  (label1/label2), so no claim of real-data correction is made from it.
- **4.** If in any combination the adoption rate under the control differs from level 0 by more than 5 points or the
  level-0 change by more than 0.3 pp, the paper reports it; otherwise one sentence. For DiffRegCD, if the control
  recovers at least half of its DSIFN-CD gap between level 0 and level 3, the anomaly of Table 5 is explained in one
  sentence with the numbers.
- **5.** Classical baselines join Table 7 and Figure 4 as rows; no claim changes unless they beat the registration
  module on both success rate and gross errors.
- **6.** Intervals are added where the text compares CARGO with always correcting; a comparison whose interval
  includes zero is worded as such.
