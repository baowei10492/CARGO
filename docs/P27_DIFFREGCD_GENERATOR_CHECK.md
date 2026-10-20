# P27 check: DiffRegCD on its own perturbation generator (2026-09-19)

> Written before any run of `tools/p27_diffregcd_generator_check.py`; not changed after seeing results.
> Diagnosis for the main comparison table; not a new comparison row and not a gate for the proposed method.

## 0. Question

In the P27 test run the released DiffRegCD falls far below the detector that ignores misalignment
(LEVIR-CD level 3: 25.0 against 44.9). Is that a property of the released weights and test path, or an
artefact of how this project perturbs and feeds the pairs?

Facts from the released code (commit 6b9acb6c) that motivate the check:

- Training (`model/cd_model_256.py`, lines 178 and 215): the change head only ever sees image B's features
  warped by the predicted flow, `netCD(feats_A, warp(feats_B, flow))`.
- Released test path (`model/cd_model_256.py`, line 483; `main.py -p test`): `netCD(feats_A, feats_B)`,
  no warp; the flow is computed only for its loss.
- Their perturbation generator (`scripts/prepare_gt_flow.py`, lines 64-73): per 256 px patch,
  angle ~ U(-a, a), scale ~ U(s0, s1), tx, ty ~ U(-t, t) drawn in that order,
  `M = getRotationMatrix2D(centre, angle, scale)`, `M[:, 2] += [tx, ty]`,
  `cv2.warpAffine(img, M, (w, h), flags=INTER_LINEAR)` (constant zero border); the label is not warped.
- The README names the warped view image A (`t1`) and the reference image B (`t0`); the training code aligns B
  to A (A is the reference). This project fed A = reference, B = perturbed (P26-A, main record 191.3).

## 1. Design

```text
Crops: P27 test crops (256 px grid), evenly spread: LEVIR-CD 600, DSIFN-CD all 192, WHU-CD 600
Weights, config, diffusion seeds: as the P27 DiffRegCD run (configs/p27_diffregcd.yaml)
Draws: their generator, exactly (same formula, same draw order, same cv2 call), one RandomState per crop
  and range set, seeded from the crop id; the same matrix is used by both input orders
Range sets:
  paper          angle ±30°, scale 0.80-1.25, translation ±25 px   (paper, Table 1)
  paper_norot    angle 0,    scale 0.80-1.25, translation ±25 px   (the P27 level-3 ranges)
  repo_default   angle ±20°, scale 0.90-1.15, translation ±16 px   (script defaults)
Input orders:
  b_warped   A = T1, B = warp(T2)   (training-code convention; the P27 run)       all three range sets
  a_warped   A = warp(T1), B = T2   (README convention: A is the warped view)     paper, paper_norot
Plus clean: A = T1, B = T2
Wirings: official_unmodified (released test path), intended_warped (training-time feature warp)
Scoring against the unwarped label, as their generator: all pixels (their protocol) and content pixels
  (where the warped image has content); pooled F1 of the change class, threshold 0.5
```

## 2. Reading (fixed before the run)

```text
drop = F1(clean) − F1(paper), official_unmodified, all pixels, LEVIR-CD, for each input order
  released_robust       drop <= 5 points in either order   → the P27 DiffRegCD row is suspect: stop,
                                                              re-examine the P27 feeding before the row is used
  released_not_robust   drop >= 20 points in both orders   → the collapse belongs to the released weights and
                                                              test path; the P27 row stands, with this check cited
  intermediate          otherwise                          → report only; decide with the user
DSIFN-CD and WHU-CD are reported the same way, without a verdict. The intended_warped wiring and the
content-pixel scoring are descriptive.
Sanity: clean LEVIR-CD F1 (official, all pixels) >= 0.85 on these crops, else the adapter is broken → stop.
```

## 3. Cost

Six DiffRegCD calls per crop (clean + five perturbed views), about 0.34 s each on the RTX 4090 (P27 run):
about 50 minutes for 1392 crops.
