# P30: border-free comparison with the augmentation baseline

Written before any P30 run. Only the localizer line depends on a result not yet
seen (the no-padding check, docs/P30_NOPAD_CHECK.md), and that dependence is
fixed here. Nothing else changes after a P30 result has been seen.

## Why

The P27 protocol (as DiffRegCD's released data generator) warps each 256 px
crop with a zero border whose shape encodes the transform. The P29 augmentation
baseline reads that border (P29b). P30 compares the methods under a
misalignment without any border, each method trained on the same kind of data
it is tested on. The P27 results stay the main results (comparable with
DiffRegCD); P30 is reported next to them whatever the outcome.

## Evaluation protocol (`configs/p30_eval_border_free.yaml`)

The P27 evaluation (`configs/p27_eval_perturbation.yaml`) with three changes:

1. **Crops**: validation and test crops keep 64 real pixels to every tile edge
   (LEVIR-CD 16 per 1024 px tile at origins 64 / 277 / 491 / 704; DSIFN-CD and
   WHU-CD 4 per 512 px tile at 64 / 192). The largest level-3 perturbation
   samples 63.25 px beyond the crop, so no padding is ever needed; runs stop if
   one would be.
2. **Perturbation**: the level's matrix is applied to the whole T2 tile about the
   crop centre and the crop is cut out, so the misaligned T2 shows real content up
   to its edges. Levels, ranges, per-crop draws (keyed by crop id) and seed are
   those of P27.
3. **Correction**: a correction (true matrix or estimate) is applied to the
   perturbed tile and the crop is cut out again, so the corrected T2 has no border
   either. With an estimate that reaches beyond the tile, the missing pixels are
   zero and are reported as a count.

Scoring pixels, label frame, thresholds and the localization success rule are
those of P27.

## Components

| Component | P30 |
|---|---|
| Detectors (DINOv3, P27 row) | unchanged, frozen |
| Localizer | if the no-padding check reads `no_border_dependence`: the frozen P27 localizer. If it reads `retrain_localizer`: **v2**, trained with `configs/p27_localizer_v1.yaml` unchanged except that every target is cut from the warped tile (training crops keep 96 px to the tile edge; the training perturbation reaches 85.3 px) and the hold-out evaluation used for checkpoint selection is border-free (64 px margin). Same steps, seed and selection rule. |
| Selector | retrained per dataset with the P27 selector recipe (`configs/p27_selector.yaml` unchanged), on border-free training crops with tile-level correction; calibration rule unchanged |
| Augmentation baseline | the P29 recipe (`configs/p29/dinov3_aug_*.yaml`) with the perturbed samples cut from the warped tile (crops keep 64 px to the edge; no pixel lacks content, so no loss masking); best clean validation epoch, as P29 |

## Rows (validation first, then the test split once)

1. DINOv3 detector, no correction.
2. P29 augmented detector (trained with the zero border), no correction.
3. P30 augmented detector (trained border-free), no correction.
4. **Proposed method under P30**: DINOv3 detector + P30 localizer + P30 selector.
5. For reference: DINOv3 detector + P27 localizer + P27 selector (all frozen).

Paths as in P27 (raw, oracle, estimated, gated, selector). Test markers are
under `reports/p27_localizer/plugins/p30*`; each component set meets the test
split once.

## Reporting

Rows 1-5 on validation and test, in the paper next to the P27 table, with this
protocol and the P29b diagnosis. No row is dropped and no setting is re-tuned
after results are seen. DiffRegCD and the three open-source detectors are not in
this first P30 batch.
