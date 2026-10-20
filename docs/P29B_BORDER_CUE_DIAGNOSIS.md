# P29b: does the augmented detector read the zero border? (diagnosis)

Written after the P29 F1 step on validation (main record 192.31) and before any
of the runs below. Diagnosis only: it decides what to test next, not a result of
the paper.

## Observation that motivates it

On the P27 validation crops, the DINOv3 detectors retrained with test-matched
misalignment augmentation (P29) score well on the misaligned pairs (raw path,
level 3: LEVIR-CD 78.8, WHU-CD 83.3) but far worse once the same pairs are put
back with the true matrix (oracle path: 60.2 and 70.6), although the oracle pair
is aligned. The original detectors lose about one point on the oracle path.

## Hypothesis

In the P27 protocol (as in DiffRegCD's) the perturbed T2 is a warped 256 px crop
with zeros outside the image. The shape of that zero border encodes the
translation and scale. A detector trained end to end on such pairs can read the
border and shift its prediction accordingly; given an aligned pair that still
carries a border, it shifts the prediction wrongly. Real misaligned pairs are
cut from larger images and carry no such border.

## Views (validation crops of the three datasets, levels 1-3, T1 and label unchanged)

| View | T2 given to the detector |
|---|---|
| clean | T2 |
| raw | T2 warped with the level's matrix, zero outside (the P27 raw path) |
| border_raw | T2 not moved, zeroed where `raw` is zero: the border cue without misalignment |
| oracle | `raw` put back with the true matrix (the P27 oracle path) |
| border_oracle | T2 not moved, zeroed where `oracle` is zero |
| context_raw | the whole tile warped with the level's matrix (reflected at the tile edge) and the crop cut out: the same misalignment without a zero border |

Detectors: the original DINOv3 detector of each dataset (configs/p27_f1_gate.yaml)
and the P29 augmented detector (configs/p27_plugins/dinov3aug_f1.yaml). Scored
on the P27 scoring pixels, as every P27 result.

## Reading, fixed now

Let d(view) be the F1 drop of a view from `clean` at level 3, in points, averaged
over the three datasets.

- The border cue is **confirmed** if, for the augmented detector,
  d(border_raw) >= 5 and d(context_raw) - d(raw) >= 5, while for the original
  detector d(border_raw) <= 2.
- It is **ruled out** if, for the augmented detector, d(border_raw) <= 2 and
  d(context_raw) - d(raw) <= 2.
- Anything else is **partial** and is reported as such.

If confirmed, the next step is a border-free protocol (context-style
perturbation) under which both the augmentation baseline (retrained with
border-free augmentation) and the P27 method are compared; that protocol is
written before it is run. The P27 and P29 results under the original protocol
stay as they are and are reported.
