# P30 pre-check: the frozen P27 localizer without the zero border (validation only)

Written after the P29b border diagnosis (main record 192.33) and before this
check was run. It informs whether a border-free comparison (P30) needs the
localizer retrained; it is not a result of the paper.

## Setting

P27 validation crops of LEVIR-CD, DSIFN-CD and WHU-CD, levels 1-3 with the P27
per-crop matrices. Two ways to make the misaligned T2:

- `bordered`: the P27 perturbation (the 256 px crop warped, zero outside);
- `border_free`: the same matrix applied to the whole tile (reflected at the
  tile edge) and the crop cut out, as the `context_raw` view of P29b.

For each, the frozen P27 localizer (configs/p27_selector.yaml) estimates the
transform with the P27 step-1 routine (success: mean transfer error within 2 px,
raw or relative to the clean estimate), and the original DINOv3 detector of the
dataset scores the raw, oracle, always-corrected and gated paths on the P27
scoring pixels. No selector (it would need retraining on border-free crops).

## Rule, fixed now

Let s(mode) be the level-3 localization success, averaged over the three datasets.

- The localizer is taken **not to depend on the border** if s(bordered) - s(border_free) <= 5 points;
  the P30 comparison then keeps it frozen.
- Otherwise it is retrained on border-free pairs before P30.

The F1 of the corrected paths is reported for both modes as information.
