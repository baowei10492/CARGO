# P5c: global registration before multi-scale change decoding

## Motivation

The inverse-shift oracle recovers F1 to approximately 0.9025 under synthetic
T2 translation, so the information needed for detection remains available.
P5b showed that a one-channel residual after decoding is too late to correct
the geometric mismatch. P5c therefore estimates translation from the original
DINO descriptors and applies it before multi-scale differencing.

P5a and P5b are retained as controls:

- P5a: shift augmentation improves severe T2 shift but damages clean and
  unrelated corruptions.
- P5b: an output-level selective residual preserves clean but produces almost
  no shift robustness gain.
- P5c: explicit feature-level registration followed by multi-scale aligned
  differencing.

## Architecture

The P3b DINOv3 backbone and FPN decoder are loaded from the frozen P3b
checkpoint.

The registration module:

1. uses the original highest-resolution DINO feature channels;
2. learns only positive per-channel reliability weights;
3. computes global cosine scores for a 7x7 translation grid;
4. converts the scores to a differentiable displacement distribution;
5. estimates a subpixel expected displacement;
6. warps all four T2 feature scales into the T1 coordinate system;
7. mixes raw and aligned differences with one learned strength per scale.

The search radius is three level-0 feature cells, corresponding to up to 12
input pixels at stride four.

## Stage A: registration learning

Configuration: `configs/p5c_global_registration_stage_a.yaml`.

- P3b backbone and decoder: frozen.
- Alignment fusion: disabled.
- Trainable parameters: channel reliability and score temperature.
- Training supervision: known synthetic T2 translations of 2, 4, and 8
  pixels, plus 25% clean samples.
- Always use `last.pth` for the Stage A registration evaluation. Clean F1 is
  constant during this stage and is not a valid registration selector.

Run the deterministic registration evaluator on clean and all three shift
levels before Stage B.

Stage A is a go when:

- clean mean predicted displacement is no larger than 0.5 pixels;
- shift-4 and shift-8 mean EPE are each no larger than 1.5 pixels;
- shift-4 and shift-8 within-2-pixel rates are at least 0.85;
- no NaN/Inf appears and the score scale does not remain stuck while EPE is
  flat.

It is a conditional go when EPE improves substantially over the original
probe (2.88 pixels at shift-4 and 2.33 pixels at shift-8) but misses one of the
strict gates. Tune the registration stage only; do not start Stage B blindly.

### Stage A1 completed result

The 97-parameter channel-weighting and temperature model failed the fixed
registration gate:

| Shift | Mean soft EPE | Within 2 px |
|---|---:|---:|
| clean | 1.8135 | 0.5938 |
| 2 px | 2.3812 | 0.2969 |
| 4 px | 2.7982 | 0.2812 |
| 8 px | 4.5706 | 0.2188 |

Confidence stayed near 0.25 for all groups. A1 could neither identify clean
pairs nor interpret multi-modal score surfaces, and soft expectation degraded
the severe-shift result. It is retained as a negative registration ablation.

### Stage A2: correlation-surface decoding

Configuration: `configs/p5c_a2_surface_registration_stage_a.yaml`.

A2 is a method change rather than a learning-rate sweep. It standardizes each
7x7 correlation surface, applies a small 2-D surface decoder, adds an explicit
zero-offset adjustment for clean pairs, and reports both hard argmax and soft
subpixel displacement. The paired metrics distinguish candidate-selection
errors from averaging errors.

### A2 downstream alignment probe

Full-strength alignment established that registration is useful for the
downstream task even though strict EPE gates were missed:

| Variant | Clean F1 | T2 shift-2 | T2 shift-4 | T2 shift-8 |
|---|---:|---:|---:|---:|
| P3b | 0.902281 | 0.866611 | 0.805023 | 0.680037 |
| A2 soft/full | 0.883232 | 0.867079 | 0.856446 | 0.851372 |
| A2 hard/full | 0.887610 | 0.852576 | 0.833162 | 0.858458 |
| A2 soft/hard-nonzero gate | 0.892658 | 0.863671 | 0.838977 | 0.845920 |

Soft full alignment improves mean T2-shift F1 by 0.0744 and shift-8 F1 by
0.1713, but over-corrects clean pairs. The hard-nonzero gate recovers 0.0094
clean F1 while retaining a 0.1659 shift-8 gain. This is a conditional go for
Stage B.

### Frozen validation candidate

The frozen primary candidate uses Stage B epoch 2, soft displacement, and a
zero-displacement probability threshold of 0.20:

| Metric | P3b | P5c | Delta |
|---|---:|---:|---:|
| Clean F1 | 0.902281 | 0.901080 | -0.001201 |
| T2 shift-2 F1 | 0.866611 | 0.870351 | +0.003740 |
| T2 shift-4 F1 | 0.805023 | 0.827153 | +0.022130 |
| T2 shift-8 F1 | 0.680037 | 0.763289 | +0.083252 |
| Mean T2 shift F1 | 0.783890 | 0.820264 | +0.036374 |

Across the full 37-scenario diagnostic, mean F1 differs from P3b by -0.000392.
Mean brightness, blur, and noise changes are -0.001557, -0.003170, and
-0.002149. The method is therefore scoped to T2 geometric misregistration,
not claimed as a universal corruption improvement.

Checkpoint SHA256:
`be01ea4412d96f1f769b36a81d87c84edf18c048f1f5164f9ababc9f85593d77`.

The architecture, Stage B epoch (2), gate threshold (0.20), and evaluation
threshold (0.5) are frozen after this validation result. Subsequent runs vary
only the training seed. Test data must remain untouched until multi-seed and
cross-dataset validation are complete.

## Stage B: multi-scale fusion

Configuration: `configs/p5c_multiscale_alignment_stage_b.yaml`.

- Initialize from the selected Stage A checkpoint.
- Registration, P3b backbone, and P3b decoder: frozen.
- Trainable parameters: four scale-specific alignment strengths.
- Mixed clean/shift training is used to learn how much aligned difference each
  decoder level should receive.

For A2, use `configs/p5c_a2_gated_multiscale_stage_b.yaml`. It uses soft
subpixel displacement, the hard nonzero activation gate, and a smooth sigmoid
parameterization for four scale-specific fusion strengths. All epoch
checkpoints are retained because clean-only checkpoint selection cannot choose
the final clean/robustness trade-off.

Stage B first passes the clean gate:

- clean validation F1 at least 0.8973.

Then evaluate the fixed 37-scenario protocol. The primary robustness gates
remain:

- shift-8 T2 gain at least 0.05 over 0.680037;
- mean T2 shift-2/4/8 gain at least 0.03 over the P3b reference;
- no material regression in brightness, blur, or noise means.

## Interpretation boundaries

P5c assumes T1 is the reference coordinate system and estimates how T2 should
be sampled to match it. This matches the intended T2-misregistration
application and the primary T2 robustness endpoint. T1-shift and common-shift
scenarios remain diagnostics but are not interchangeable with the stated
reference-frame assumption.
