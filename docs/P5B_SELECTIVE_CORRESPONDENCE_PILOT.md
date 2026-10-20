# P5b selective-correspondence pilot

## Purpose

This pilot tests one narrow hypothesis established by the P5b diagnostics:

- a one-sided T2 translation is almost fully recoverable with an oracle inverse
  shift;
- the highest-resolution DINOv3 feature map contains useful local
  correspondence;
- therefore a learned local correspondence residual may recover T2-shift
  robustness without replacing the clean DINOv3 detector.

This is a go/no-go experiment, not yet the final paper model.

## Model

The frozen P3b DINOv3 model remains the raw prediction path. A new
time-symmetric branch:

1. projects the highest-resolution T1/T2 features with shared weights;
2. builds bidirectional local cost volumes over a radius of three feature
   cells (up to 12 image pixels at stride four);
3. estimates correspondence confidence, a null/unmatchable probability, and
   a coarse change probability;
4. applies alignment only where correspondence is confident and change is
   unlikely;
5. predicts a zero-initialized residual correction to the raw change logits.

The base backbone and decoder are frozen during this pilot. The initial model
therefore reproduces P3b exactly before the new branch learns.

## Required initialization

Set `training.initial_checkpoint` in
`configs/p5b_dinov3_selective_corr_pilot.yaml` to the absolute path of the P3b
best checkpoint. `initial_checkpoint_strict: false` is intentional because the
checkpoint predates the correspondence branch.

## First-run protocol

1. Run all unit tests.
2. Run GPU smoke tests with batches 16, 32, and optionally 64. The smoke path
   includes synthetic shift metadata and all auxiliary losses.
3. Select the largest batch with a reasonable memory margin. Do not infer the
   training batch from the earlier P3b smoke test because the local cost volume
   has a different compute profile.
4. Run the 15-epoch pilot.
5. Evaluate the best checkpoint on the clean validation set and the fixed
   37-scenario robustness protocol.

## Decision criteria

Compare against the clean P3b reference:

- clean validation F1: `0.902281`;
- T2 shift F1 at 2/4/8 pixels: `0.866611 / 0.805023 / 0.680037`.

The pilot is a clear **go** when:

- clean F1 is at least `0.8973` (drop no larger than 0.005);
- T2 shift-8 F1 improves by at least `0.05`;
- the mean of T2 shift-2/4/8 improves by at least `0.03`;
- brightness, blur, and noise means do not materially regress.

It is a **conditional go** when shift robustness improves strongly but the
clean drop is between 0.005 and 0.010. In that case tune residual gating and
the clean/shift sampling ratio.

It is a **no-go** when the shift-8 gain is below 0.02 or when clean F1 drops by
more than 0.010. Do not scale training before inspecting offset accuracy,
confidence, and null/change masks.

## Important interpretation

The scientific claim is not “shift augmentation improves robustness.” P5a
already rejected that claim. The tested mechanism is that selective
correspondence on unchanged regions can suppress misregistration-induced
false changes while preserving true building changes.

## Completed result (seed 3407)

P5b preserved clean performance but did not deliver a meaningful robustness
gain:

| Checkpoint | Clean F1 | T2 shift-2 | T2 shift-4 | T2 shift-8 |
|---|---:|---:|---:|---:|
| P3b reference | 0.902281 | 0.866611 | 0.805023 | 0.680037 |
| P5b best (epoch 1) | 0.902199 | 0.866627 | 0.805200 | 0.680397 |
| P5b last (epoch 15) | 0.900213 | 0.865940 | 0.806917 | 0.684048 |

The last checkpoint improved mean T2-shift F1 by only 0.001745 while clean F1
dropped by 0.002068. Offset KL and offset regression losses remained nearly
flat. The final clean prediction shifted toward higher recall and lower
precision, indicating a generic positive-region expansion rather than
successful geometric correction.

Decision: retain P5b as a negative control and do not extend its training.
P5c moves registration before the decoder and aligns all DINO feature scales.
