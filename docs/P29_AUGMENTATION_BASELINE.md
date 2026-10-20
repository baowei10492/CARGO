# P29: misalignment-augmentation baseline

Written before any P29 detector is trained. Nothing below changes after a P29
result has been seen.

## Question

A reviewer may ask whether the localizer and selector are needed at all: would
the detector alone, trained on the same kind of misalignment it meets at test
time, do as well? P29 trains that detector and runs it through the P27 test
harness, with and without the P27 localizer and selector.

An earlier pilot (P5a, main record sections 18-19; LEVIR-CD validation only,
integer shifts of 2/4/8 px, fine-tuned from the clean detector) lost 0.88 F1
points on clean images and gained 4.3 points at 8 px. It used a different
perturbation and split and is not reported as a result.

## Detectors

For each dataset, the DINOv3 detector of the P27 row is trained again from the
same initialization (DINOv3 ConvNeXt-T weights) with the same recipe:

| Dataset | Recipe of the P27 detector | P29 config |
|---|---|---|
| LEVIR-CD | `configs/p3b_dinov3_convnext_tiny_partial_ft.yaml` | `configs/p29/dinov3_aug_levir.yaml` |
| DSIFN-CD | `configs/p15_dsifn_p3b.yaml` | `configs/p29/dinov3_aug_dsifn.yaml` |
| WHU-CD | `configs/p8_whu_dinov3_partial_ft.yaml` | `configs/p29/dinov3_aug_whu.yaml` |

The only differences are the experiment name, the LEVIR-CD dataset root (the
path the P27 evaluation reads; same data) and the augmentation:

- in 75% of the training crops T2 is perturbed exactly like the P27 test levels
  (`configs/p27_eval_perturbation.yaml`): level 1, 2 or 3 uniformly, the same
  scale and translation construction about the crop centre, no rotation,
  bilinear, zero outside the image; the other 25% stay unperturbed (the test has
  one clean level and three perturbed levels);
- the augmentation follows the crop, flips and 90-degree rotations, and T1 and
  the label are not moved (the P27 test scores the label in the T1 frame);
- pixels without T2 content are left out of both loss terms.

The checkpoint is chosen as for every other detector of the paper: the best
clean validation F1 (`best.pth`). A run whose best validation F1 is below 0.5 is
reported and not used.

This is the augmentation most favourable to the baseline: it matches the test
distribution exactly, which a deployed detector could not know.

## Evaluation

The three augmented detectors form the P27 plug-in row `dinov3aug`
(`configs/p27_plugins/dinov3aug_{f1,selector,test}.yaml`, rules as
`configs/p27_f1_gate.yaml` and `configs/p27_selector.yaml`, markers under
`reports/p27_localizer/plugins/dinov3aug/`). The row's test run gives, on the
P27 test crops:

- `raw`: the augmented detector alone, the baseline of the main comparison table;
- `selector`: the augmented detector with the frozen P27 localizer and a selector
  trained for it, as for every other detector.

The test split is used once for this row. The frozen P27 localizer is not
retrained.

## Reporting

Main comparison table: "DINOv3 detector + misalignment augmentation" among the
methods that handle misalignment. Generality table: the augmented detector
without and with the localizer and selector. Both whatever the outcome.

Machine: the teammate's server (same project layout). Code, configs and recipes
are not changed there; a failed or diverging run is reported, not re-tuned.
