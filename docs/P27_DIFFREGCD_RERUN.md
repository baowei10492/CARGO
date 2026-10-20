# DiffRegCD re-run with the correct wiring (2026-09-19)

> Written before any run of `tools/p27_diffregcd_rerun.py`; not changed after seeing results.
> Replaces the DiffRegCD row of the P27 main comparison (invalid, main record 192.47) and adds DiffRegCD
> to the border-free (P30) comparison.

## 0. Why (must be disclosed)

```text
The P27 run fed A = unperturbed T1, B = perturbed T2 (convention taken from the registration head, P26-A).
The generator check (docs/P27_DIFFREGCD_GENERATOR_CHECK.md, record 192.47) showed that the released change
head needs the perturbed image in A and the unperturbed reference in B (change map in B's frame):
LEVIR-CD F1 with the authors' generator, paper ranges: A perturbed 86.5, B perturbed 18.5 (clean 86.7), and the
A-perturbed numbers match the paper (LEVIR 87.2, WHU 87.4, DSIFN 90.0). The P27 DiffRegCD row and the
resampling diagnosis of record 192.20 used the wrong slot and are withdrawn.
```

## 1. Wiring (all parts)

```text
A = the perturbed T1 (the moving image), B = the unperturbed T2 (the reference), label in the original frame.
The same matrix M the other methods see: T2 is perturbed by M for them, T1 is perturbed by M for DiffRegCD;
the misalignment between the dates has the same size. Temporal order is kept (A = earlier date).
Released test path (official_unmodified) = the reported row; training-time wiring (intended_warped) descriptive.
Weights, config, split (test), diffusion seeds and threshold: configs/p27_diffregcd.yaml, unchanged.
```

## 2. Parts

```text
bordered     P27 protocol: all P27 test crops (256 px grid), levels 0-3, P27 per-crop draws, the P27 perturbation
             (per-crop warp, zero border) applied to T1, scored on the P27 scoring pixels (and all pixels,
             descriptive). Replaces the P27 main-table row.
border_free  P30 protocol: all margin crops (64 px to the tile edge), levels 0-3, same draws; A = the crop of the
             whole T1 tile warped by M (rs_cd.p30_geometry, no padding anywhere), scored on the P27 scoring
             pixels. Plus, on the same crops, the level-3 per-crop warp with zero border (bordered_l3), so that
             border dependence is measured on identical crops and draws. The DiffRegCD row of the P30 table.
mechanism    test crops evenly spread (LEVIR-CD 600, DSIFN-CD 192, WHU-CD 600), level-3 draw, B = T2 always:
               normal       A = perturbed T1
               border_only  A = T1 unmoved, with zeros exactly where the perturbed T1 has no content
               same_date    A = perturbed T2 (no real change between A and B)
               unrelated    A = perturbed T1 of a crop from another tile
               black        A = zeros
             Scored on the level-3 scoring pixels.
```

## 3. Reading (fixed before the run)

```text
Rows (bordered, border_free): reported as they are, official path, no verdict.
Border dependence, border_free part, official path, level 3, mean over the three datasets of
  F1(bordered_l3) − F1(border-free level 3):
    >= 5 points → border_dependent;  <= 2 points → border_independent;  otherwise partial
Use of the earlier date, mechanism part, official path, LEVIR-CD:
  ratio = F1(same_date) / F1(normal)
    >= 0.8 → ignores_dates (predicts change without a temporal difference);  <= 0.3 → uses_dates;  otherwise partial
border_only, unrelated and black are descriptive (does the border alone, or any image in A, switch the head on).
```

## 4. Cost

About 0.34 s per DiffRegCD call (P27 run). mechanism 1392 crops × 5 ≈ 40 min; border_free 4792 × 5 ≈ 2.3 h;
bordered 4792 × 4 ≈ 1.8 h. Parts are resumable per dataset.
