# P30 no-padding check (validation only)

Written after the P29b diagnosis and the P30 pre-check, and before the runs
below.

## Why

In both earlier runs the border-free T2 was cut from the whole tile warped with
the level's matrix; where the warp needed pixels beyond the tile it used
reflection (BORDER_REFLECT_101). Most P27 crops touch a tile edge (LEVIR-CD 12
of 16 per tile, DSIFN-CD and WHU-CD all), so mirrored content entered the
border-free inputs. Mirrored texture might make matching harder or mislead a
detector, independently of the zero-border cue.

## What changes

Both tools run again with `--margin 64`: every crop keeps 64 real pixels to each
tile edge (LEVIR-CD: 16 crops per 1024 px tile at origins 64 / 277 / 491 / 704;
DSIFN-CD and WHU-CD: 4 crops per 512 px tile at 64 / 192, overlapping). The
largest level-3 perturbation (scale 0.80, translation 25 px) samples at most
63.25 px beyond the crop, so no padding is ever needed; the tools stop if a
crop and level would need it. Everything else is unchanged; the crop ids differ,
so the per-crop draws differ from P27's.

## Reading, fixed now

- Diagnosis: the P29b rule, unchanged. The `border_oracle` drop (aligned content,
  zero border outside the scored pixels) is also reported for both detectors; it
  involves no border-free construction at all.
- Pre-check: the P30 pre-check rule, unchanged (level-3 localization success,
  mean over the three datasets, bordered minus border-free <= 5 points: keep the
  localizer frozen; otherwise retrain it on border-free pairs). **This no-padding
  result decides**; the reflected run stays on record as a first look.
