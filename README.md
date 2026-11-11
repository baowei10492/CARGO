# CARGO

Change-aware registration and benefit-based output selection for frozen
remote-sensing change detectors.

This is the **private code archive for the accepted Remote Sensing article** at
https://github.com/baowei10492/CARGO. Uploading this repository does not publish
the manuscript or the patent material. A public release and software license
have not been enabled.

## Method

CARGO estimates translation and isotropic scale using frozen DINOv3 ConvNeXt-Tiny
features, trainable adapters, change-aware correspondence weighting, robust
fitting and local refinement. A separately trained selector predicts whether
to use the raw or registered output of the same frozen change detector.
The default selector uses 19 features and a 19-32-16-1 MLP; the six-registration-
feature variant is also retained in the experiment code.

The earlier/later time order of the detector inputs is preserved when changing
which image is the geometric reference. Final registered detector crops use
surrounding tile content, as documented in the border-free protocol.

## Contents

- `src/rs_cd/`: models, training losses, geometry, selection and metrics.
- `tools/`: original training, evaluation, analysis and dataset preparation tools.
- `configs/`: original configurations, including historical development settings.
- `scripts/`: original experiment launch scripts.
- `tests/`: existing tests; no experiments were rerun for this upload.
- `manifests/`: the available dataset/synthetic split manifests.
- `docs/`: original technical protocols and implementation notes.
- `FILE_MANIFEST.json`: SHA256 and provenance for every copied source file.

Main implementation files:

- `src/rs_cd/models/change_aware_similarity_localizer.py`
- `src/rs_cd/similarity_geometry.py`
- `src/rs_cd/localizer_training.py`
- `src/rs_cd/p27_selector.py`
- `src/rs_cd/p30_geometry.py`
- `src/rs_cd/dual_path.py`

## Environment

Use Python 3.10 or newer and install PyTorch suitable for the target machine.
Install the project and geometry/test dependencies with:

```sh
python -m pip install -e ".[dev,geometry]"
```

The frozen DINOv3 features require the pretrained checkpoint described by the
configuration. Third-party detectors and matching baselines can require their
own repositories, dependencies and checkpoints. They are not vendored here.

## Main experiment entry points

The border-free registration-module configuration is
`configs/p30_localizer_v2.yaml`; `tools.train_similarity_localizer` is its
training entry point. `tools.run_p30` implements selector feature generation,
training/calibration and evaluation. `configs/p31_border_free.yaml` extends the
protocol to the additional detectors; P32/P33 scripts retain subsequent
experiments and ablations.

The repository preserves original source bytes and configuration hashes.
Configurations and shell scripts contain the original research-directory
paths and checkpoint pointers. Adapt those paths to the intended environment;
hash-bound configurations should be updated consistently rather than silently
replaced. Historical P27 protocols used crop-level padding; the P30/P31
border-free protocols are the later setting. Do not mix historical stages when
interpreting results.

## Data, checkpoints and scope

Raw dataset images, large model checkpoints, generated predictions, experiment
archives, manuscript sources, patent drafts and account credentials are not
included. The five dataset manifests under
`manifests/datasets/` match Supplementary Data S1. See
[录用版实验入口与运行说明](docs/ACCEPTED_CODE.md) for the revision experiments,
their archived inputs, and the reproduction environment. Re-running trained
models requires access to the corresponding images and model assets.
No public software license has
been chosen; dataset and third-party licenses continue to apply.
