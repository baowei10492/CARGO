"""Build the P32 detectors and run two forwards, before any hours-long training.

    python -m tools.p32_smoke --config configs/p21/bifa_levir.yaml --device cuda

Checks what only shows up on the server: that the checkout imports, that the
ImageNet weights land in BiFA's backbone (external_bifa.py item 4), that a
training-mode forward and a loss step run, and that an evaluation forward on a
tile larger than the training crop is stitched back to the right shape. Nothing
is written and no dataset is touched.
"""
from __future__ import annotations

import argparse

import torch

from rs_cd.losses import BinaryChangeLoss
from rs_cd.runtime import build_model, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.config}: {parameters / 1e6:.2f} M parameters, {trainable / 1e6:.2f} M trainable")
    if hasattr(model, "pretrained"):
        print(f"ImageNet weights verified: {model.pretrained}")

    shape = (args.batch_size, 3, args.train_size, args.train_size)
    t1, t2 = torch.rand(shape, device=device), torch.rand(shape, device=device)
    target = (torch.rand((args.batch_size, 1, args.train_size, args.train_size), device=device) > 0.9).float()
    loss_config = config.get("loss", {})
    criterion = BinaryChangeLoss(
        bce_weight=float(loss_config.get("bce_weight", 1.0)),
        dice_weight=float(loss_config.get("dice_weight", 1.0)),
        pos_weight=loss_config.get("pos_weight"),
    ).to(device)
    model.train()
    outputs = model(t1, t2, targets=target, batch=None)
    logits = outputs["change_logits"]
    if logits.shape != target.shape:
        raise SystemExit(f"training forward returned {tuple(logits.shape)}, expected {tuple(target.shape)}")
    loss, components = criterion(outputs, target)
    loss.backward()
    without_gradient = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    print(f"train forward ok, loss {float(loss):.4f} {components}, "
          f"{len(without_gradient)} parameters without a gradient")
    if without_gradient:
        print("  first without a gradient: " + ", ".join(without_gradient[:5]))

    model.eval()
    shape = (1, 3, args.eval_size, args.eval_size)
    with torch.no_grad():
        logits = model(torch.rand(shape, device=device), torch.rand(shape, device=device))["change_logits"]
    if tuple(logits.shape) != (1, 1, args.eval_size, args.eval_size):
        raise SystemExit(f"eval forward returned {tuple(logits.shape)}")
    print(f"eval forward ok at {args.eval_size} px, logits {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
