from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_SEVERITY_VALUES: dict[str, tuple[float, float, float]] = {
    "shift": (2.0, 4.0, 8.0),
    # Sub-pixel translations. The learned localizer decodes onto a 2 px
    # descriptor grid, so 0.5 and 1.0 px are below its representable
    # resolution by construction -- that is the point of measuring them.
    "subpixel_shift": (0.5, 1.0, 3.0),
    # Rotation in degrees. No single translation can correct a rotation,
    # so this perturbation lies outside the method's stated applicability
    # condition ("residual misalignment is approximately a global
    # translation"). At 1024 px a rotation of t degrees displaces the
    # corners by about 12.6 * t pixels: 3.2 / 6.3 / 12.6 px here, which
    # brackets the 2/4/8 px translation range.
    "rotate": (0.25, 0.5, 1.0),
    "brightness": (0.10, 0.20, 0.30),
    "gaussian_blur": (0.8, 1.6, 2.4),
    "gaussian_noise": (0.01, 0.03, 0.05),
}


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    severity: int = 0
    mode: str = "common"
    seed: int = 3407

    def __post_init__(self) -> None:
        if self.name == "clean":
            if self.severity != 0:
                raise ValueError("Clean corruption must use severity 0")
        elif self.name not in _SEVERITY_VALUES:
            raise ValueError(f"Unsupported corruption '{self.name}'")
        elif self.severity not in {1, 2, 3}:
            raise ValueError("Corruption severity must be 1, 2, or 3")
        if self.mode not in {"common", "t1", "t2", "independent"}:
            raise ValueError(
                "Corruption mode must be common, t1, t2, or independent"
            )

    @property
    def value(self) -> float:
        if self.name == "clean":
            return 0.0
        return _SEVERITY_VALUES[self.name][self.severity - 1]

    @property
    def key(self) -> str:
        return f"{self.name}:s{self.severity}:{self.mode}"


def _stable_seed(
    base_seed: int,
    sample_id: str,
    corruption: str,
    severity: int,
    branch: str,
) -> int:
    text = (
        f"{base_seed}|{sample_id}|{corruption}|{severity}|{branch}"
    ).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _to_unit(image: torch.Tensor, normalization: str) -> torch.Tensor:
    if normalization == "imagenet":
        mean = _IMAGENET_MEAN.to(image)
        std = _IMAGENET_STD.to(image)
        return (image * std + mean).clamp(0.0, 1.0)
    if normalization == "unit":
        return image.clamp(0.0, 1.0)
    if normalization == "none":
        return (image / 255.0).clamp(0.0, 1.0)
    raise ValueError(f"Unsupported normalization '{normalization}'")


def _from_unit(image: torch.Tensor, normalization: str) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    if normalization == "imagenet":
        mean = _IMAGENET_MEAN.to(image)
        std = _IMAGENET_STD.to(image)
        return (image - mean) / std
    if normalization == "unit":
        return image
    if normalization == "none":
        return image * 255.0
    raise ValueError(f"Unsupported normalization '{normalization}'")


def _translation_direction(seed: int, pixels: int) -> tuple[int, int]:
    directions = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )
    direction_x, direction_y = directions[seed % len(directions)]
    return direction_x * pixels, direction_y * pixels


def _unit_direction(seed: int) -> tuple[float, float]:
    """Unit-length version of the eight integer directions.

    Sub-pixel severities are magnitudes, so the direction must not also
    scale them: the diagonal integer directions have length sqrt(2).
    """

    directions = (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.7071067811865476, 0.7071067811865476),
        (0.7071067811865476, -0.7071067811865476),
        (-0.7071067811865476, 0.7071067811865476),
        (-0.7071067811865476, -0.7071067811865476),
    )
    return directions[seed % len(directions)]


def _translate_reflect(
    tensor: torch.Tensor, dx: int, dy: int
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = tensor.shape[-2:]
    pad = max(abs(dx), abs(dy))
    if pad == 0:
        return tensor.clone(), torch.ones(
            1, height, width, dtype=torch.float32
        )
    padded = F.pad(
        tensor.unsqueeze(0),
        (pad, pad, pad, pad),
        mode="reflect",
    ).squeeze(0)
    start_y = pad - dy
    start_x = pad - dx
    output = padded[
        :,
        start_y : start_y + height,
        start_x : start_x + width,
    ]
    valid = torch.ones(1, height, width, dtype=torch.float32)
    if dx > 0:
        valid[:, :, :dx] = 0.0
    elif dx < 0:
        valid[:, :, width + dx :] = 0.0
    if dy > 0:
        valid[:, :dy, :] = 0.0
    elif dy < 0:
        valid[:, height + dy :, :] = 0.0
    return output.contiguous(), valid


def _translate_mask(
    mask: torch.Tensor, dx: int, dy: int
) -> torch.Tensor:
    height, width = mask.shape[-2:]
    output = torch.zeros_like(mask)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    target_x0 = source_x0 + dx
    target_x1 = source_x1 + dx
    target_y0 = source_y0 + dy
    target_y1 = source_y1 + dy
    output[
        :,
        target_y0:target_y1,
        target_x0:target_x1,
    ] = mask[
        :,
        source_y0:source_y1,
        source_x0:source_x1,
    ]
    return output


def _affine_sample(
    tensor: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample with a 2x3 affine and return the image plus a valid mask.

    The valid mask marks pixels whose source fell inside the original
    support; border pixels sampled from outside are zero-padded and
    excluded from every metric, exactly as the integer shift path does.
    """

    batched = tensor.unsqueeze(0)
    grid = F.affine_grid(
        theta.unsqueeze(0),
        list(batched.shape),
        align_corners=False,
    )
    sampled = F.grid_sample(
        batched,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0)
    support = F.grid_sample(
        torch.ones_like(batched[:, :1]),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0)
    return sampled.contiguous(), (support >= 0.999).float()


def _translate_subpixel(
    tensor: torch.Tensor,
    dx: float,
    dy: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear translation following reference(x) <-> target(x + dx).

    Matches the integer path's sign convention: a positive dx moves image
    content in the +x direction, so the content that lands at column x
    came from column x - dx.
    """

    height, width = tensor.shape[-2:]
    # align_corners=False maps pixel x to (2x + 1) / W - 1, so one pixel
    # is 2 / W in normalized units. Using W - 1 here (the align_corners
    # =True convention) leaves a sub-pixel error that shows up as
    # interpolation blur even for integer shifts.
    theta = torch.tensor(
        [
            [1.0, 0.0, -2.0 * float(dx) / max(1, width)],
            [0.0, 1.0, -2.0 * float(dy) / max(1, height)],
        ],
        dtype=tensor.dtype,
    )
    return _affine_sample(tensor, theta)


def _rotate(
    tensor: torch.Tensor,
    degrees: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate about the image centre by ``degrees``."""

    radians = math.radians(float(degrees))
    cos = math.cos(radians)
    sin = math.sin(radians)
    theta = torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0]],
        dtype=tensor.dtype,
    )
    return _affine_sample(tensor, theta)


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(
        -radius, radius + 1, dtype=image.dtype, device=image.device
    )
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    channels = int(image.shape[0])
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    value = image.unsqueeze(0)
    value = F.pad(value, (radius, radius, 0, 0), mode="reflect")
    value = F.conv2d(value, horizontal, groups=channels)
    value = F.pad(value, (0, 0, radius, radius), mode="reflect")
    value = F.conv2d(value, vertical, groups=channels)
    return value.squeeze(0)


class CorruptedPairDataset(Dataset[dict[str, Any]]):
    """Deterministic corruption wrapper for an evaluated paired dataset."""

    def __init__(
        self,
        base_dataset: Dataset[dict[str, Any]],
        spec: CorruptionSpec,
        normalization: str = "imagenet",
    ) -> None:
        self.base_dataset = base_dataset
        self.spec = spec
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _applied_translation(
        self,
        sample_id: str,
        branch: str,
    ) -> tuple[float, float]:
        if self.spec.name not in {"shift", "subpixel_shift"}:
            return 0.0, 0.0
        seed = _stable_seed(
            self.spec.seed,
            sample_id,
            self.spec.name,
            self.spec.severity,
            branch,
        )
        if self.spec.name == "shift":
            dx, dy = _translation_direction(seed, int(self.spec.value))
            return float(dx), float(dy)
        unit_x, unit_y = _unit_direction(seed)
        return (
            unit_x * float(self.spec.value),
            unit_y * float(self.spec.value),
        )

    def _apply_image(
        self,
        image: torch.Tensor,
        sample_id: str,
        branch: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unit = _to_unit(image, self.normalization)
        height, width = unit.shape[-2:]
        valid = torch.ones(1, height, width, dtype=torch.float32)
        seed = _stable_seed(
            self.spec.seed,
            sample_id,
            self.spec.name,
            self.spec.severity,
            branch,
        )
        if self.spec.name == "shift":
            dx, dy = _translation_direction(
                seed, int(self.spec.value)
            )
            shifted, valid = _translate_reflect(unit, dx=dx, dy=dy)
            return _from_unit(shifted, self.normalization), valid
        if self.spec.name == "subpixel_shift":
            unit_x, unit_y = _unit_direction(seed)
            shifted, valid = _translate_subpixel(
                unit,
                dx=unit_x * float(self.spec.value),
                dy=unit_y * float(self.spec.value),
            )
            return _from_unit(shifted, self.normalization), valid
        if self.spec.name == "rotate":
            sign = -1.0 if seed % 2 == 0 else 1.0
            rotated, valid = _rotate(
                unit,
                degrees=sign * float(self.spec.value),
            )
            return _from_unit(rotated, self.normalization), valid
        if self.spec.name == "brightness":
            sign = -1.0 if seed % 2 == 0 else 1.0
            factor = 1.0 + sign * self.spec.value
            unit = (unit * factor).clamp(0.0, 1.0)
        elif self.spec.name == "gaussian_blur":
            unit = _gaussian_blur(unit, sigma=self.spec.value)
        elif self.spec.name == "gaussian_noise":
            generator = torch.Generator()
            generator.manual_seed(seed)
            noise = torch.randn(
                unit.shape,
                dtype=unit.dtype,
                generator=generator,
            )
            unit = (unit + noise * self.spec.value).clamp(0.0, 1.0)
        elif self.spec.name != "clean":
            raise ValueError(f"Unsupported corruption '{self.spec.name}'")
        return _from_unit(unit, self.normalization), valid

    def __getitem__(self, index: int) -> dict[str, Any]:
        original = self.base_dataset[index]
        sample = dict(original)
        t1 = original["image_t1"].clone()
        t2 = original["image_t2"].clone()
        mask = original["mask"].clone()
        height, width = mask.shape[-2:]
        valid = torch.ones(1, height, width, dtype=torch.float32)
        sample_id = str(original["id"])
        shift_dx: float = 0
        shift_dy: float = 0
        rotation_degrees: float = 0.0

        if self.spec.name == "clean":
            pass
        elif self.spec.mode == "t1":
            t1, valid = self._apply_image(t1, sample_id, "shared")
            applied_dx, applied_dy = self._applied_translation(
                sample_id,
                "shared",
            )
            # shift_dx/dy always mean the relative T2 displacement with
            # respect to T1. Moving T1 therefore reverses the sign.
            shift_dx, shift_dy = -applied_dx, -applied_dy
        elif self.spec.mode == "t2":
            t2, valid = self._apply_image(t2, sample_id, "shared")
            if self.spec.name in {"shift", "subpixel_shift", "rotate"}:
                seed = _stable_seed(
                    self.spec.seed,
                    sample_id,
                    self.spec.name,
                    self.spec.severity,
                    "shared",
                )
                if self.spec.name in {"shift", "subpixel_shift"}:
                    shift_dx, shift_dy = self._applied_translation(
                        sample_id,
                        "shared",
                    )
                else:
                    # A rotation has no single correct translation. The
                    # ground-truth offset stays zero and the angle is
                    # recorded separately, so that any tool comparing an
                    # estimate against shift_dx/shift_dy cannot silently
                    # treat rotation as if it were a translation.
                    rotation_degrees = (
                        -1.0 if seed % 2 == 0 else 1.0
                    ) * float(self.spec.value)
        elif self.spec.mode == "common":
            t1, valid_t1 = self._apply_image(t1, sample_id, "common")
            t2, valid_t2 = self._apply_image(t2, sample_id, "common")
            valid = valid_t1 * valid_t2
            if self.spec.name == "shift":
                dx, dy = self._applied_translation(
                    sample_id,
                    "common",
                )
                mask = _translate_mask(
                    mask,
                    dx=int(dx),
                    dy=int(dy),
                )
            # Both dates receive the same transform, so their relative
            # translation remains zero.
            shift_dx, shift_dy = 0.0, 0.0
        elif self.spec.mode == "independent":
            t1, valid_t1 = self._apply_image(t1, sample_id, "t1")
            t2, valid_t2 = self._apply_image(t2, sample_id, "t2")
            valid = valid_t1 * valid_t2
            t1_dx, t1_dy = self._applied_translation(sample_id, "t1")
            t2_dx, t2_dy = self._applied_translation(sample_id, "t2")
            shift_dx = t2_dx - t1_dx
            shift_dy = t2_dy - t1_dy

        sample["image_t1"] = t1
        sample["image_t2"] = t2
        sample["mask"] = mask
        sample["valid_mask"] = valid
        sample["corruption"] = self.spec.name
        sample["corruption_mode"] = self.spec.mode
        sample["severity"] = self.spec.severity
        sample["shift_dx"] = shift_dx
        sample["shift_dy"] = shift_dy
        sample["rotation_degrees"] = rotation_degrees
        return sample


class OracleInverseShiftDataset(Dataset[dict[str, Any]]):
    """Undo a known synthetic one-sided shift for a recoverability check."""

    def __init__(
        self,
        shifted_dataset: Dataset[dict[str, Any]],
        target: str = "t2",
    ) -> None:
        if target not in {"t1", "t2"}:
            raise ValueError("Oracle inverse target must be t1 or t2")
        self.shifted_dataset = shifted_dataset
        self.target = target

    def __len__(self) -> int:
        return len(self.shifted_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        original = self.shifted_dataset[index]
        sample = dict(original)
        dx = int(original["shift_dx"])
        dy = int(original["shift_dy"])
        image_key = f"image_{self.target}"
        recovered, _ = _translate_reflect(
            original[image_key],
            dx=-dx,
            dy=-dy,
        )
        sample[image_key] = recovered
        height, width = recovered.shape[-2:]
        sample["valid_mask"] = _paired_shift_valid_mask(
            height,
            width,
            dx,
            dy,
        )
        sample["oracle_inverse_shift"] = True
        return sample


def _paired_shift_valid_mask(
    height: int,
    width: int,
    dx: int,
    dy: int,
) -> torch.Tensor:
    """Intersection support shared by raw and inverse-shift inference.

    A one-sided synthetic shift loses one border. Applying the inverse
    transform loses the opposite border as well. Comparing the two F1
    values therefore requires the symmetric interior that is valid for
    both paths; otherwise their confusion matrices have different pixel
    support and the headroom is not paired.
    """

    valid = torch.ones(1, height, width, dtype=torch.float32)
    border_x = abs(int(dx))
    border_y = abs(int(dy))
    if border_x:
        valid[:, :, :border_x] = 0.0
        valid[:, :, width - border_x :] = 0.0
    if border_y:
        valid[:, :border_y, :] = 0.0
        valid[:, height - border_y :, :] = 0.0
    return valid


class PairedShiftSupportDataset(Dataset[dict[str, Any]]):
    """Use inverse-shift intersection support without changing images."""

    def __init__(
        self,
        shifted_dataset: Dataset[dict[str, Any]],
    ) -> None:
        self.shifted_dataset = shifted_dataset

    def __len__(self) -> int:
        return len(self.shifted_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        original = self.shifted_dataset[index]
        sample = dict(original)
        height, width = original["mask"].shape[-2:]
        sample["valid_mask"] = _paired_shift_valid_mask(
            height,
            width,
            int(original["shift_dx"]),
            int(original["shift_dy"]),
        )
        sample["paired_shift_support"] = True
        return sample
