from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from rs_cd.affine_crops import cut_border_free

from .manifest import ManifestRow, read_manifest


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def affine_misregistration_matrix(
    level: int,
    u_scale: float,
    u_x: float,
    u_y: float,
    *,
    size: int,
    max_translation_px: float,
    scale_range: tuple[float, float],
    levels: int,
) -> np.ndarray:
    """The P27 test perturbation without rotation (rs_cd.affine_registration).

    Level k uses k / levels of the scale and translation ranges with the same
    uniform draws, built about the crop centre like cv2.getRotationMatrix2D.
    """
    fraction = level / levels
    low, high = scale_range
    scale = 1.0 + fraction * (low + (high - low) * u_scale - 1.0)
    tx = fraction * max_translation_px * (2.0 * u_x - 1.0)
    ty = fraction * max_translation_px * (2.0 * u_y - 1.0)
    centre = size / 2.0
    return np.array(
        [[scale, 0.0, (1.0 - scale) * centre + tx],
         [0.0, scale, (1.0 - scale) * centre + ty]],
        dtype=np.float64,
    )


class PairedChangeDataset(Dataset[dict[str, Any]]):
    """Paired T1/T2 binary change-detection dataset backed by a manifest."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        split: str,
        image_size: int | None = None,
        normalization: str = "imagenet",
        augment: bool = False,
        random_flip: bool = True,
        random_rotate90: bool = True,
        mask_threshold: int = 127,
        cache_mode: str = "none",
        positive_crop_probability: float = 0.0,
        positive_crop_min_pixels: int = 1,
        positive_crop_trials: int = 8,
        misregistration_probability: float = 0.0,
        misregistration_pixels: tuple[int, ...] | list[int] = (2, 4, 8),
        misregistration_target: str = "t2",
        affine_misregistration_probability: float = 0.0,
        affine_max_translation_px: float = 25.0,
        affine_scale_range: tuple[float, float] | list[float] = (0.80, 1.25),
        affine_levels: int = 3,
        affine_context_margin: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.rows: list[ManifestRow] = read_manifest(manifest_path, split=split)
        self.split = split
        self.image_size = image_size
        self.normalization = normalization
        self.augment = augment
        self.random_flip = random_flip
        self.random_rotate90 = random_rotate90
        self.mask_threshold = int(mask_threshold)
        if not 0 <= self.mask_threshold <= 255:
            raise ValueError("mask_threshold must be in [0, 255]")
        self.positive_crop_probability = float(positive_crop_probability)
        self.positive_crop_min_pixels = int(positive_crop_min_pixels)
        self.positive_crop_trials = int(positive_crop_trials)
        if not 0.0 <= self.positive_crop_probability <= 1.0:
            raise ValueError("positive_crop_probability must be in [0, 1]")
        if self.positive_crop_min_pixels < 1:
            raise ValueError("positive_crop_min_pixels must be positive")
        if self.positive_crop_trials < 1:
            raise ValueError("positive_crop_trials must be positive")
        self.misregistration_probability = float(
            misregistration_probability
        )
        self.misregistration_pixels = tuple(
            int(value) for value in misregistration_pixels
        )
        self.misregistration_target = str(misregistration_target)
        if not 0.0 <= self.misregistration_probability <= 1.0:
            raise ValueError(
                "misregistration_probability must be in [0, 1]"
            )
        if (
            not self.misregistration_pixels
            or any(value < 1 for value in self.misregistration_pixels)
        ):
            raise ValueError(
                "misregistration_pixels must contain positive integers"
            )
        if self.misregistration_target not in {"t1", "t2", "random"}:
            raise ValueError(
                "misregistration_target must be t1, t2, or random"
            )
        # P29: T2 perturbed like the P27 test levels; pixels without T2
        # content are reported in "valid_mask" and left out of the loss.
        self.affine_misregistration_probability = float(
            affine_misregistration_probability
        )
        self.affine_max_translation_px = float(affine_max_translation_px)
        self.affine_scale_range = tuple(float(v) for v in affine_scale_range)
        self.affine_levels = int(affine_levels)
        if not 0.0 <= self.affine_misregistration_probability <= 1.0:
            raise ValueError(
                "affine_misregistration_probability must be in [0, 1]"
            )
        if len(self.affine_scale_range) != 2 or self.affine_levels < 1:
            raise ValueError("affine_scale_range needs two values and affine_levels >= 1")
        if (
            self.affine_misregistration_probability > 0.0
            and self.misregistration_probability > 0.0
        ):
            raise ValueError("use either integer or affine misregistration, not both")
        # P30: the crop keeps this many real pixels on every side and T2 is warped
        # with them, so the perturbed crop has no zero border.
        self.affine_context_margin = (
            None if affine_context_margin is None else int(affine_context_margin)
        )
        if self.affine_context_margin is not None and image_size is None:
            raise ValueError("affine_context_margin needs a crop size")
        self.cache_mode = cache_mode
        if self.cache_mode not in {"none", "ram"}:
            raise ValueError("cache_mode must be 'none' or 'ram'")
        if normalization not in {"imagenet", "unit", "none"}:
            raise ValueError(
                "normalization must be one of: imagenet, unit, none"
            )
        self._cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
        self._positive_indices: list[np.ndarray] | None = None
        if self.cache_mode == "ram":
            self._cache = [self._load_arrays(row) for row in self.rows]
            self._positive_indices = [
                np.flatnonzero(mask > self.mask_threshold).astype(
                    np.uint32, copy=False
                )
                for _, _, mask in self._cache
            ]
            cache_bytes = sum(
                array.nbytes
                for sample in self._cache
                for array in sample
            )
            index_bytes = sum(
                indices.nbytes for indices in self._positive_indices
            )
            print(
                f"Cached {len(self._cache)} {self.split} samples in RAM "
                f"({cache_bytes / (1024**3):.2f} GiB uint8 + "
                f"{index_bytes / (1024**3):.2f} GiB positive indices)"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, relative_path: str) -> Path:
        path = (self.dataset_root / relative_path).resolve()
        try:
            path.relative_to(self.dataset_root)
        except ValueError as error:
            raise ValueError(
                f"Manifest path escapes dataset root: {relative_path}"
            ) from error
        return path

    def _load_arrays(
        self, row: ManifestRow
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with Image.open(self._resolve(row.t1_path)) as image:
            t1 = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(self._resolve(row.t2_path)) as image:
            t2 = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(self._resolve(row.mask_path)) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        if t1.shape[:2] != t2.shape[:2] or t1.shape[:2] != mask.shape:
            raise ValueError(
                f"Size mismatch for '{row.sample_id}': "
                f"T1={t1.shape}, T2={t2.shape}, mask={mask.shape}"
            )
        return t1, t2, mask

    @staticmethod
    def _pad_to_size(
        image: Image.Image, target_h: int, target_w: int, fill: int | tuple[int, ...]
    ) -> Image.Image:
        pad_w = max(0, target_w - image.width)
        pad_h = max(0, target_h - image.height)
        if pad_w == 0 and pad_h == 0:
            return image
        return ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=fill)

    def _joint_transform(
        self,
        t1: Image.Image,
        t2: Image.Image,
        mask: Image.Image,
        positive_flat: np.ndarray | None = None,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        if self.image_size is not None:
            size = int(self.image_size)
            t1 = self._pad_to_size(t1, size, size, (0, 0, 0))
            t2 = self._pad_to_size(t2, size, size, (0, 0, 0))
            mask = self._pad_to_size(mask, size, size, 0)
            max_x = t1.width - size
            max_y = t1.height - size
            if self.augment:
                x0, y0 = self._random_crop_origin(
                    np.asarray(mask, dtype=np.uint8),
                    size=size,
                    max_x=max_x,
                    max_y=max_y,
                    positive_flat=positive_flat,
                )
            else:
                x0 = max_x // 2
                y0 = max_y // 2
            box = (x0, y0, x0 + size, y0 + size)
            t1, t2, mask = t1.crop(box), t2.crop(box), mask.crop(box)

        if self.augment and self.random_flip:
            if random.random() < 0.5:
                t1 = ImageOps.mirror(t1)
                t2 = ImageOps.mirror(t2)
                mask = ImageOps.mirror(mask)
            if random.random() < 0.5:
                t1 = ImageOps.flip(t1)
                t2 = ImageOps.flip(t2)
                mask = ImageOps.flip(mask)

        if self.augment and self.random_rotate90:
            rotations = random.randint(0, 3)
            if rotations:
                angle = 90 * rotations
                t1 = t1.rotate(angle, resample=Image.Resampling.BILINEAR)
                t2 = t2.rotate(angle, resample=Image.Resampling.BILINEAR)
                mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
        return t1, t2, mask

    def _random_crop_origin(
        self,
        mask: np.ndarray,
        size: int,
        max_x: int,
        max_y: int,
        positive_flat: np.ndarray | None = None,
    ) -> tuple[int, int]:
        def fully_random() -> tuple[int, int]:
            return (
                random.randint(0, max_x) if max_x > 0 else 0,
                random.randint(0, max_y) if max_y > 0 else 0,
            )

        if (
            self.positive_crop_probability <= 0.0
            or random.random() >= self.positive_crop_probability
        ):
            return fully_random()

        if positive_flat is None:
            positive_flat = np.flatnonzero(mask > self.mask_threshold)
        if positive_flat.size == 0:
            return fully_random()

        best_origin: tuple[int, int] | None = None
        best_positive_count = -1
        width = int(mask.shape[1])
        for _ in range(self.positive_crop_trials):
            flat_index = int(
                positive_flat[random.randrange(int(positive_flat.size))]
            )
            anchor_y, anchor_x = divmod(flat_index, width)
            low_x = max(0, anchor_x - size + 1)
            high_x = min(anchor_x, max_x)
            low_y = max(0, anchor_y - size + 1)
            high_y = min(anchor_y, max_y)
            x0 = random.randint(low_x, high_x) if high_x > low_x else low_x
            y0 = random.randint(low_y, high_y) if high_y > low_y else low_y
            positive_count = int(
                np.count_nonzero(
                    mask[y0 : y0 + size, x0 : x0 + size]
                    > self.mask_threshold
                )
            )
            if positive_count > best_positive_count:
                best_positive_count = positive_count
                best_origin = (x0, y0)
            if positive_count >= self.positive_crop_min_pixels:
                return x0, y0
        return best_origin if best_origin is not None else fully_random()

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if self.normalization in {"unit", "imagenet"}:
            tensor = tensor / 255.0
        if self.normalization == "imagenet":
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor

    @staticmethod
    def _translate_reflect(
        image: Image.Image,
        dx: int,
        dy: int,
    ) -> Image.Image:
        array = np.asarray(image, dtype=np.uint8)
        height, width = array.shape[:2]
        pad = max(abs(dx), abs(dy))
        if pad == 0:
            return image.copy()
        padded = np.pad(
            array,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="reflect",
        )
        start_y = pad - dy
        start_x = pad - dx
        translated = padded[
            start_y : start_y + height,
            start_x : start_x + width,
        ]
        return Image.fromarray(translated.copy(), mode="RGB")

    def _apply_misregistration(
        self,
        t1: Image.Image,
        t2: Image.Image,
    ) -> tuple[Image.Image, Image.Image, int, int, str]:
        if (
            not self.augment
            or self.misregistration_probability <= 0.0
            or random.random() >= self.misregistration_probability
        ):
            return t1, t2, 0, 0, "none"
        pixels = random.choice(self.misregistration_pixels)
        direction_x, direction_y = random.choice(
            (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            )
        )
        dx = direction_x * pixels
        dy = direction_y * pixels
        target = self.misregistration_target
        if target == "random":
            target = random.choice(("t1", "t2"))
        if target == "t1":
            t1 = self._translate_reflect(t1, dx=dx, dy=dy)
        else:
            t2 = self._translate_reflect(t2, dx=dx, dy=dy)
        return t1, t2, dx, dy, target

    @property
    def _affine_enabled(self) -> bool:
        return self.augment and self.affine_misregistration_probability > 0.0

    def _apply_affine_misregistration(
        self, t2: Image.Image
    ) -> tuple[Image.Image, np.ndarray, int]:
        """(T2, valid mask, level); level 0 leaves T2 unchanged."""
        width, height = t2.size
        valid = np.ones((height, width), dtype=np.float32)
        if random.random() >= self.affine_misregistration_probability:
            return t2, valid, 0
        if width != height:
            raise ValueError("affine misregistration needs square crops")
        level = random.randint(1, self.affine_levels)
        matrix = affine_misregistration_matrix(
            level, random.random(), random.random(), random.random(),
            size=width,
            max_translation_px=self.affine_max_translation_px,
            scale_range=self.affine_scale_range,
            levels=self.affine_levels,
        )
        warped = cv2.warpAffine(
            np.asarray(t2, dtype=np.uint8), matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        covered = cv2.warpAffine(
            np.ones((height, width), dtype=np.uint8), matrix, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return Image.fromarray(warped, mode="RGB"), covered.astype(np.float32), level

    def _context_sample(
        self, t1: np.ndarray, t2: np.ndarray, mask: np.ndarray, positive_flat: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """P30 crop: T1, border-free T2, mask (all size x size) and the level (0 = unperturbed).

        Only a perturbed crop keeps the margin; an unperturbed one is placed as in the
        original recipe.
        """
        perturbed = random.random() < self.affine_misregistration_probability
        size = int(self.image_size)
        margin = self.affine_context_margin if perturbed else 0
        height, width = mask.shape[:2]
        max_x, max_y = width - size, height - size
        if max_x < 2 * margin or max_y < 2 * margin:
            raise ValueError(f"a {height}x{width} tile has no {size} px crop with a {margin} px margin")
        x0, y0 = self._random_crop_origin(mask, size=size, max_x=max_x, max_y=max_y, positive_flat=positive_flat)
        x0 = min(max(x0, margin), max_x - margin)
        y0 = min(max(y0, margin), max_y - margin)
        crop_t1 = t1[y0:y0 + size, x0:x0 + size]
        crop_mask = mask[y0:y0 + size, x0:x0 + size]
        region = t2[y0 - margin:y0 + size + margin, x0 - margin:x0 + size + margin]
        if self.random_flip:
            if random.random() < 0.5:
                crop_t1, crop_mask, region = (np.fliplr(a) for a in (crop_t1, crop_mask, region))
            if random.random() < 0.5:
                crop_t1, crop_mask, region = (np.flipud(a) for a in (crop_t1, crop_mask, region))
        if self.random_rotate90:
            turns = random.randint(0, 3)
            crop_t1, crop_mask, region = (np.rot90(a, turns) for a in (crop_t1, crop_mask, region))
        region = np.ascontiguousarray(region)
        level = 0
        crop_t2 = region[margin:margin + size, margin:margin + size]
        if perturbed:
            level = random.randint(1, self.affine_levels)
            matrix = affine_misregistration_matrix(
                level, random.random(), random.random(), random.random(),
                size=size,
                max_translation_px=self.affine_max_translation_px,
                scale_range=self.affine_scale_range,
                levels=self.affine_levels,
            )
            crop_t2 = cut_border_free(region, matrix, margin, size)
        return (np.ascontiguousarray(crop_t1), np.ascontiguousarray(crop_t2),
                np.ascontiguousarray(crop_mask), level)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if self._cache is None:
            t1_array, t2_array, mask_array_raw = self._load_arrays(row)
            positive_flat = None
        else:
            t1_array, t2_array, mask_array_raw = self._cache[index]
            positive_flat = (
                self._positive_indices[index]
                if self._positive_indices is not None
                else None
            )
        if self._affine_enabled and self.affine_context_margin is not None:
            t1_crop, t2_crop, mask_crop, level = self._context_sample(
                t1_array, t2_array, mask_array_raw, positive_flat
            )
            mask_tensor = torch.from_numpy(
                (mask_crop > self.mask_threshold).astype(np.float32)
            ).unsqueeze(0)
            return {
                "image_t1": self._image_to_tensor(Image.fromarray(t1_crop, mode="RGB")),
                "image_t2": self._image_to_tensor(Image.fromarray(t2_crop, mode="RGB")),
                "mask": mask_tensor,
                "id": row.sample_id,
                "parent_id": row.parent_id,
                "split": row.split,
                "misregistration_dx": 0,
                "misregistration_dy": 0,
                "misregistration_target": "none",
                # Every pixel has T2 content; the mask keeps the P29 batch layout.
                "valid_mask": torch.ones_like(mask_tensor),
                "affine_level": level,
            }
        t1 = Image.fromarray(t1_array, mode="RGB")
        t2 = Image.fromarray(t2_array, mode="RGB")
        mask = Image.fromarray(mask_array_raw, mode="L")
        t1, t2, mask = self._joint_transform(
            t1, t2, mask, positive_flat=positive_flat
        )
        t1, t2, shift_dx, shift_dy, shift_target = (
            self._apply_misregistration(t1, t2)
        )
        extra: dict[str, Any] = {}
        if self._affine_enabled:
            t2, valid, level = self._apply_affine_misregistration(t2)
            extra = {
                "valid_mask": torch.from_numpy(valid).unsqueeze(0),
                "affine_level": level,
            }
        mask_array = (
            np.asarray(mask, dtype=np.uint8) > self.mask_threshold
        ).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return {
            "image_t1": self._image_to_tensor(t1),
            "image_t2": self._image_to_tensor(t2),
            "mask": mask_tensor,
            "id": row.sample_id,
            "parent_id": row.parent_id,
            "split": row.split,
            "misregistration_dx": shift_dx,
            "misregistration_dy": shift_dy,
            "misregistration_target": shift_target,
            **extra,
        }
