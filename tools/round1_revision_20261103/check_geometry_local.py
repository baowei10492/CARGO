"""CPU coordinate checks only: no server access, model loading, or data sampling."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np

path = Path(__file__).with_name("round1_formal.py")
spec = importlib.util.spec_from_file_location("round1_geometry", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

size, tile_size = 256, 512
item = {"top": 64, "left": 192, "crop_id": "coordinate_check"}
y, x = np.mgrid[:tile_size, :tile_size]
tile = np.stack([x % 256, y % 256, (x + y) % 256], axis=2).astype(np.uint8)

identity = np.eye(2, 3)
geometry = m.Geometry(tile, item, identity, size)
original = tile[64:320, 192:448].copy()
np.testing.assert_array_equal(geometry.raw(), original)
corrected, missing = geometry.corrected(identity)
np.testing.assert_array_equal(corrected, original)
assert missing == 0
assert geometry.inside_crop().all()

translation = np.array([[1., 0., 9.], [0., 1., -7.]])
geometry = m.Geometry(tile, item, translation, size)
corrected, missing = geometry.corrected(translation)
np.testing.assert_array_equal(corrected, original)
assert missing == 0
gx, gy = geometry.forward_crop_grid()
np.testing.assert_allclose(gx, np.mgrid[:size, :size][1] + 9)
np.testing.assert_allclose(gy, np.mgrid[:size, :size][0] - 7)

# A raw prediction tied to the observed target must sample at M(p), not inv(M)(p).
truth = np.zeros((size, size), np.float32)
truth[60:120, 70:130] = 1
raw_probability = cv2.warpAffine(truth, translation, (size, size))
np.testing.assert_array_equal(m.output_remap(raw_probability, translation), truth)
np.testing.assert_array_equal(m.output_remap(truth, identity), truth)

for seed in range(12):
    angle = (-3 + seed * 6 / 11)
    matrix = cv2.getRotationMatrix2D((128, 128), angle, 0.94 + seed * 0.01)
    matrix[:, 2] += [7, -6]
    unit = m.smooth_unit_field(tile_size, f"case{seed}", 3407, 9)
    np.testing.assert_allclose(np.sqrt(np.square(unit.astype(float)).sum(axis=2).mean()), 1, atol=1e-6)
    np.testing.assert_allclose(unit.mean(axis=(0, 1)), 0, atol=1e-5)
    for rms in (1, 3):
        dense = m.Geometry(tile, item, matrix, size, unit * rms)
        dense.raw()  # Every source pixel in this interior crop must be supported.
        qx, qy = dense.forward_crop_grid()
        displacement = cv2.remap(unit * rms, (qx + item["left"]).astype(np.float32),
                                 (qy + item["top"]).astype(np.float32), cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        back = cv2.invertAffineTransform(m.to_tile(matrix, item["top"], item["left"]))
        px, py = m.affine_xy(back, qx + item["left"] + displacement[..., 0],
                            qy + item["top"] + displacement[..., 1])
        yy, xx = np.mgrid[:size, :size]
        error = np.hypot(px - (xx + item["left"]), py - (yy + item["top"]))
        assert error.max() < 0.06, error.max()
        assert dense.inside_crop().mean() > 0.80

counts = m.binary_counts(truth, truth.astype(bool), np.ones_like(truth, bool), 0.5)
assert counts == [3600, 0, 0, size * size - 3600]
assert float(m.f1(counts)) == 1
assert float(m.f1([0, 0, 0, 1])) == 0
print("PASS: identity, signed translation, output-map direction, dense inverse, real-content support, counts/F1")
