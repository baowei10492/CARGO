"""CPU-only validation of the stronger transform protocol; no model inference."""
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

import round1_formal as base
from round1_stress import StressGeometry, continuous_field_sample, require_pinned_base, source_support


def main():
    root = Path(__file__).resolve().parent
    cv2.setNumThreads(4)
    protocol = json.loads((root / 'round1_stress_protocol.json').read_text())
    require_pinned_base(protocol, root)
    item = {'left': 64, 'top': 64, 'crop_id': 'coordinate-check'}
    identity = np.eye(2, 3)
    assert source_support(identity, None, item)[0]
    shifted = identity.copy(); shifted[0, 2] = 100
    assert not source_support(shifted, None, item)[0]
    field = base.smooth_unit_field(512, item['crop_id'], 3407, 9) * 6
    assert abs(float(np.sqrt(np.square(field.astype(float)).sum(2).mean())) - 6) < 1e-5
    assert np.linalg.norm(field.mean(axis=(0, 1))) < 1e-4
    angle, scale = 21, .87
    matrix = cv2.getRotationMatrix2D((128, 128), angle, scale)
    matrix[:, 2] += [17, -13]
    geometry = StressGeometry(None, item, matrix, 256, field)
    qx, qy = geometry.forward_crop_grid()
    x, y = qx + 64, qy + 64
    sampled = continuous_field_sample(field, x, y)
    mx, my = base.affine_xy(base.to_tile(matrix, 64, 64), *np.mgrid[64:320, 64:320][::-1])
    error = np.hypot(x + sampled[..., 0] - mx, y + sampled[..., 1] - my)
    assert float(error.max()) <= 1e-5
    dummy = np.zeros((512, 512, 3), np.uint8)
    geometry = StressGeometry(dummy, item, matrix, 256, field)
    covered, _ = source_support(matrix, field, item)
    if covered:
        geometry.make()
        assert np.all(geometry.crop(geometry.content) == 1)
    # The corrected support solver must not change the generated test inputs.
    generator = np.random.default_rng(3407)
    image = generator.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    image_a = base.Geometry(image, item, identity, 256, field)
    image_b = StressGeometry(image, item, identity, 256, field)
    assert np.array_equal(image_a.raw(), image_b.raw())
    assert np.array_equal(image_a.corrected(identity)[0], image_b.corrected(identity)[0])
    # Independent closed-form inverse for a constant displacement field.
    constant_field = np.broadcast_to(np.array([6., -4.], dtype=np.float32), (512, 512, 2)).copy()
    constant_geometry = StressGeometry(None, item, matrix, 256, constant_field)
    qcx, qcy = constant_geometry.forward_crop_grid()
    grid_y, grid_x = np.mgrid[:256, :256].astype(float)
    mcx, mcy = base.affine_xy(constant_geometry.tile_matrix, grid_x + 64, grid_y + 64)
    assert np.max(np.abs(qcx + 64 - (mcx - 6))) < 1e-9
    assert np.max(np.abs(qcy + 64 - (mcy + 4))) < 1e-9
    paths = sorted((Path('reports/round1_20261103/E2/records')).glob('*.json'))
    counts = {'ts_level3': 0, 'ts_level3_rot10': 0, 'ts_level3_rot30': 0, 'ts_level3_local6': 0}
    eligible, excluded, checked_common_masks = [], [], []
    maximum_inverse_residual = 0.
    maximum_inverse_iterations = 0
    regression_case = None
    started = time.time()
    for index, path in enumerate(paths, 1):
        record = json.loads(path.read_text())
        c = record['conditions']
        m1 = np.array(c['ts_level1']['synthetic_matrix'])
        scale1 = float(m1[0, 0])
        scale3 = 1 + 3 * (scale1 - 1)
        shift3 = 3 * (m1[:, 2] - 128 * (1 - scale1))
        unit_angle = c['ts_level1_rot3']['rotation_deg'] / 3
        field = base.smooth_unit_field(512, record['crop_id'], 3407, 9) * 6
        masks = []
        okay = True
        bundle = []
        for condition in protocol['E2']['conditions']:
            m = cv2.getRotationMatrix2D((128, 128), unit_angle * condition['rotation_limit_deg'], scale3)
            m[:, 2] += shift3
            d = field if condition['backward_field_rms_px'] else None
            support, _ = source_support(m, d, record)
            counts[condition['name']] += int(support)
            okay &= support
            bundle.append((m, d))
        if okay:
            eligible.append(record['crop_id'])
            common = np.ones((256, 256), dtype=bool)
            for m, d in bundle:
                geometry = StressGeometry(None, record, m, 256, d)
                common &= geometry.inside_crop()
                if d is not None:
                    maximum_inverse_residual = max(maximum_inverse_residual, geometry.inverse_stats['max_residual_px'])
                    maximum_inverse_iterations = max(maximum_inverse_iterations, geometry.inverse_stats['iterations'])
                    if record['crop_id'] == 'whu_test_r02c15|y0192x0192':
                        failed_previously = False
                        try:
                            base.Geometry(None, record, m, 256, d).forward_crop_grid()
                        except RuntimeError:
                            failed_previously = True
                        assert failed_previously
                        regression_case = {'crop_id':record['crop_id'], 'old_solver_failed':True,
                                           'new_solver': geometry.inverse_stats}
            assert common.any()
            checked_common_masks.append(int(common.sum()))
        else:
            excluded.append(record['crop_id'])
        if index % 500 == 0 or index == len(paths):
            print(f'CPU_SOURCE_GEOMETRY {index}/{len(paths)} eligible={len(eligible)}', flush=True)
    assert len(paths) == 2552
    assert counts['ts_level3_rot10'] == 2472
    assert counts['ts_level3_rot30'] == 2156
    assert regression_case is not None
    fingerprint = hashlib.sha256('\n'.join(sorted(eligible)).encode()).hexdigest()
    assert fingerprint == protocol['E2']['expected_geometry_selection']['fingerprint']
    assert len(eligible) == protocol['E2']['expected_geometry_selection']['crops']
    output = {'validation_scope': 'Full CPU source-support and forward-inverse checks for all geometry-eligible crops, all four conditions; no GPU inference, source imagery, or new F1 measured',
              'candidate_crops': len(paths), 'source_supported_by_condition': counts,
              'common_source_supported_crops': len(eligible), 'source_excluded_crops': len(excluded),
              'nonlinear_inverse_check_max_residual_px': maximum_inverse_residual,
              'nonlinear_inverse_iterations_max': maximum_inverse_iterations,
              'complete_common_mask_checks': len(checked_common_masks),
              'common_valid_pixel_min': min(checked_common_masks),
              'source_selection_fingerprint': fingerprint,
              'regression_case':regression_case,
              'unchanged_image_generation_verified':True,
              'constant_field_analytic_inverse_verified':True,
              'elapsed_seconds': time.time() - started,
              'uploaded_files_sha256': {name: base.digest(root / name) for name in
                                         ['round1_stress.py', 'round1_stress_protocol.json', 'run_round1_stress.sh']}}
    base.atomic_json(root / 'stress_v2_full_geometry_validation.json', output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
