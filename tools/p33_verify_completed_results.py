"""CPU verification of completed P33 selector records; no image inference."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from tools.p31_statistics import Cell, check_cell
from tools.p31_selector_ablation import sources
from tools.p33_selector_runs import frozen_anchor


torch.set_num_threads(2)
root = Path('reports/p33')
verified = []
errors = []
for variant in ['roma_ts_ransac', 'phasecorr_fm', 'v2r_conf_only', 'v2r_no_change_label']:
    for path in sorted((root / 'selector' / variant).glob('*/test_*.json')):
        entry = json.loads(path.read_text())
        row, name = entry['row'], entry['dataset']
        tag = f'{variant}/{row}/{name}'
        try:
            bundle_path = path.parent / f'selector_{name}.pth'
            bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
            cell = Cell(row, row, name, path.parent / f'records_{name}.csv', bundle)
            check_cell(cell, entry['f1'])
            counts = {str(level): int((cell.level == level).sum()) for level in range(4)}
            assert all(count == entry['crops'] for count in counts.values()), counts
            assert len(set(zip(cell.crop, cell.level))) == len(cell.level), 'duplicated crop/level'
            assert all(np.isfinite(values).all() for values in cell.features.values()), 'non-finite features'
            for values in cell.counts.values():
                assert np.isfinite(values).all() and (values >= 0).all(), 'invalid confusion counts'
            reference = frozen_anchor(row, name)
            raw_delta = max(abs(entry['f1'][str(k)]['raw'] - reference[str(k)]['raw']) for k in range(4))
            assert raw_delta <= 1e-6, ('raw anchor mismatch', raw_delta)
            expected_bundle = torch.load(sources(row)[0] / 'bundles' / f'selector_{name}.pth',
                                         map_location='cpu', weights_only=False)
            frozen_records = sources(row)[1][['LEVIR-CD', 'DSIFN-CD', 'WHU-CD'].index(name)]
            frozen = Cell(row, row, name, frozen_records, expected_bundle)
            assert np.array_equal(cell.crop, frozen.crop) and np.array_equal(cell.level, frozen.level), 'sample order'
            assert np.array_equal(cell.counts['raw'], frozen.counts['raw']), 'raw per-crop counts'
            verified.append({'cell': tag, 'levels': counts, 'raw_anchor_max_difference': raw_delta,
                             'records_f1_and_selector_probability_passed': True})
        except (Exception, SystemExit) as exc:
            errors.append({'cell': tag, 'error': str(exc)})
result = {'verified': verified, 'errors': errors, 'verified_cells': len(verified)}
out = root / 'verification/completed_selector_records.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2), flush=True)
if errors:
    raise SystemExit(1)
