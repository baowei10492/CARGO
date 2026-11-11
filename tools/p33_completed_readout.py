import json
from pathlib import Path
import numpy as np
import torch
from tools.p31_statistics import Cell, check_cell, draws, paired, f1, resampled
from tools.p31_selector_ablation import sources
from tools.p33_selector_runs import frozen_anchor

root = Path('reports/p33')
result = {}

# Tile bootstrap of the completed S2Looking run. No model inference is performed.
p = root / 'selector/v2/dinov3_s2looking/test_S2Looking.json'
if p.is_file():
    e = json.loads(p.read_text())
    out = p.parent
    b = torch.load(out / 'selector_S2Looking.pth', map_location='cpu', weights_only=False)
    c = Cell('dinov3_s2looking', 'DINOv3', 'S2Looking', out / 'records_S2Looking.csv', b)
    check_cell(c, e['f1'])
    ci = paired(c, c.selected(), c.counts['raw'], draws(c.tiles, 2000, 20261009))
    result['s2looking'] = {'f1': e['f1'], 'ci_selector_minus_raw': ci,
                           'adoption': e['selection_rate'], 'tiles': c.tiles,
                           'crops': e['crops'], 'detector_sha256': e['detector_sha256']}

rows = ('dinov3', 'bit', 'cdlamba', 'cdxlstm')
names = ('LEVIR-CD', 'DSIFN-CD', 'WHU-CD')
for variant in ('roma_ts_ransac', 'v2r_conf_only', 'v2r_no_change_label'):
    cells = []
    samples = []
    for row in rows:
        for name in names:
            p = root / 'selector' / variant / row / f'test_{name}.json'
            if not p.is_file():
                continue
            e = json.loads(p.read_text())
            a = frozen_anchor(row, name)
            v = {str(k): 100 * (e['f1'][str(k)]['selector'] - e['f1'][str(k)]['raw']) for k in (0, 3)}
            ref = {str(k): 100 * (a[str(k)]['selector'] - a[str(k)]['raw']) for k in (0, 3)}
            cells.append({'row': row, 'dataset': name, 'variant_gain_pp': v, 'paper_gain_pp': ref})
            if variant.startswith('v2r'):
                b = torch.load(p.parent / f'selector_{name}.pth', map_location='cpu', weights_only=False)
                c = Cell(row, row, name, p.parent / f'records_{name}.csv', b)
                check_cell(c, e['f1'])
                train, paths = sources(row)
                fb = torch.load(train / 'bundles' / f'selector_{name}.pth', map_location='cpu', weights_only=False)
                fc = Cell(row, row, name, paths[names.index(name)], fb)
                check_cell(fc, a)
                assert np.array_equal(c.crop, fc.crop) and np.array_equal(c.level, fc.level)
                assert np.array_equal(c.tile, fc.tile)
                assert np.array_equal(c.counts['raw'], fc.counts['raw'])
                w = draws(c.tiles, 2000, 20261009 + names.index(name))
                sa, sb = c.tile_sums(c.selected())[3], fc.tile_sums(fc.selected())[3]
                samples.append(100 * (f1(resampled(w, sa)) - f1(resampled(w, sb))))
    if cells:
        block = {'cells': cells,
                 'mean_variant_gain_pp': {k: float(np.mean([c['variant_gain_pp'][k] for c in cells])) for k in ('0', '3')},
                 'mean_paper_gain_pp': {k: float(np.mean([c['paper_gain_pp'][k] for c in cells])) for k in ('0', '3')}}
        if samples:
            delta = np.mean(samples, axis=0)
            block['l3_variant_minus_full_ci_pp'] = [float(v) for v in np.percentile(delta, [2.5, 97.5])]
            block['records_and_raw_anchor_checks_passed'] = True
        result[variant] = block
print(json.dumps(result, indent=2))
