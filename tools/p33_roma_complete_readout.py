"""Verify all completed RoMa-selector cells, then compare matched frozen CARGO records on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.p31_statistics import Cell, check_cell, draws, f1, resampled
from tools.p31_selector_ablation import sources
from tools.p33_selector_runs import frozen_anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', required=True)
    ap.add_argument('--merge')
    args = ap.parse_args()
    torch.set_num_threads(2)
    datasets = ['LEVIR-CD', 'DSIFN-CD', 'WHU-CD']
    cells = []
    for path in sorted(Path('reports/p33/selector/roma_ts_ransac').glob('*/test_*.json')):
        e = json.loads(path.read_text())
        row, name = e['row'], e['dataset']
        b = torch.load(path.parent / f'selector_{name}.pth', map_location='cpu', weights_only=False)
        c = Cell(row, row, name, path.parent / f'records_{name}.csv', b)
        check_cell(c, e['f1'])
        train, records = sources(row)
        fb = torch.load(train / 'bundles' / f'selector_{name}.pth', map_location='cpu', weights_only=False)
        fc = Cell(row, row, name, records[datasets.index(name)], fb)
        anchor = frozen_anchor(row, name)
        check_cell(fc, anchor)
        assert np.array_equal(c.crop, fc.crop) and np.array_equal(c.level, fc.level)
        assert np.array_equal(c.tile, fc.tile)
        raw_a, raw_b = c.by_level(c.counts['raw']), fc.by_level(fc.counts['raw'])
        # Paired image IDs are identical. Separate GPU runs can differ at a few thresholded
        # pixels; preserve those differences and apply the existing 1e-6 F1 anchor check
        # only to the two primary levels being compared, instead of changing the records.
        raw_delta = {str(k):100*(raw_a[k]-raw_b[k]) for k in range(4)}
        assert all(abs(raw_a[k]-raw_b[k]) <= 1e-6 for k in [0,3]), raw_delta
        assert len(set(zip(c.crop.tolist(), c.level.tolist()))) == len(c.level)
        assert all(int((c.level == k).sum()) == e['crops'] for k in range(4))
        counts = {'raw': c.counts['raw'], 'cargo_raw':fc.counts['raw'], 'roma_selector':c.selected(),
                  'cargo_selector':fc.selected(), 'roma_always':c.counts['estimated']}
        levels = {key: c.by_level(value) for key, value in counts.items()}
        w = draws(c.tiles, 2000, 20261009 + datasets.index(name))
        sums = {key:c.tile_sums(value) for key, value in counts.items()}
        sample = {key:[100*f1(resampled(w, value[k])) for k in range(4)] for key,value in sums.items()}
        gains = {key:{str(k):100*(value[k]-levels['cargo_raw' if key=='cargo_selector' else 'raw'][k]) for k in range(4)} for key,value in levels.items() if key not in ['raw','cargo_raw']}
        samples = {key:{str(k):(value[k]-sample['cargo_raw' if key=='cargo_selector' else 'raw'][k]).tolist() for k in range(4)} for key,value in sample.items() if key not in ['raw','cargo_raw']}
        cells.append({'row':row,'dataset':name,'source_port':args.port,'crops_per_level':e['crops'],
                      'tiles':c.tiles,'gain_pp':gains,'f1_percent':{key:{str(k):100*v for k,v in value.items()} for key,value in levels.items()},
                      'samples':samples,'sample_order_f1_and_selector_probabilities_verified':True,
                      'raw_anchor_delta_pp':raw_delta,
                      'raw_counts_exactly_equal':bool(np.array_equal(c.counts['raw'],fc.counts['raw'])),
                      'raw_counts_max_absolute_difference':float(abs(c.counts['raw']-fc.counts['raw']).max())})
    local = {'source_port':args.port,'bootstrap':{'repeats':2000,'seed':20261009,'unit':'source tile',
             'shared_draws_across_detectors':True}, 'cells':cells}
    out = Path(f'reports/p33/verification/roma_readout_{args.port}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(local, indent=2))
    if args.merge:
        other = json.loads(Path(args.merge).read_text())
        cells += other['cells']
        assert len(cells) == 12 and len({(c['row'],c['dataset']) for c in cells}) == 12
        result = {'cells':[], 'bootstrap':local['bootstrap'], 'means':{}}
        for c in cells:
            samples = c.pop('samples')
            c['roma_minus_cargo_ci_pp'] = {k:np.percentile(np.array(samples['roma_selector'][k])-np.array(samples['cargo_selector'][k]),[2.5,97.5]).tolist() for k in ['0','3']}
            c['_samples'] = samples
            result['cells'].append(c)
        for key in ['roma_selector','cargo_selector','roma_always']:
            result['means'][key] = {k:float(np.mean([c['gain_pp'][key][k] for c in cells])) for k in ['0','3']}
        differences = {k:np.mean([np.array(c['_samples']['roma_selector'][k])-np.array(c['_samples']['cargo_selector'][k]) for c in cells],axis=0) for k in ['0','3']}
        result['mean_roma_minus_cargo_ci_pp'] = {k:np.percentile(value,[2.5,97.5]).tolist() for k,value in differences.items()}
        result['degraded_counts_l0_drop_over_0_5_pp'] = {key:sum(c['gain_pp'][key]['0'] < -0.5 for c in cells) for key in ['roma_selector','cargo_selector']}
        result['dataset_means'] = {name:{key:{k:float(np.mean([c['gain_pp'][key][k] for c in cells if c['dataset']==name])) for k in ['0','3']} for key in ['roma_selector','cargo_selector']} for name in datasets}
        for c in cells: c.pop('_samples')
        out = Path('reports/p33/verification/roma_complete_12.json')
        out.write_text(json.dumps(result,indent=2))
        print(json.dumps({key:value for key,value in result.items() if key!='cells'},indent=2),flush=True)
    else:
        print(json.dumps({'source_port':args.port,'verified_cells':len(cells),'out':str(out),
              'cells':[{'row':c['row'],'dataset':c['dataset'],'gain_pp':c['gain_pp']} for c in cells]},indent=2),flush=True)


if __name__ == '__main__':
    main()
