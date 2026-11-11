"""CPU reanalysis: calibrate p_mis with the original held-out Eq.8 rule."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from rs_cd.p27_selector import calibrate_tau, f1_table, selection_rates
from tools.run_p27_selector import read_records
from tools.p31_selector_ablation import sources, DATASETS
from tools.p31_statistics import Cell, check_cell, draws, f1, resampled
from tools.p33_selector_runs import frozen_anchor
from tools.p33_statistics import assignment, tile_totals

torch.set_num_threads(3)
result = {'protocol': {'threshold_source': 'risk_calibration tiles only',
    'target_clean_drop': .0025, 'grid_start': .5, 'grid_stop': 1., 'grid_step': .01,
    'test_thresholds_frozen': True, 'repeats': 2000, 'seed': 20261009,
    'models_or_images_run': False}, 'cells': {}, 'mixed_WH U': {}, 'missing_calibration_records': []}
boot = {0: [], 3: []}
for row in ('dinov3', 'bit', 'cdlamba', 'cdxlstm'):
    train, paths = sources(row)
    for idx, dataset in enumerate(DATASETS):
        train_path = train/'features'/f'features_{dataset}.csv'
        if not train_path.exists():
            result['missing_calibration_records'].append(row+'/'+dataset)
            continue
        training = read_records(train_path)
        calibration = [r for r in training if r['split'] == 'risk_calibration']
        if not calibration: raise ValueError(f'No calibration rows: {row}/{dataset}')
        tau, sweep = calibrate_tau(calibration, [r['misaligned_probability'] for r in calibration],
            target=.0025, start=.5, stop=1., step=.01)
        records = read_records(paths[idx])
        bundle = torch.load(train/'bundles'/f'selector_{dataset}.pth', map_location='cpu', weights_only=False)
        c = Cell(row, row, dataset, paths[idx], bundle)
        anchor = frozen_anchor(row,dataset)
        check_cell(c,anchor)
        scores = np.array([r['misaligned_probability'] for r in records])
        table = f1_table(records,scores.tolist(),tau,('raw','selector'))
        rates = selection_rates(records,scores.tolist(),tau)
        counts = c.selected(scores,tau)
        sums = c.tile_sums(counts)
        orig = c.tile_sums(c.selected())
        weights = draws(c.tiles,2000,20261009+idx)
        for level in (0,3):boot[level].append(100*(f1(resampled(weights,sums[level]))-f1(resampled(weights,orig[level]))))
        entry = {'tau':None if math.isinf(tau) else tau,'calibration_rows':len(calibration),
            'calibration_clean_rows':sum(r['level']==0 for r in calibration),
            'calibration_sweep':sweep, 'test_f1':table,
            'test_adoption':{k:v['selector'] for k,v in rates.items()},
            'test_gains_pp':{k:100*(v['selector']-v['raw']) for k,v in table.items()},
            'cargo_gains_pp':{str(k):100*(anchor[str(k)]['selector']-anchor[str(k)]['raw']) for k in (0,3)},
            'records_f1_and_probability_checks_passed':True,
            'training_csv_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),
            'test_csv_sha256':hashlib.sha256(paths[idx].read_bytes()).hexdigest()}
        result['cells'][row+'/'+dataset]=entry
        if dataset=='WHU-CD':
            keep=assignment(c,3407,[0,1,3])
            result['mixed_WH U'][row]={key:float(f1(tile_totals(c,v,keep).sum(axis=0))) for key,v in
                [('raw',c.counts['raw']),('gate',counts),('cargo',c.selected())]}
vals=list(result['cells'].values())
result['mean_available']={'combinations':len(vals),'L0_change_pp':float(np.mean([x['test_gains_pp']['0'] for x in vals])),
 'L0_worst_pp':min(x['test_gains_pp']['0'] for x in vals),
 'L3_gain_pp':float(np.mean([x['test_gains_pp']['3'] for x in vals])),
 'degraded':sum(x['test_gains_pp']['0'] < -.5 for x in vals),
 'gate_minus_cargo_CI_pp':{str(level):np.percentile(np.mean(samples,axis=0),[2.5,97.5]).tolist() for level,samples in boot.items()}}
result['mixed_WHU']=result.pop('mixed_WH U')
print(json.dumps(result,indent=2))
