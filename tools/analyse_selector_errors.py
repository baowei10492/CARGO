"""Reanalyse frozen predictions; no image inference, fitting, or checkpoint changes."""
from pathlib import Path
import csv, json, hashlib, math
import numpy as np
import torch
from torch import nn

import os
PROJECT = Path.cwd()
ROOT = PROJECT / 'reports/selector_errors'
ROOT.mkdir(parents=True, exist_ok=True)

BACKUP = Path(os.environ.get('CARGO_ARCHIVE_PROJECT', str(PROJECT)))
NAMES = ('LEVIR-CD', 'DSIFN-CD', 'WHU-CD')
ROWS = ('dinov3', 'bit', 'cdlamba', 'cdxlstm')
torch.set_num_threads(2)

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def f1(counts):
    tp, fp, fn = np.moveaxis(np.asarray(counts, dtype=float), -1, 0)
    p = tp/(tp+fp+1e-12); r = tp/(tp+fn+1e-12)
    return 2*p*r/(p+r+1e-12)

def vectors(records):
    return {p: np.array([[float(r[f'{p}_{k}']) for k in ('tp','fp','fn')] for r in records])
            for p in ('raw','estimated')}

def score(bundle, records):
    first, second = bundle['hidden_dims']
    model = nn.Sequential(nn.Linear(len(bundle['feature_names']), first), nn.GELU(), nn.LayerNorm(first),
                          nn.Dropout(bundle['dropout']), nn.Linear(first, second), nn.GELU(),
                          nn.Dropout(bundle['dropout']), nn.Linear(second, 1))
    model.load_state_dict({k.removeprefix('network.'):v for k,v in bundle['model_state'].items()})
    model.eval()
    x = torch.tensor([[float(r[k]) for k in bundle['feature_names']] for r in records], dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model((x-bundle['feature_mean'])/bundle['feature_std'])).reshape(-1).numpy()

def test_paths(row):
    if row=='dinov3':
        root=PROJECT/'reports/p30/test/20260918T134224Z'
        anchors=json.loads((root/'p30_test.json').read_text())['datasets']
        return [(n,root/f'records_proposed_{n}.csv',anchors[n]['proposed']) for n in NAMES]
    root=PROJECT/'reports/p31'/row/'test'
    anchors=json.loads((root/'p31_test.json').read_text())['datasets']
    return [(n,root/f'records_{n}.csv',anchors[n]) for n in NAMES]

result={'protocol':{'frozen_test_predictions':True, 'models_fitted':False,
                    'pixel_harm_definition':'estimated FP+FN strictly exceeds raw FP+FN',
                    'loss_harm_definition':'estimated BCE+Dice strictly exceeds raw BCE+Dice',
                    'calibration_bootstrap_seed':20261010, 'calibration_tile_resamples':100},
        'cells':{}, 'aggregate':{}, 'calibration':{}}
pool={level:[] for level in range(4)}
for row in ROWS:
    for name,path,anchor in test_paths(row):
        records=read(path); v=vectors(records)
        probability=np.array([float(r['selector_probability']) for r in records])
        tau=math.inf if anchor['tau'] is None else anchor['tau']
        adopted=probability>=tau
        selected=np.where(adopted[:,None],v['estimated'],v['raw'])
        levels=np.array([int(r['level']) for r in records])
        valid=np.array([float(r['valid_pixels']) for r in records])
        pixel_harm=v['estimated'][:,1:].sum(axis=1)>v['raw'][:,1:].sum(axis=1)
        loss_harm=np.array([float(r['estimated_loss'])>float(r['raw_loss']) for r in records])
        block={}
        for level in range(4):
            keep=levels==level; n=int(keep.sum()); acc=adopted&keep
            got=float(f1(selected[keep].sum(axis=0)))
            assert abs(got-anchor['f1'][str(level)]['selector'])<1e-6,(row,name,level)
            raw=float(f1(v['raw'][keep].sum(axis=0)))
            assert abs(raw-anchor['f1'][str(level)]['raw'])<1e-6
            delta=selected[keep].sum(axis=0)-v['raw'][keep].sum(axis=0)
            b={'records':n,'adopted':int(acc.sum()),'pixel_harmful_adopted':int((acc&pixel_harm).sum()),
               'loss_harmful_adopted':int((acc&loss_harm).sum()),'raw_f1':raw,'selector_f1':got,
               'delta_tp_fp_fn':delta.tolist(),'scored_pixels':int(valid[keep].sum())}
            b['pixel_harmful_adopted_percent']=100*b['pixel_harmful_adopted']/b['adopted'] if b['adopted'] else None
            b['loss_harmful_adopted_percent']=100*b['loss_harmful_adopted']/b['adopted'] if b['adopted'] else None
            block[str(level)]=b; pool[level].append(b)
        result['cells'][row+'/'+name]={'tau':anchor['tau'],'levels':block,
                                       'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
for level,blocks in pool.items():
    total={k:sum(b[k] for b in blocks) for k in ('records','adopted','pixel_harmful_adopted','loss_harmful_adopted','scored_pixels')}
    total['delta_tp_fp_fn']=np.sum([b['delta_tp_fp_fn'] for b in blocks],axis=0).tolist()
    total['adoption_percent']=100*total['adopted']/total['records']
    for key in ('pixel','loss'):
        total[key+'_harmful_adopted_percent']=100*total[key+'_harmful_adopted']/total['adopted']
    total['delta_tp_fp_fn_per_1000_scored_pixels']=(1000*np.array(total['delta_tp_fp_fn'])/total['scored_pixels']).tolist()
    result['aggregate'][str(level)]=total

# Keep the trained scores fixed. Calibrate from held-out level-0 records only,
# exactly replaying the original first-feasible threshold grid and fallback.
all_test_changes=[]
for idx,name in enumerate(NAMES):
    trainroot=BACKUP/'reports/p30/selector/20260918T104133Z'
    path=trainroot/'features'/f'features_{name}.csv'
    records=[r for r in read(path) if r['split']=='risk_calibration']
    bundle=torch.load(trainroot/'bundles'/f'selector_{name}.pth',map_location='cpu',weights_only=False)
    p=score(bundle,records); v=vectors(records)
    keep=np.array([int(r['level'])==0 for r in records])
    rc=v['raw'][keep]; ec=v['estimated'][keep]; pc=p[keep]
    tiles=np.array([r['tile_id'] for r in records])[keep]
    unique=sorted(set(tiles)); ti={s:i for i,s in enumerate(unique)}
    raw_tiles=np.zeros((len(unique),3)); np.add.at(raw_tiles,[ti[s] for s in tiles],rc)
    grid=np.arange(50,101)/100
    choose=pc[None,:]>=grid[:,None]
    counts=np.where(choose[:,:,None],ec[None,:,:],rc[None,:,:])
    selected_tiles=np.zeros((len(grid),len(unique),3))
    for i in range(len(grid)):np.add.at(selected_tiles[i],[ti[s] for s in tiles],counts[i])
    raw_f1=float(f1(rc.sum(axis=0)))
    drops=raw_f1-f1(counts.sum(axis=1))
    def calibration(w, tolerance):
        raw=float(f1(w@raw_tiles))
        cand=f1(np.einsum('t,gtc->gc',w,selected_tiles))
        indices=np.flatnonzero(raw-cand<=tolerance/100)
        return float(grid[indices[0]]) if len(indices) else math.inf
    tau=calibration(np.ones(len(unique)),.25)
    expected=math.inf if bundle['tau'] is None else bundle['tau']
    assert tau==expected,(name,tau,expected)
    _,test_path,anchor=next(x for x in test_paths('dinov3') if x[0]==name)
    test=read(test_path); tv=vectors(test); tp=np.array([float(r['selector_probability']) for r in test])
    recomputed=score(bundle,test)
    assert np.max(abs(recomputed-tp))<1e-4, name
    lev=np.array([int(r['level']) for r in test])
    def test_delta(tau):
        chosen=np.where((tp>=tau)[:,None],tv['estimated'],tv['raw'])
        return {str(l):100*float(f1(chosen[lev==l].sum(axis=0))-f1(tv['raw'][lev==l].sum(axis=0))) for l in (0,3)}
    tolerance={str(t):{'tau':None if math.isinf(calibration(np.ones(len(unique)),t)) else calibration(np.ones(len(unique)),t),
                        'test_change_pp':test_delta(calibration(np.ones(len(unique)),t))} for t in (0,.1,.25,.5)}
    rng=np.random.default_rng(20261010+idx)
    taus=[]; changes=[]
    for i in range(100):
        w=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))
        rt=calibration(w,.25); taus.append(rt);changes.append(test_delta(rt))
    samples=np.array([[c['0'],c['3']] for c in changes]);all_test_changes.append(samples)
    finite=[t for t in taus if not math.isinf(t)]
    result['calibration'][name]={'tiles':len(unique),'clean_records':int(keep.sum()),'original_tau':bundle['tau'],
        'reproduces_original_threshold_and_scores':True,'tolerance_sweep':tolerance,
        'resampled_tau_counts':{str(t):taus.count(t) for t in sorted(set(taus))},
        'resampled_test_change_pp_percentiles_5_50_95':np.percentile(samples,[5,50,95],axis=0).tolist(),
        'note':'Empirical sensitivity at fixed checkpoints; percentiles are not training-seed confidence intervals.',
        'training_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
result['mean_calibration_resample_test_change_pp_percentiles_5_50_95']=np.percentile(np.mean(all_test_changes,axis=0),[5,50,95],axis=0).tolist()
(ROOT/'selector_error_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'aggregate':result['aggregate'],'calibration':result['calibration'],
                  'mean_calibration_resample_test_change_pp_percentiles_5_50_95':result['mean_calibration_resample_test_change_pp_percentiles_5_50_95']},ensure_ascii=False,indent=2))
