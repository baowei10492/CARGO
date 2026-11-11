"""Same-machine latency/memory: original DINOv3 checkpoints, uncached estimators.

Inputs are resident in host memory. One timed call includes registration in both
directions, coordinate setup, tile-context resampling, tensor assembly and H2D,
detector inference, selector features and selection. Disk I/O and checkpoint
loading are reported separately. Synthetic input creation is outside deployment
timing. No detector training or full test-set evaluation is performed.
"""
from __future__ import annotations
import argparse, json, math, os, statistics, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from rs_cd.affine_crops import TileCache, in_frame
from rs_cd.affine_f1 import scoring_mask
from rs_cd.affine_registration import affine_parameters, perturbation_matrix
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.p27_selector import (REGISTRATION_FEATURES, TASK_FEATURES, crop_records,
                               registration_features, task_features)
from rs_cd.p30_geometry import BorderFreeGeometry
from tools import run_p30
from tools.p33_selector_runs import configs, open_row, open_localizer
from tools.run_p27_selector import read_records, train_dataset, open_dataset
from tools.run_p27_step1 import sha256

IDENTITY=np.eye(2,3,dtype=np.float64)
METHODS=('cargo_full','cargo_registration_only','roma_selector')

class Selector:
    def __init__(self,bundle):
        self.names=bundle['feature_names']
        first,second=bundle['hidden_dims']
        self.model=nn.Sequential(nn.Linear(len(self.names),first),nn.GELU(),nn.LayerNorm(first),
            nn.Dropout(bundle['dropout']),nn.Linear(first,second),nn.GELU(),
            nn.Dropout(bundle['dropout']),nn.Linear(second,1)).eval()
        self.model.load_state_dict({k.removeprefix('network.'):v for k,v in bundle['model_state'].items()})
        self.mean=bundle['feature_mean'];self.std=bundle['feature_std']
        self.tau=math.inf if bundle['tau'] is None else bundle['tau']
    def predict(self,features):
        x=torch.tensor([[features[k] for k in self.names]],dtype=torch.float32)
        return float(torch.sigmoid(self.model((x-self.mean)/self.std)).item())

def stats(values):
    return {'n':len(values),'median_ms':float(np.median(values)),
            'mean_ms':float(np.mean(values)),'p10_ms':float(np.percentile(values,10)),
            'p90_ms':float(np.percentile(values,90))}

def timed(function,device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter();result=function();torch.cuda.synchronize(device)
    elapsed=(time.perf_counter()-start)*1000
    memory={'allocated_peak_MiB':torch.cuda.max_memory_allocated(device)/2**20,
            'reserved_peak_MiB':torch.cuda.max_memory_reserved(device)/2**20}
    return result,elapsed,memory

def sample_inputs(spec,h,p30,count):
    dataset=open_dataset(spec,'test');cache=TileCache(dataset,int(spec['tile_size']))
    crops=run_p30.margin_crops(dataset,spec,h['size'],int(p30['crop_margin']))
    # Deterministic fixed indices, no inspection of method performance.
    indexes=np.linspace(0,len(crops)-1,count,dtype=int)
    samples=[];base=[]
    for i in indexes:
        item=crops[int(i)];a,b,mask=cache.crop(item,h['size'])
        tile=cache.arrays[1].copy()
        geometry=BorderFreeGeometry(tile,item['top'],item['left'],h['size'])
        base.append({'item':item,'t1':a.copy(),'t2':b.copy(),'mask':mask.copy(),'tile':tile})
        for level in (0,3):
            truth=IDENTITY if level==0 else perturbation_matrix(*affine_parameters(item['crop_id'],level,h['ranges'],h['seed']),width=h['size'],height=h['size'])
            observed=geometry.perturbed_tile(truth).copy()
            t2=np.ascontiguousarray(observed[item['top']:item['top']+h['size'],item['left']:item['left']+h['size']])
            samples.append({'crop_id':item['crop_id'],'level':level,'t1':a.copy(),'t2':t2,
                            'observed_tile':observed,'top':item['top'],'left':item['left']})
    return samples,base

def pair_call(method,sample,localizer,detector,selector,h,device):
    a,b=sample['t1'],sample['t2']
    forward=localizer.estimate_many([a],[b])[0]
    backward=localizer.estimate_many([b],[a])[0]
    features=registration_features(forward.matrix,backward.matrix,forward.misaligned_probability,h['points'],h['size'])
    # Only observed image content and estimated transformation enter inference.
    geometry=BorderFreeGeometry(sample['observed_tile'],sample['top'],sample['left'],h['size'])
    if method=='cargo_registration_only':
        probability=selector.predict(features);use=probability>=selector.tau
        view=geometry.corrected(IDENTITY,forward.matrix) if use else b
        logits=detector['model'](normalize(a)[None].to(device),normalize(view)[None].to(device))['change_logits']
        return bool(use),logits
    view=geometry.corrected(IDENTITY,forward.matrix)
    t1=normalize(a)[None].repeat(2,1,1,1).to(device)
    t2=torch.stack([normalize(b),normalize(view)]).to(device)
    logits=detector['model'](t1,t2)['change_logits']
    valid=torch.from_numpy(scoring_mask([forward.matrix],h['size']).astype(np.float32))[None,None].to(device)
    task=task_features(logits[:1].float(),logits[1:].float(),valid,detector['threshold']).cpu().tolist()[0]
    features.update(zip(TASK_FEATURES,task));probability=selector.predict(features)
    use=probability>=selector.tau
    return bool(use),logits[1:2] if use else logits[:1]

@torch.inference_mode()
def benchmark(method,name,samples,base,localizer,detector,selector,h,spec,args,device):
    for i in range(args.warmup):
        pair_call(method,samples[i%len(samples)],localizer,detector,selector,h,device)
    torch.cuda.synchronize(device);torch.cuda.empty_cache()
    entries=[]
    for sample in samples:
        latencies=[];memories=[];adoptions=[]
        for repeat in range(args.repeats):
            value,ms,memory=timed(lambda:pair_call(method,sample,localizer,detector,selector,h,device),device)
            use,logits=value;adoptions.append(use);latencies.append(ms);memories.append(memory)
            del value,logits
        entries.append({'crop_id':sample['crop_id'],'level':sample['level'],'latencies_ms':latencies,
                        'adoption':adoptions,'peak_memory':memories})
        print(f'[M5 {name} {method} L{sample["level"]} {sample["crop_id"]}] median {np.median(latencies):.2f} ms',flush=True)
    block={'pairs':len(samples),'repeats_per_pair':args.repeats,'tau':None if math.isinf(selector.tau) else selector.tau,
           'entries':entries,'levels':{}}
    for level in (0,3):
        selected=[e for e in entries if e['level']==level]
        times=[t for e in selected for t in e['latencies_ms']]
        memories=[m for e in selected for m in e['peak_memory']]
        block['levels'][str(level)]={**stats(times),'adoption_fraction':float(np.mean([a for e in selected for a in e['adoption']])),
            'peak_allocated_MiB':max(m['allocated_peak_MiB'] for m in memories),
            'peak_reserved_MiB':max(m['reserved_peak_MiB'] for m in memories)}
    if not args.skip_adaptation and method!='cargo_registration_only':
        # Per source crop, actual feature-record computation used during adaptation:
        # all four levels and raw/oracle/registered detector branches. No fitting.
        times=[]
        for source in base[:2]:
            def precompute():
                geometry=BorderFreeGeometry(source['tile'],source['item']['top'],source['item']['left'],h['size'])
                return crop_records(localizer,detector['model'],threshold=detector['threshold'],
                    loss_weights=detector['loss'],dataset=name,item=source['item'],t1=source['t1'],t2=source['t2'],
                    mask=source['mask'],mask_threshold=int(spec['mask_threshold']),ranges=h['ranges'],levels=h['levels'],
                    seed=h['seed'],points=h['points'],tolerance=h['tolerance'],trigger_probability=h['trigger_probability'],
                    device=device,geometry=geometry)
            value,ms,_=timed(precompute,device);times.append(ms);del value
        block['adaptation_feature_precompute_per_source_crop']=stats(times)
        block['adaptation_feature_precompute_note']='Four perturbation states; measured subset, not a complete adaptation run. Six-feature selection can reuse the same records.'
    return block

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',default=['LEVIR-CD','DSIFN-CD','WHU-CD'])
    parser.add_argument('--crops-per-dataset',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--skip-adaptation',action='store_true')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda');rows,p30,evaluation,recipe=configs()
    from tools.run_p27_selector import harness
    h=harness(evaluation)
    result={'protocol':{'precision':'float32; TF32 disabled for all methods','device':torch.cuda.get_device_name(device),
        'gpu_properties':{'total_memory_MiB':torch.cuda.get_device_properties(device).total_memory/2**20},
        'torch_version':torch.__version__,'opencv_threads':4,'host_threads':4,'input_size':h['size'],
        'timer':'synchronized wall-clock; complete per-pair call, no sum of component medians',
        'input_loading':'tile content resident in host memory; file I/O and checkpoint loading excluded',
        'estimator_cache':'disabled; every repeat runs both registration directions',
        'checkpoint_or_threshold_changes':'none for full CARGO/RoMa; six-feature selector refit with existing original recipe',
        'sampling':'4 evenly spaced existing test crop indices per dataset; same inputs for all methods',
        'synthetic_perturbation_generation_timed':False,'repeats':args.repeats,'warmup':args.warmup},'datasets':{}}
    import cv2
    cv2.setNumThreads(4)
    trainroot=run_p30.pointed('reports/p30/selector/LATEST')
    for spec in evaluation['datasets']:
        name=spec['name']
        if name not in args.datasets:continue
        samples,base=sample_inputs(spec,h,p30,args.crops_per_dataset)
        detector=open_row('dinov3',name,rows,p30,device)
        bundles={'cargo_full':torch.load(trainroot/'bundles'/f'selector_{name}.pth',map_location='cpu',weights_only=False),
                 'roma_selector':torch.load(Path('reports/p33/selector/roma_ts_ransac/dinov3')/f'selector_{name}.pth',map_location='cpu',weights_only=False)}
        six_path=out.parent/f'six_features_{name}.pth'
        if not six_path.exists():
            fitstart=time.perf_counter()
            bundle,summary=train_dataset(name,read_records(trainroot/'features'/f'features_{name}.csv'),recipe,REGISTRATION_FEATURES)
            torch.save(bundle,six_path)
            summary['refit_wall_seconds']=time.perf_counter()-fitstart
            six_path.with_suffix('.json').write_text(json.dumps(summary,indent=2,default=str))
        bundles['cargo_registration_only']=torch.load(six_path,map_location='cpu',weights_only=False)
        block={'detector_sha256':detector['sha256'],'sample_crop_ids':[b['item']['crop_id'] for b in base],'methods':{}}
        for method in METHODS:
            start=time.perf_counter()
            localizer=open_localizer('roma_ts_ransac' if method=='roma_selector' else 'v2',p30,evaluation,device)
            if hasattr(localizer,'estimator') and 'Cached' in type(localizer.estimator).__name__:raise RuntimeError('Estimator cache is forbidden')
            if method=='roma_selector':
                sampler=localizer.estimator.sampler
                original_sample=sampler.sample
                def uncached_sample(*a,**kw):
                    sampler.cache.clear()
                    return original_sample(*a,**kw)
                sampler.sample=uncached_sample
                # RoMa uses module-level autocast flags; enforce the advertised
                # common FP32 protocol rather than relying on parameter dtype.
                for module in sampler.base.model.modules():
                    if hasattr(module,'amp'):module.amp=False
                    if hasattr(module,'amp_dtype'):module.amp_dtype=torch.float32
            load_seconds=time.perf_counter()-start
            block['methods'][method]=benchmark(method,name,samples,base,localizer,detector,Selector(bundles[method]),h,spec,args,device)
            block['methods'][method]['localizer_load_seconds_excluded']=load_seconds
            if method=='roma_selector':
                from tools import p27_localizer_ablation
                sampler.sample=original_sample
                p27_localizer_ablation._ROMA_SAMPLERS.clear()
                del sampler,original_sample,uncached_sample
            del localizer
            import gc
            gc.collect();torch.cuda.empty_cache()
            result['datasets'][name]=block;out.write_text(json.dumps(result,indent=2,default=str),encoding='utf8')
        del detector,bundles;torch.cuda.empty_cache()
    result['finished']=True;out.write_text(json.dumps(result,indent=2,default=str),encoding='utf8')
    print('[M5 COMPLETE] '+str(out),flush=True)

if __name__=='__main__':main()
