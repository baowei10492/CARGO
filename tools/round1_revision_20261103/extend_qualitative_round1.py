"""Extend the approved qualitative layout using archived images/predictions."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
WORK=ROOT
ASSETS=WORK/'figures/qualitative_data'
ASSETS.mkdir(parents=True,exist_ok=True)
OLD=Path.cwd()/'reports/p31/final/20260919T134004Z/qualitative'
original=json.loads((OLD/'qualitative_index.json').read_text())
selected=['LEVIR-CD_corrected_1.png','DSIFN-CD_kept_clean_3.png','WHU-CD_corrected_2.png']
entries=[];sources={}
for name in selected:
    entry=next(e for e in original if e['file']==name)
    entry=json.loads(json.dumps(entry))
    for key,rel in entry['tiles'].items():
        source=OLD/rel;target=ASSETS/(Path(rel).name)
        shutil.copy2(source,target);entry['tiles'][key]=target.name
        sources[str(source)]={'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    entries.append(entry)

native=Path.cwd()/'reports/round1_20261103/E1/cases/case017_L0'
meta=json.loads((native/'COMPLETE.json').read_text())
arrays=np.load(native/'arrays.npz')
truth=arrays['label'].astype(bool);valid=arrays['valid'].astype(bool)
def error_map(pred):
    pred=pred.astype(bool);image=np.zeros((*truth.shape,3),dtype=np.uint8)
    image[truth & pred]=(255,255,255)
    image[~truth & pred]=(220,40,40)
    image[truth & ~pred]=(40,90,230)
    image[~valid]=(128,128,128)
    return image

entry={'file':'S2Looking_native_kept.png','dataset':'S2Looking','level':0,
       'selector_probability':meta['result']['selector_score'],'tau':.5,'tiles':{},'rendered_f1':{},
       'crop_id':meta['crop']['crop_id'],'source':'fixed hashed native case017, no added perturbation'}
for key,array_key in [('t1','t1'),('t2_input','t2'),('t2_corrected','registered_t2')]:
    value=arrays[array_key];target=ASSETS/f'S2Looking_native_kept_{key}.png';Image.fromarray(value).save(target);entry['tiles'][key]=target.name
ref=np.repeat((truth.astype(np.uint8)*255)[...,None],3,axis=2);ref[~valid]=(128,128,128)
Image.fromarray(ref).save(ASSETS/'S2Looking_native_kept_reference.png');entry['tiles']['reference']='S2Looking_native_kept_reference.png'
for key,predkey,countkey in [('without_correction','raw_probability','raw'),('localizer_only','registered_probability','estimated'),('with_selector','cargo_probability','cargo')]:
    pred=arrays[predkey]>=.5;counts=[int((valid & truth & pred).sum()),int((valid & ~truth & pred).sum()),int((valid & truth & ~pred).sum()),int((valid & ~truth & ~pred).sum())]
    assert counts==meta['result']['counts'][countkey],(key,counts)
    f1=2*counts[0]/(2*counts[0]+counts[1]+counts[2]);entry['rendered_f1'][key]=f1
    target=ASSETS/f'S2Looking_native_kept_{key}.png';Image.fromarray(error_map(pred)).save(target);entry['tiles'][key]=target.name
entries.append(entry)
sources[str(native/'arrays.npz')]={'sha256':hashlib.sha256((native/'arrays.npz').read_bytes()).hexdigest()}
(ASSETS/'qualitative_index.json').write_text(json.dumps(entries,indent=2)+'\n',encoding='utf-8')
script=WORK/'figures/src_data/plot_qualitative_round1.py'
subprocess.run([sys.executable,str(script)],check=True)
