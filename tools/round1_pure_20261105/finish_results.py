"""Validate result counts and create a small log/result archive before shutdown."""
from pathlib import Path
import argparse,csv,hashlib,json,math,tarfile,time

def finish(root,status):
    root=Path(root).resolve();stage=root/'E2';details={'queue_status':status,'validated':False}
    if status=='completed':
        summary=json.loads((stage/'summary.json').read_text());complete=json.loads((stage/'COMPLETE.json').read_text())
        selection=json.loads((stage/'geometry_selection.json').read_text())
        names=list(summary['conditions']);totals={n:{p:[0]*4 for p in ['raw','estimated','cargo']} for n in names}
        records=list((stage/'records').glob('*.json'));assert len(records)==complete['crops']==summary['crops']==selection['eligible_crops']
        for path in records:
            x=json.loads(path.read_text());assert set(x['conditions'])==set(names);pixels=set()
            for n,v in x['conditions'].items():
                pixels.add(v['valid_pixels']);assert v['counts']['cargo']==v['counts']['estimated' if v['adopted'] else 'raw']
                assert v['added_translation_px']==[0.,0.]
                if n.startswith('rotation'):assert v['scale']==1. and v['field_backward_rms_px']==0
                if n=='scale_only':assert v['rotation_deg']==0 and v['field_backward_rms_px']==0
                if n=='local6':assert v['rotation_deg']==0 and v['scale']==1.
                for p,counts in v['counts'].items():
                    assert len(counts)==4 and all(isinstance(c,int) and c>=0 for c in counts) and sum(counts)==v['valid_pixels']
                    totals[n][p]=[a+b for a,b in zip(totals[n][p],counts)]
            assert len(pixels)==1
        for n,paths in totals.items():
            for p,(tp,fp,fn,tn) in paths.items():
                value=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
                assert math.isclose(value,summary['conditions'][n]['f1'][p],abs_tol=1e-12)
        details.update(validated=True,crops=len(records),conditions=names,common_pixels_verified=True,pooled_f1_recomputed=True)
    (root/'validation.json').write_text(json.dumps(details,indent=2)+'\n')
    state={'status':status,'validated':details['validated'],'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (root/'queue_status.json').write_text(json.dumps(state,indent=2)+'\n')
    archive=root.parent/'pure_conditions_20261105_results.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tf:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name not in ['queue.lock','queue.pid','LOCAL_BACKUP_VERIFIED.json','TRANSFER_READY.json']:tf.add(p,arcname=str(Path('pure_conditions')/p.relative_to(root)))
    ready={**state,'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    (root/'TRANSFER_READY.json').write_text(json.dumps(ready,indent=2)+'\n')
    print('TRANSFER_READY',json.dumps(ready),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('status');a=p.parse_args();finish(a.root,a.status)
