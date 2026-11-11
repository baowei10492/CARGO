"""Frozen WHU-CD evaluation with isolated scale, rotation and local distortions."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
import cv2,numpy as np

OLD=Path('/root/autodl-tmp/rs_cd_work/rs_cd_project/tools/round1_revision_20261103')
sys.path.insert(0,str(OLD))
import round1_formal as base
from round1_stress import StressGeometry,source_support

def geometries(item,protocol,size):
    from rs_cd.affine_registration import AffineRanges,affine_parameters,perturbation_matrix
    angle,scale,_,_=affine_parameters(item['crop_id'],3,AffineRanges(30,.80,1.25,25,3),protocol['seed'])
    unit=base.smooth_unit_field(protocol['datasets']['WHU-CD']['tile_size'],item['crop_id'],protocol['seed'],9)
    result={}
    for condition in protocol['E2']['conditions']:
        name=condition['name'];a=angle*condition.get('rotation_limit_deg',0)/30
        s=scale if condition.get('scale_only',False) else 1.
        rms=condition.get('backward_field_rms_px',0)
        matrix=perturbation_matrix(a,s,0.,0.,width=size,height=size)
        center=np.array([size/2,size/2,1.])
        assert np.allclose(matrix@center,center[:2],atol=1e-10)
        assert np.isclose(np.linalg.det(matrix[:,:2]),s*s,atol=1e-10)
        result[name]=(matrix,unit*rms if rms else None,{'rotation_deg':a,'scale':s,'added_translation_px':[0.,0.],'field_backward_rms_px':rms})
    return result

class PureEvaluation(base.Evaluation):
    def common_geometry(self,item):
        states=geometries(item,self.protocol,self.size)
        valid=base.scoring_mask(self.standard_matrices(item),self.size)
        outside={}
        for name,(matrix,field,meta) in states.items():
            covered,bounds=source_support(matrix,field,item,self.size,self.spec['tile_size'])
            if not covered:outside[name]=bounds
            if not outside:valid &= StressGeometry(None,item,matrix,self.size,field).inside_crop()
        return states,valid,outside

    def select_support(self):
        selected=[];excluded=[];counts=[]
        attempted=len(self.crops)
        for i,item in enumerate(self.crops,1):
            states,valid,outside=self.common_geometry(item)
            if outside:excluded.append({'crop_id':item['crop_id'],'reason':'out_of_tile_source','bounds':outside})
            elif not valid.any():excluded.append({'crop_id':item['crop_id'],'reason':'empty_common_score'})
            else:selected.append(item);counts.append(int(valid.sum()))
            if i%200==0 or i==attempted:print(f'GEOMETRY {i}/{attempted} eligible={len(selected)} excluded={len(excluded)}',flush=True)
        if not selected:raise RuntimeError('No common geometry support')
        report={'attempted_crops':attempted,'eligible_crops':len(selected),'eligible_tiles':len({x['sample_id'] for x in selected}),
                'excluded_crops':len(excluded),'selection_stage':'before loading model checkpoints or inspecting predictions',
                'rule':self.settings['coverage_policy'],'items':selected,'excluded':excluded,
                'selection_fingerprint':hashlib.sha256('\n'.join(x['crop_id'] for x in selected).encode()).hexdigest(),
                'scored_pixels':{'minimum':min(counts),'median':float(np.median(counts)),'maximum':max(counts),'total':sum(counts)}}
        path=self.root/'geometry_selection.json'
        if path.exists() and json.loads(path.read_text())!=report:raise RuntimeError('Geometry selection changed on resume')
        base.atomic_json(path,report);self.selection_report=report;self.crops=selected
        print('SELECTION_LOCKED',len(selected),'crops; no model checkpoint loaded',flush=True)

    def run_pure(self):
        newly=0
        for i,item in enumerate(self.crops,1):
            if not self.path(item).exists():
                t1,_,mask=self.cache.crop(item,self.size)
                states,valid,outside=self.common_geometry(item)
                if outside or not valid.any():raise RuntimeError('Locked geometry no longer valid')
                records={}
                for name,(matrix,field,meta) in states.items():
                    result=self.evaluate(t1,mask,valid,StressGeometry(self.cache.arrays[1],item,matrix,self.size,field))
                    result.update(meta);records[name]=result
                self.save_record(item,records);newly+=1
            self.progress(i,len(self.crops),newly)
        real_done=self.done;self.done=self.root/'SUMMARY_CORE_WRITTEN.json'
        try:super().summarize([c['name'] for c in self.settings['conditions']],self.settings['paths'])
        finally:self.done=real_done
        result=json.loads((self.root/'summary.json').read_text())
        result['geometry_selection']={k:v for k,v in self.selection_report.items() if k not in ['items','excluded']}
        native=result['conditions']['native']['f1']
        for state in result['conditions'].values():state['difference_from_native_pp']={k:100*(v-native[k]) for k,v in state['f1'].items()}
        result['interpretation']='Each non-native row adds only its named synthetic distortion to supplied pre-registered pairs. All rows and paths share crops and scored pixels. Separate from the compound L3 stress test.'
        base.atomic_json(self.root/'summary.json',result)
        with (self.root/'summary.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['condition','raw_f1_percent','always_f1_percent','cargo_f1_percent','adoption_percent','cargo_minus_raw_pp','ci95_low_pp','ci95_high_pp'])
            for name,x in result['conditions'].items():
                d=x['differences']['cargo_minus_raw'];w.writerow([name,*[100*x['f1'][p] for p in ['raw','estimated','cargo']],100*x['adoption_rate'],100*d['estimate'],*[100*v for v in d['ci95']]])
        base.atomic_json(self.done,{'status':'completed','stage':'E2','crops':len(self.crops),'identity':self.identity['id']})
        print('COMPLETE PURE_CONDITIONS',self.root,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.project=a.project.resolve();a.protocol=a.protocol.resolve();a.output=a.output.resolve();a.runner=Path(__file__).resolve();a.stage='E2'
    protocol=json.loads(a.protocol.read_text())
    for name,expected in protocol['pinned_helpers'].items():
        if base.digest(OLD/name)!=expected:raise RuntimeError('Pinned helper differs: '+name)
    os.chdir(a.project);sys.path[:0]=[str(a.project),str(a.project/'src')];base.runtime_imports()
    e=PureEvaluation(a)
    if e.done.exists():print('KEPT_COMPLETE',e.root);return
    e.select_support();e.load_models()
    with base.torch.inference_mode():e.run_pure()

if __name__=='__main__':main()
