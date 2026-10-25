#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,numpy as np
T=Path(__file__).resolve().parents[2];R=T/'results/softmax-row224-m256-aba-r1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
ctl=read(T/'softmax-row224-m256-aba-r1-controller.json');assert ctl['status']=='complete' and ctl['returncode']==0
assert read(R/'meta.json')['status']=='complete';cfg=read(R/'config.json');assert cfg==read(T/'softmax-row224-m256-aba-r1-config.json')
for n,h in cfg['source_sha256'].items():assert sha(T/n)==h
assert sha(Path(cfg['candidate_status_path']))==cfg['candidate_status_sha256']
e=[json.loads(l) for l in (R/'events.jsonl').read_text().splitlines()];assert [x['event'] for x in e]==['start','result','start','result','start','result','complete']
rows=[];outputs={};meta=read(T/'fixtures-canonical/c512-m256-l0/probe-fixture.json');refpath=T/'fixtures-canonical/c512-m256-l0/probe-reference.f32';assert sha(refpath)==meta['reference_sha256'];ref=np.fromfile(refpath,dtype='<f4').astype(float)
for x in e:
 if x['event']!='result':continue
 case=x['case'];d=read(R/(case+'.json'));assert d==x['metrics'];assert d['released_bytes']==0 and d['repeats']==30 and d['warmup_repeats']==3 and d['scope']=='two_layer_probe'
 assert not d['npd_enabled'] and not d['io_profile_enabled'];assert (d['capacity'],d['query_rows'],d['fusion_layers'])==(512,256,2)
 assert d['mismatched_elements']==d['differing_repeats']==0
 f=R/(case+'.f32');assert f.stat().st_size==1441792 and sha(f)==x['output_sha256'];v=np.fromfile(f,dtype='<f4').astype(float);assert np.isfinite(v).all();outputs[case]=v
 t=np.array(d['api_seconds'])*1000;assert len(t)==30 and np.isfinite(t).all() and min(t)>0
 rows.append({'case':case,'p50_ms':float(np.median(t)),'p95_ms':float(np.quantile(t,.95)),'min_ms':float(min(t)),'max_ms':float(max(t)),'output_sha256':sha(f),'released_bytes':d['released_bytes']})
assert sha(R/'original-before.f32')==sha(R/'original-after.f32')==sha(T/'results/single-block-c512-r1/c512-m256-l0.f32')
a=outputs['original-before'];b=outputs['candidate'];segments=[];off=0
for name,n in meta['outputs']:
 x=a[off:off+n];y=b[off:off+n];z=ref[off:off+n];segments.append({'name':name,'elements':n,'changed_elements':int(np.sum(x!=y)),'max_abs_diff':float(np.max(np.abs(x-y))),'relative_l2_to_original':float(np.linalg.norm(y-x)/max(np.linalg.norm(x),1e-30)),'original_reference_rel_l2':float(np.linalg.norm(x-z)/max(np.linalg.norm(z),1e-30)),'candidate_reference_rel_l2':float(np.linalg.norm(y-z)/max(np.linalg.norm(z),1e-30))});off+=n
assert off==len(a)
r=[json.loads(l) for l in (R/'resources.jsonl').read_text().splitlines()]
result={'scope':'pure API single fused2 at history128; not full prefill/TTFT or natural quality','rows':rows,'before_after_bitwise_equal':True,'candidate_bitwise_equal':sha(R/'candidate.f32')==sha(R/'original-before.f32'),'candidate_to_before_p50_ratio':rows[1]['p50_ms']/rows[0]['p50_ms'],'after_to_before_p50_ratio':rows[2]['p50_ms']/rows[0]['p50_ms'],'segments':segments,'resource_samples':len(r),'max_temp_C':max(max(x['temperatures_mC'].values()) for x in r)/1000,'min_mem_available_B':min(int(x['memory']['MemAvailable'].split()[0])*1024 for x in r),'config_sha256':sha(R/'config.json'),'sources':cfg['source_sha256']}
(R/'ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
