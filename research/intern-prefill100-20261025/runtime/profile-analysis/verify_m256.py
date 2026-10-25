#!/usr/bin/env python3
"""Rehash retained M256 profiling files before any diagnostic interpretation."""
from pathlib import Path
import hashlib,json,numpy as np
T=Path(__file__).resolve().parents[2];R=T/'results/profile-m256-l0-r1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
ctl=read(T/'profile-m256-l0-r1-controller.json');assert ctl['status']=='complete' and ctl['returncode']==0
assert read(R/'meta.json')['status']=='complete'
cfg=read(R/'config.json');assert cfg==read(T/'profile-m256-l0-r1-config.json')
for n,h in cfg['source_sha256'].items():assert sha(T/n)==h,n
assert sha(T/'single-block-c512-r1-config.json')==cfg['source_config_sha256']
assert sha(T/'graphs/cases/c512-m256-l0/export_nbg_unify/network_binary.nb')==cfg['sequence'][0]['nb_sha256']
assert sha(T/'fixtures-canonical/c512-m256-l0/probe-input.f32')==cfg['sequence'][0]['input_sha256']
events=[json.loads(l) for l in (R/'events.jsonl').read_text().splitlines()]
assert [e['event'] for e in events]==['start','result','start','result','complete']
base=T/'results/single-block-c512-r1/c512-m256-l0.f32'
old=[json.loads(l) for l in (T/'results/single-block-c512-r1/events.jsonl').read_text().splitlines()]
baseline=next(e for e in old if e['event']=='result' and e['case']=='c512-m256-l0')
assert sha(base)==baseline['output_sha256']
assert base.stat().st_size==1441792 and np.isfinite(np.fromfile(base,dtype='<f4')).all()
rows=[]
for e in events:
 if e['event']!='result':continue
 folder=R/e['case'];d=read(folder/'result.json');assert d==e['metrics']
 assert e['timing_eligible'] is False and d['released_bytes']==0
 assert (d['capacity'],d['query_rows'],d['fusion_layers'],d['scope'])==(512,256,2,'two_layer_probe')
 assert (d['repeats'],d['warmup_repeats'])==((5,3) if e['mode']=='io' else (1,0))
 assert d['npd_enabled']==(e['mode']=='npd') and d['io_profile_enabled']==(e['mode']=='io')
 for a in e['artifacts']:
  f=R/a['path'];assert f.stat().st_size==a['bytes'] and sha(f)==a['sha256'],str(f)
 f=folder/'output.f32';assert f.stat().st_size==e['output_bytes']==1441792
 assert sha(f)==e['output_sha256'] and np.isfinite(np.fromfile(f,dtype='<f4')).all()
 assert sha(f)==sha(base),'Profile output changed from pure baseline'
 npds=list(folder.glob('*.npd'));assert len(npds)==(1 if e['mode']=='npd' else 0)
 rows.append({'case':e['case'],'mode':e['mode'],'output_sha256':sha(f),'output_bytes':f.stat().st_size,'all_output_byte_equal':True,'npd_files':[{'path':str(x.relative_to(T)),'sha256':sha(x),'bytes':x.stat().st_size} for x in npds],'release_bytes':d['released_bytes'],'timing_eligible':False})
out={'scope':'M256 two-layer independent diagnostics; no full model timing claim','all_outputs_bitwise_equal_pure_baseline':True,'baseline_sha256':sha(base),'rows':rows,'source_sha256':cfg['source_sha256'],'controller_sha256':sha(T/'profile-m256-l0-r1-controller.json')}
(R/'IO_OUTPUT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
