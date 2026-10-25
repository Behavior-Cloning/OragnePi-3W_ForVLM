#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,numpy as np
T=Path(__file__).resolve().parents[2];run='compact-past128-m256-abcd-a-r1';R=T/'results'/run
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ctl=read(T/(run+'-controller.json'));assert ctl['status']=='complete' and ctl['returncode']==0 and read(R/'meta.json')['status']=='complete'
cfg=read(R/'config.json');assert cfg==read(T/(run+'-config.json'))
for n,h in cfg['source_sha256'].items():assert sha(T/n)==h,n
assert sha(Path(cfg['candidate_status_path']))==cfg['candidate_status_sha256']
events=[json.loads(l) for l in (R/'events.jsonl').read_text().splitlines()];assert [x['event'] for x in events]==['start','result']*5+['complete']
refpath=T/'fixtures-canonical/c512-m256-l0/probe-reference.f32';meta=read(T/'fixtures-canonical/c512-m256-l0/probe-fixture.json');assert sha(refpath)==meta['reference_sha256'];ref=np.fromfile(refpath,dtype='<f4').astype(float)
outputs={};rows=[]
for e,c in zip([x for x in events if x['event']=='result'],cfg['sequence']):
 case=e['case'];assert case==c['run_id'];d=read(R/(case+'.json'));assert d==e['metrics']
 assert (d['capacity'],d['query_rows'],d['fusion_layers'],d['scope'])==(c['capacity'],256,2,'two_layer_probe')
 assert d['released_bytes']==0 and d['repeats']==30 and d['warmup_repeats']==3 and not d['npd_enabled'] and not d['io_profile_enabled'] and d['mismatched_elements']==d['differing_repeats']==0
 f=R/(case+'.f32');assert f.stat().st_size==1441792 and sha(f)==e['output_sha256'];v=np.fromfile(f,dtype='<f4').astype(float);assert np.isfinite(v).all();outputs[case]=v
 sec=np.array(d['api_seconds']);assert len(sec)==30 and np.isfinite(sec).all() and min(sec)>0
 rows.append({'case':case,'past_physical_capacity':d['capacity'],'query_rows':256,'p50_ms':float(np.median(sec)*1000),'p95_ms':float(np.quantile(sec,.95)*1000),'max_ms':float(max(sec)*1000),'output_sha256':sha(f),'released_bytes':d['released_bytes']})
assert sha(R/'original-before.f32')==sha(R/'original-after.f32')==sha(T/'results/single-block-c512-r1/c512-m256-l0.f32')
assert sha(R/'row224-c512.f32')==sha(T/'results/softmax-row224-m256-aba-r1/candidate.f32')
comparisons=[]
for left,right in [('original-before','plain-c128'),('row224-c512','row224-c128'),('original-before','row224-c128'),('plain-c128','row224-c128')]:
 a,b=outputs[left],outputs[right];segments=[];off=0
 for name,n in meta['outputs']:
  x,y,z=a[off:off+n],b[off:off+n],ref[off:off+n];segments.append({'name':name,'changed_elements':int(np.sum(x!=y)),'max_abs_diff':float(np.max(abs(x-y))),'relative_l2':float(np.linalg.norm(y-x)/max(np.linalg.norm(x),1e-30)),'left_reference_rel_l2':float(np.linalg.norm(x-z)/max(np.linalg.norm(z),1e-30)),'right_reference_rel_l2':float(np.linalg.norm(y-z)/max(np.linalg.norm(z),1e-30))});off+=n
 assert off==len(a);comparisons.append({'left':left,'right':right,'output_bitwise_equal':sha(R/(left+'.f32'))==sha(R/(right+'.f32')),'segments':segments})
r=[json.loads(l) for l in (R/'resources.jsonl').read_text().splitlines()]
def swap(x):return (int(x['memory']['SwapTotal'].split()[0])-int(x['memory']['SwapFree'].split()[0]))*1024
def vm(x):return {k:int(v) for k,v in (l.split() for l in x['vmstat'].splitlines())}
resources={'samples':len(r),'max_temp_C':max(max(x['temperatures_mC'].values()) for x in r)/1000,'min_mem_available_B':min(int(x['memory']['MemAvailable'].split()[0])*1024 for x in r),'swap_used_first_B':swap(r[0]),'swap_used_last_B':swap(r[-1]),'swapin_delta_pages':vm(r[-1])['pswpin']-vm(r[0])['pswpin'],'swapout_delta_pages':vm(r[-1])['pswpout']-vm(r[0])['pswpout']}
out={'scope':'five-group two-layer same effective history128/query256; NPU API only, not full prefill/TTFT','rows':rows,'comparisons':comparisons,'original_AA_output_equal':True,'new_probe_old_graph_outputs_match_archived_old_probe':True,'resources':resources,'source_sha256':cfg['source_sha256']}
(R/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':rows,'resources':resources,'comparisons':[{'left':x['left'],'right':x['right'],'bitwise':x['output_bitwise_equal'],'hidden_rel_l2':x['segments'][0]['relative_l2']} for x in comparisons]},indent=2))
