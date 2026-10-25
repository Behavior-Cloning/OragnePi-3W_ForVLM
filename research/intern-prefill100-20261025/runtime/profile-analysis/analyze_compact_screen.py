#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,numpy as np
T=Path(__file__).resolve().parents[2];run='compact-past128-screen-aba-r1';R=T/'results'/run
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
ctl=read(T/(run+'-controller.json'));assert ctl['status']=='complete' and ctl['returncode']==0 and read(R/'meta.json')['status']=='complete';cfg=read(R/'config.json');assert cfg==read(T/(run+'-config.json'))
for n,h in {**cfg['source_sha256'],**cfg['fixture_metadata_sha256']}.items():assert sha(T/n)==h,n
assert sha(Path(cfg['candidate_status_path']))==cfg['candidate_status_sha256']
e=[json.loads(l) for l in (R/'events.jsonl').read_text().splitlines()];assert [x['event'] for x in e]==['start','result']*9+['complete'];ev={x['case']:x for x in e if x['event']=='result'};pairs=[]
for tokens,first in [(64,0),(128,0),(256,2)]:
 key=f'm{tokens}-l{first}';meta=read(T/'fixtures-canonical'/('c512-'+key)/'probe-fixture.json');rp=T/'fixtures-canonical'/('c512-'+key)/'probe-reference.f32';assert sha(rp)==meta['reference_sha256'];ref=np.fromfile(rp,dtype='<f4').astype(float);rows=[];out={}
 for label in ['before','compact','after']:
  name=key+'-'+label;d=read(R/(name+'.json'));assert d==ev[name]['metrics'];c=next(x for x in cfg['sequence'] if x['run_id']==name)
  assert (d['capacity'],d['query_rows'],d['fusion_layers'],d['scope'])==(c['capacity'],tokens,2,'two_layer_probe')
  assert d['repeats']==30 and d['warmup_repeats']==3 and d['released_bytes']==0 and d['mismatched_elements']==d['differing_repeats']==0
  assert not d['npd_enabled'] and not d['io_profile_enabled'] and d['resident_copies']==0
  f=R/(name+'.f32');assert sha(f)==ev[name]['output_sha256'] and f.stat().st_size==4*tokens*(896+4*128);v=np.fromfile(f,dtype='<f4').astype(float);assert np.isfinite(v).all();out[label]=v
  sec=np.array(d['api_seconds'])*1000;assert len(sec)==30 and np.isfinite(sec).all() and min(sec)>0
  rows.append({'case':name,'past_capacity':c['capacity'],'p50_ms':float(np.median(sec)),'p95_ms':float(np.quantile(sec,.95)),'max_ms':float(max(sec)),'sha256':sha(f),'release_bytes':d['released_bytes']})
 assert sha(R/(key+'-before.f32'))==sha(R/(key+'-after.f32'))==sha(T/'results/single-block-c512-r1'/('c512-'+key+'.f32'))
 segments=[];off=0
 for name,n in meta['outputs']:
  a,b,z=out['before'][off:off+n],out['compact'][off:off+n],ref[off:off+n];off+=n
  segments.append({'name':name,'changed_elements':int(np.sum(a!=b)),'max_abs':float(np.max(abs(a-b))),'relative_l2':float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-30)),'original_reference_relative_l2':float(np.linalg.norm(a-z)/max(np.linalg.norm(z),1e-30)),'compact_reference_relative_l2':float(np.linalg.norm(b-z)/max(np.linalg.norm(z),1e-30))})
 assert off==len(ref)
 pairs.append({'case':key,'first_layer':first,'query_rows':tokens,'effective_history':128,'rows':rows,'compact_to_original_ratio':rows[1]['p50_ms']/rows[0]['p50_ms'],'after_to_before_ratio':rows[2]['p50_ms']/rows[0]['p50_ms'],'original_AA_and_legacy_byte_equal':True,'segments':segments})
r=[json.loads(l) for l in (R/'resources.jsonl').read_text().splitlines()]
def vm(x):return {k:int(v) for k,v in (l.split() for l in x['vmstat'].splitlines())}
def swap(x):return (int(x['memory']['SwapTotal'].split()[0])-int(x['memory']['SwapFree'].split()[0]))*1024
res={'samples':len(r),'max_temp_C':max(max(x['temperatures_mC'].values()) for x in r)/1000,'min_mem_available_B':min(int(x['memory']['MemAvailable'].split()[0])*1024 for x in r),'swap_first_B':swap(r[0]),'swap_last_B':swap(r[-1]),'swapin_pages':vm(r[-1])['pswpin']-vm(r[0])['pswpin'],'swapout_pages':vm(r[-1])['pswpout']-vm(r[0])['pswpout']}
out={'scope':'270 pure vip_run_network calls, three two-layer ABA comparisons; not full prefill or TTFT','pairs':pairs,'resources':res,'source_sha256':cfg['source_sha256']};(R/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'pairs':[{k:v for k,v in p.items() if k!='segments'}|{'hidden_relative_l2':p['segments'][0]['relative_l2']} for p in pairs],'resources':res},indent=2))
