from pathlib import Path
import hashlib,json,tarfile,numpy as np
T=Path(__file__).resolve().parents[2];run='compact-gateup-a8-m64-r1';R=T/'results'/run
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
ctl=read(T/(run+'-controller.json'));assert ctl['status']=='complete' and ctl['returncode']==0 and read(R/'meta.json')['status']=='complete'
a=T/(run+'.tar.gz');assert sha(a)==ctl['archive_sha256']
with tarfile.open(a) as tf:
 for m in tf.getmembers():
  if m.isfile():assert hashlib.sha256(tf.extractfile(m).read()).hexdigest()==sha(R/m.name.removeprefix('./'))
cfg=read(R/'config.json');assert cfg==read(T/(run+'-config.json'))
for n,h in {**cfg['source_sha256'],**cfg['fixture_metadata_sha256']}.items():assert sha(T/n)==h,n
assert sha(Path(cfg['candidate_status_path']))==cfg['candidate_status_sha256']
e=[json.loads(l) for l in (R/'events.jsonl').read_text().splitlines()];assert [x['event'] for x in e]==['start','result']*4+['complete'];ev={x['case']:x for x in e if x['event']=='result'}
meta=read(T/'fixtures-canonical/c512-m64-l0/probe-fixture.json');rows=[];outputs={}
for label in ['before','range0','range1','after']:
 d=read(R/(label+'.json'));assert d==ev[label]['metrics']
 assert (d['capacity'],d['query_rows'],d['fusion_layers'],d['scope'])==(128,64,2,'two_layer_probe')
 assert d['repeats']==30 and d['warmup_repeats']==3 and d['released_bytes']==0 and d['mismatched_elements']==d['differing_repeats']==0
 assert not d['npd_enabled'] and not d['io_profile_enabled'] and d['resident_copies']==0
 f=R/(label+'.f32');assert sha(f)==ev[label]['output_sha256'] and f.stat().st_size==4*64*(896+4*128)
 v=np.fromfile(f,dtype='<f4').astype(float);assert np.isfinite(v).all();outputs[label]=v
 ms=np.array(d['api_seconds'])*1000;assert len(ms)==30 and np.isfinite(ms).all() and min(ms)>0
 rows.append({'case':label,'p50_ms':float(np.median(ms)),'p95_ms':float(np.quantile(ms,.95)),'max_ms':float(max(ms)),'sha256':sha(f),'released_bytes':0})
assert sha(R/'before.f32')==sha(R/'after.f32')
def ports(label):
 return [l for l in (R/(label+'.log')).read_text().splitlines() if l.startswith(('in[','out['))]
base_ports=ports('before');assert len(base_ports)==16
for label in ['range0','range1','after']:assert ports(label)==base_ports,(label,ports(label),base_ports)
segments=[]
for label in ['range0','range1']:
 off=0
 for name,n in meta['outputs']:
  a,b=outputs['before'][off:off+n],outputs[label][off:off+n];off+=n
  segments.append({'case':label,'name':name,'changed_elements':int(np.sum(a!=b)),'max_abs':float(abs(a-b).max()),'relative_l2':float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-30))})
 assert off==len(outputs[label])
r=[read(R/'pre-health.json')]+[json.loads(l) for l in (R/'resources.jsonl').read_text().splitlines()]+[read(R/'post-health.json')]
def vm(x):return {k:int(v) for k,v in (l.split() for l in x['vmstat'].splitlines())}
res={'max_temperature_C':max(max(x['temperatures_mC'].values()) for x in r)/1000,'min_mem_available_B':min(int(x['memory']['MemAvailable'].split()[0])*1024 for x in r),'swapin_pages':vm(r[-1])['pswpin']-vm(r[0])['pswpin'],'swapout_pages':vm(r[-1])['pswpout']-vm(r[0])['pswpout']}
out={'scope':'M64/current DFP weights, history128 two-layer API only, 120 timed and12warmup calls; one canonical input, not semantic quality or full TTFT','rows':rows,'segments':segments,'original_AA_byte_equal':True,'actual_sdk_16_ports_byte_equal':True,'resources':res,'goal_achieved':False}
(R/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
