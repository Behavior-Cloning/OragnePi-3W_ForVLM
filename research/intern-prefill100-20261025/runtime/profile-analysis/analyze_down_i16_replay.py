"""Local error on identical frozen original NPU inputs; never full-model quality."""
from pathlib import Path
import json,hashlib,tarfile,numpy as np
T=Path(__file__).resolve().parents[2];RUN='down-i16-original-input-replay-r1';D=T/'results'/RUN
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ctl=json.loads((T/(RUN+'-controller.json')).read_text());assert ctl['status']=='complete' and ctl['returncode']==0
assert json.loads((D/'meta.json').read_text())['status']=='complete'
cfg=json.loads((D/'config.json').read_text());es=[json.loads(x) for x in (D/'events.jsonl').read_text().splitlines()];events={x['case']:x for x in es if x['event']=='result'};assert len(events)==104
for n,h in cfg['source_sha256'].items():assert sha(T/n)==h,n
archive=T/(RUN+'.tar.gz');assert sha(archive)==ctl['archive_sha256']
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  if m.isfile():assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==sha(D/m.name.removeprefix('./'))
raw={};baseports=None
for name,e in events.items():
 f=D/(name+'.f32');assert sha(f)==e['output_sha256'] and f.stat().st_size==90112
 x=np.fromfile(f,'<f4');assert np.isfinite(x).all();raw[name]=x;m=e['metrics']
 assert m['released_bytes']==0 and m['mismatched_elements']==0 and m['repeats']==2 and m['warmup_repeats']==1
 assert len(m['ports'])==16
 if baseports is None:baseports=m['ports']
 assert m['ports']==baseports
 assert not m['npd_enabled'] and not m['io_profile_enabled']
def parts(x,v):
 out={'hidden':x[:16*896].reshape(16,896)[:v]};off=16*896
 for key in ['l2_k','l3_k','l2_v','l3_v']:
  out[key]=x[off:off+2*16*64].reshape(2,16,64)[:,:v,:].transpose(1,0,2).reshape(v,128);off+=2*16*64
 assert off==len(x);return out
def stats(a,b):
 a=np.asarray(a);b=np.asarray(b);da=a.astype('float64');db=b.astype('float64');dif=db-da;den=np.linalg.norm(da)
 rowden=np.linalg.norm(da,axis=1);rowerr=np.linalg.norm(dif,axis=1)/np.maximum(rowden,1e-30)
 return {'rows':len(a),'rel_l2':float(np.linalg.norm(dif)/max(den,1e-30)),'max_abs':float(abs(dif).max()),'rms_abs':float(np.sqrt(np.mean(dif**2))),'max_row_rel_l2':float(rowerr.max()),'median_row_rel_l2':float(np.median(rowerr)),'changed_elements':int((a.view(np.uint32)!=b.view(np.uint32)).sum()),'elements':a.size}
rows=[];groups={};historical=0;aa=0
for fixture in cfg['fixtures']:
 k=fixture['key'];v=fixture['valid_rows'];pos=fixture['position'];s=int(fixture['case'].split('-')[0][1:]);base=raw[k+'-original'];hist=Path(fixture['original_output_path']);assert sha(hist)==fixture['original_output_sha256'];assert sha(Path(fixture['input_path']))==fixture['input_sha256'];assert np.array_equal(base.view(np.uint32),np.fromfile(hist,'<f4').view(np.uint32));historical+=1
 if k+'-original-after' in raw:assert np.array_equal(base.view(np.uint32),raw[k+'-original-after'].view(np.uint32));aa+=1
 bp=parts(base,v)
 for variant in ['i16-old','i16-range16']:
  cp=parts(raw[k+'-'+variant],v);rec={**fixture,'variant':variant,'api_original_ms':float(np.median(events[k+'-original']['metrics']['api_seconds'])*1000),'api_candidate_ms':float(np.median(events[k+'-'+variant]['metrics']['api_seconds'])*1000),'metrics':{key:stats(bp[key],cp[key]) for key in bp}}
  rows.append(rec)
  for j in range(v):
   region='first_token' if pos+j==0 else ('system_nonfirst' if pos+j<s else 'dynamic_with_template')
   regions=[region]
   if region=='system_nonfirst':regions.append('first_chunk_nonfirst' if pos==0 else 'later_system')
   for rr in regions:
    for key in bp:
     slot=groups.setdefault((variant,rr,key),[[],[]]);slot[0].append(bp[key][j]);slot[1].append(cp[key][j])
aggregate=[{'variant':va,'region':region,'port':port,**stats(np.stack(ab[0]),np.stack(ab[1]))} for (va,region,port),ab in groups.items()]
speed={v:{'p50_ms':float(np.median([a for e in events.values() if e['case'].endswith('-'+v) for a in e['metrics']['api_seconds']])*1000),'calls':sum(len(e['metrics']['api_seconds']) for e in events.values() if e['case'].endswith('-'+v))} for v in ['original','i16-old','i16-range16','original-after']}
samples=[json.loads(x) for x in (D/'resources.jsonl').read_text().splitlines()];pre=json.loads((D/'pre-health.json').read_text());post=json.loads((D/'post-health.json').read_text());all_samples=[pre]+samples+[post]
def vm(x):return dict((a,int(b)) for a,b in (l.split() for l in x['vmstat'].splitlines()))
resources={'samples':len(samples),'max_temperature_C':max(v for x in all_samples for v in x['temperatures_mC'].values())/1000,'min_available_MiB':min(int(x['memory']['MemAvailable'].split()[0]) for x in all_samples)/1024,'swap_in_pages_delta':vm(post)['pswpin']-vm(pre)['pswpin'],'swap_out_pages_delta':vm(post)['pswpout']-vm(pre)['pswpout']}
answer={'scope':'original-NPU-input teacher-forced two-layer replay; no propagated candidate history; not full-model TTFT or semantic qualification','integrity':{'windows':len(cfg['fixtures']),'unique_inputs':len(set(x['input_sha256'] for x in cfg['fixtures'])),'runs':len(events),'timed_api_calls':208,'warmup_calls':104,'historical_equal':historical,'aa_equal':aa,'all_ports_equal':True,'driver_zero':104,'valid_rows':528,'padded_rows_excluded':16},'rows':rows,'aggregate':aggregate,'api_only_speed':speed,'resources':resources,'source_sha256':{'analyzer':sha(Path(__file__)),'archive':sha(archive),'config':sha(D/'config.json')},'limitations':['Two prompts share prefix; first-token rows are the same underlying example twice.','No proposed deployment thresholds; low local error is insufficient for full-model semantics.','API excludes load/conversion/cache flush/invalidate; no speech/RAG residency.']}
(D/'ANALYSIS.json').write_text(json.dumps(answer,indent=2)+'\n');print(json.dumps({k:answer[k] for k in ['integrity','aggregate','api_only_speed','resources']},indent=2))
