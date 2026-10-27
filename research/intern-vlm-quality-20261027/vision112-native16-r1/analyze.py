from pathlib import Path
import json,statistics,math,array,re
P=Path(__file__).resolve().parent;D=P/'results'
s=json.loads((D/'summary.json').read_text());assert s['state']=='completed'
rows=[]
for phase in s['phases']:
 ev=phase['events'];a=[e for e in ev if e.get('event')=='encoded'];assert len(a)==11 and phase['returncode']==0 and ev[-1]['event']=='closed'
 assert all(e['side']==112 and e['image_tokens']==16 and e['dim']==896 and e['mode']=='native' for e in a)
 times=[e['encoder_projector_ms'] for e in a[1:]]
 rows.append({'phase':phase['phase'],'load_ms':next(e['load_ms'] for e in ev if e.get('event')=='ready'),'warmup_ms':a[0]['encoder_projector_ms'],'hot_n':len(times),'p50_ms':statistics.median(times),'min_ms':min(times),'max_ms':max(times)})
def read(n):
 a=array.array('f');a.frombytes((D/n).read_bytes());assert len(a)==16*896 and all(math.isfinite(x) for x in a);return a
quality={}
for image,i in [('image1',1),('image2',2)]:
 ref=read(f'fp16-before-{i}.f32')
 for phase in ['fp16-after','q8-stock','q8-repack']:
  v=read(f'{phase}-{i}.f32');err=math.sqrt(sum((x-y)**2 for x,y in zip(v,ref))/sum(x*x for x in ref))
  quality[image+'-'+phase]={'relative_l2':err}
resources=[json.loads(x) for x in (D/'resources.jsonl').read_text().splitlines()]
max_temp=max(int(v)/1000 for x in resources for v in x['temperatures'].values())
rss=[]
for x in resources:
 m=re.search(r'^VmRSS:\s+(\d+)',x.get('process_status',''),re.M)
 if m:rss.append(int(m.group(1))/1024)
audit={'scope':s['scope'],'rows':rows,'output_checks':quality,'max_temperature_c':max_temp,'max_rss_mib':max(rss),'quality_note':'112 native shuffle16 original projector; VQA evaluated separately'}
(P/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
