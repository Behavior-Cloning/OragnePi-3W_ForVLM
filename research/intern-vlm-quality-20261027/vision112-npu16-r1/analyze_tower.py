from pathlib import Path
import json,array,math,statistics
P=Path(__file__).resolve().parent;B=P/'board';S=Path(json.loads((P/'build-status.json').read_text())['scratch'])
ev=[json.loads(s) for s in (B/'results/events.jsonl').read_text().splitlines()]
assert ev[-1]['event']=='closed' and ev[-1]['driver_bytes']==0
hot=[x for x in ev if x['event']=='result' and x['run']>=0];assert len(hot)==6
quality=[]
for i in [1,2]:
 a=array.array('f');a.frombytes((B/f'results/run{i-1}-output.f32').read_bytes())
 b=array.array('f');b.frombytes((S/f'projector/reference{i}.f32').read_bytes())
 assert len(a)==len(b)==16*896 and all(math.isfinite(x) for x in a)
 quality.append({'image':i,'relative_l2':math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/sum(y*y for y in b)),'cosine':sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))})
report={'scope':'full112 NPU tower stem+24blocks+projector; preprocessed input ->16 vectors, no language or JPEG time','hot_n':len(hot),'p50_s':statistics.median(x['stack_seconds'] for x in hot),'min_s':min(x['stack_seconds'] for x in hot),'max_s':max(x['stack_seconds'] for x in hot),'api_p50_s':statistics.median(sum(x['api_seconds']) for x in hot),'host_p50_s':statistics.median(x['host_copy_sync_seconds'] for x in hot),'ready':ev[0],'closed':ev[-1],'quality':quality,'max_temp_c':max(x['temperature_mC'] for x in hot)/1000}
(P/'TOWER_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
