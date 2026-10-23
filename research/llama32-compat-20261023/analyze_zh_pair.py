"""Independent six-calibration/nine-Chinese-heldout paired output audit."""
from pathlib import Path
import hashlib,json,statistics,numpy as np
r=Path(__file__).resolve().parent;fixtures=Path('/opt/opi-lab/llama32-compat-20261023/fixtures-zh');mapping=json.loads((r/'heldout-zh-fixtures.json').read_text());case_map={x['fixture_id']:{**x,'text':c['text'],'case_id':c['case_id']} for c in mapping['cases'] for x in c['states']}
variants={'all_fp16':'board-results-all-fp16-zh','attention_fp16_mlp_w8':'board-results-attention-fp16-mlp-w8-zh'};expected={0,1,2,4,8,14}|set(case_map);results={};summary={}
for key,dirname in variants.items():
 b=r/dirname;events=[json.loads(x) for x in (b/'probe.log').read_text().splitlines() if x.startswith('{')];rows=[x for x in events if x.get('event')=='fixture'];assert len(rows)==len(expected) and {x['fixture_id'] for x in rows}==expected;metrics=[]
 for e in rows:
  ident=e['fixture_id'];p=b/'outputs'/f'p{ident}-output.f32';x=np.fromfile(p,dtype='<f4').astype(np.float64);y=np.fromfile(fixtures/f'p{ident}-reference.f32',dtype='<f4').astype(np.float64);assert x.shape==y.shape==(2048,) and np.isfinite(x).all();rel=float(np.linalg.norm(x-y)/np.linalg.norm(y));assert abs(rel-e['hidden_relative_l2'])<1e-6
  m={**e,'independent_relative_l2':rel,'cosine':float(x@y/np.linalg.norm(x)/np.linalg.norm(y)),'output_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'subset':'chinese_heldout' if ident in case_map else 'english_calibration','fixture':case_map.get(ident,{'actual_position':ident})};metrics.append(m);results.setdefault(ident,{})[key]=m
 assert events[-1]['event']=='closed' and events[-1]['driver_released']==0
 sub={}
 for group in ['english_calibration','chinese_heldout']:
  ms=[x for x in metrics if x['subset']==group];sub[group]={'n':len(ms),'relative_l2_min':min(x['independent_relative_l2'] for x in ms),'relative_l2_max':max(x['independent_relative_l2'] for x in ms),'api_median_seconds':statistics.median(x['run_seconds'] for x in ms),'coarse_screen_pass_count':sum(x['independent_relative_l2']<=.05 for x in ms)}
 summary[key]={'subsets':sub,'closed':events[-1],'api_median_seconds':statistics.median(x['run_seconds'] for x in metrics)}
d={'scope':'Same six English calibration states and nine unseen Chinese states from three independent sentences; one layer, no generation. Source is Q4 GGUF dequantized, not original BF16.','coarse_threshold':.05,'threshold_warning':'5% is a legacy layer compatibility screening threshold, not Chinese accuracy or full-model approval.','summary':summary,'results':results}
(r/'attention-fp16-mlp-w8-zh-comparison.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,indent=2))
