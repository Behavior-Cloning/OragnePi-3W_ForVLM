import json,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
cards={a['id']:a for a in json.loads((ROOT/'cards.json').read_text())};cases={c['id']:c for c in json.loads((ROOT/'cases.json').read_text())}
reviews=json.loads((ROOT/'adjudicated-review.json').read_text())['records'];review={(r['case_id'],r['card_id']):r for r in reviews}
assert len(review)==len(cards)*len(cases)
models=['bge_small','gte_base','bge_base','bge_large','qwen06']
results=[];perquery=[];measure=[]
def overall(values):return 'incompatible' if 'incompatible' in values else 'conditional' if 'conditional' in values else 'compatible'
def expand(ids):return [x for i in ids for x in (cards[i]['steps'] if cards[i]['type']=='skill' else [i])]
def sequence_notes(c,ids):
 seq=expand(ids);notes=[]
 dup=[a for a,n in collections.Counter(seq).items() if n>1]
 if dup:notes.append('重复原子步骤：'+','.join(dup))
 if c['family'] in ['greet_listen','observe_rest']:
  notes.append('指令要求先后阶段；检索结果没有阶段绑定和执行顺序，仍需编排验证，不能把点头/哈欠等单个ID自动当作完整阶段')
 if c['family'] in ['bedtime','observe_rest','morning']:
  eye=[a for a in seq if a in ['A23','A24']]
  if eye and ((c['family']=='morning' and eye[-1]=='A23') or (c['family']!='morning' and eye[-1]=='A24')):notes.append('最终眼部状态与醒来/休息目标需协调')
 return notes
for model in models:
 base=ROOT/'results'/model
 if not (base/'summary.json').exists():continue
 summary=json.loads((base/'summary.json').read_text());assert summary['complete']
 for n,h in summary['output_sha'].items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h
 raw=[json.loads(s) for s in (base/'raw.jsonl').read_text().splitlines()];assert len(raw)==240
 for m in summary['summary']:measure.append(dict(**m,pss=summary['timing_pss_peak'],pss_full=max(summary['pss_peak'],summary['timing_pss_peak']),temperature_telemetry_max=summary['temperature_max'],temperature_max=max([summary['temperature_max']]+[e['start_temp'] for e in summary['cooling_events'] if e['after_phase']=='timing']),temperature_start=summary['temperature_start'],cold_load_s=summary['load_seconds'],index_s=summary['index_seconds'],cooldown_s=summary['cooling_seconds'],available_min=summary['available_min'],swap_in_delta=summary['swap_in_delta'],swap_out_delta=summary['swap_out_delta']))
 for pool in ['atom','skill']:
  unique=[r for r in raw if r['pool']==pool and r['repeat']==0];assert len(unique)==60
  for path,field in [('raw','raw_order'),('state','order')]:
   for r in unique:
    c=cases[r['case_id']];ids=r[field][:5]
    assert all(cards[i]['type']==pool for i in ids)
    vals=[review[(c['id'],i)]['status'] for i in ids]
    perquery.append(dict(model=model,pool=pool,path=path,case_id=c['id'],query=c['query'],top5=[dict(id=i,name=cards[i]['name'],status=review[(c['id'],i)]['status'],reason=review[(c['id'],i)]['reason']) for i in ids],sequence_notes=sequence_notes(c,ids)))
   for k in range(1,6):
    eligible=[r for r in unique if len(r[field])>=k];groups=collections.Counter();rank=collections.Counter();hits=0;goaln=0;seqn=0;returned=[];coverage2=0;coverage2n=0
    for r in eligible:
     c=cases[r['case_id']];ids=r[field][:k];v=[review[(c['id'],i)]['status'] for i in ids]
     groups[overall(v)]+=1;rank[v[-1]]+=1;seqn+=bool(sequence_notes(c,ids))
     goals=c['goal_atoms' if pool=='atom' else 'goal_skills']
     if goals:goaln+=1;hits+=bool(set(goals)&set(ids))
     if c['family'] in ['greet_listen','observe_rest']:
      coverage2n+=1;flat=set(expand(ids))
      if c['family']=='greet_listen':coverage2+=bool(flat&{'A09','A10'}) and bool(flat&{'A06','A26','A41'})
      else:coverage2+=bool(flat&{'A05','A08','A31'}) and bool(flat&{'A22','A23','A41'})
    n=len(eligible)
    results.append(dict(model=model,pool=pool,path=path,k=k,n=n,shortfall=60-n,all_compatible=groups['compatible'],conditional_no_incompatible=groups['conditional'],any_incompatible=groups['incompatible'],no_hard_conflict=groups['compatible']+groups['conditional'],no_hard_conflict_rate=(groups['compatible']+groups['conditional'])/n if n else None,rank_counts=dict(rank),goal_signal_hit=hits,goal_signal_n=goaln,needs_sequence_or_duplicate_review=seqn,two_stage_atom_cue_proxy=coverage2,two_stage_cue_proxy_n=coverage2n,two_stage_cue_warning="Only atom-ID co-occurrence, not stage assignment or task coverage"))
(ROOT/'summary.json').write_text(json.dumps(dict(metrics=results,performance=measure,completed_models=sorted(set(x['model'] for x in results)),review_counts=dict(collections.Counter(x['status'] for x in reviews))),ensure_ascii=False,indent=2))
(ROOT/'ranking-evaluation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in perquery))
print('analyzed',len(perquery),'model/query/pool/path rows;',len(results),'TopK cells')
