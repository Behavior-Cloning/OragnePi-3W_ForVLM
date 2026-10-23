from pathlib import Path
import json,statistics,collections,argparse
from tokenizers import Tokenizer
R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('result',type=Path);a=p.parse_args();spec=json.loads((R/'inputs.json').read_text());tok=Tokenizer.from_file(spec['tokenizer_path']);events=[json.loads(x) for x in (a.result/'events.jsonl').read_text().splitlines()];groups=collections.defaultdict(list);natural=[];per_case=[]
for e in events:
 if e['event']!='request':continue
 r=e['result'];groups[e['kind']].append(r)
 per_case.append(dict(kind=e['kind'],case=e['case'],prompt_tokens=r['prompt_tokens'],reused=r['reused_tokens'],ttft=r['ttft_seconds'],decode=r['decode_tokens_per_second'],generated=r['generated_tokens'],scalar_prompt_tokens=r['scalar_prompt_tokens'],batch_calls=r['batch_calls'],tail_rows=r['tail_rows'],first_head=r['first_head_seconds']))
 if e['kind'].startswith('natural'):natural.append(dict(kind=e['kind'],case=e['case'],text=tok.decode(r['tokens'],skip_special_tokens=True),tokens=r['tokens'],finish_reason=r['finish_reason'],ttft=r['ttft_seconds'],decode=r['decode_tokens_per_second']))
perf={}
for kind,rows in groups.items():perf[kind]=dict(n=len(rows),ttft_median=statistics.median(r['ttft_seconds'] for r in rows),decode_median=statistics.median(r['decode_tokens_per_second'] for r in rows),generated=[r['generated_tokens'] for r in rows],reused=[r['reused_tokens'] for r in rows],mean_cpu_cores=statistics.mean(r['cpu_seconds']/r['total_seconds'] for r in rows))
samples=[x for x in json.loads((a.result/'telemetry.json').read_text())['samples'] if 'error' not in x];resource=dict(samples=len(samples),temperature_max=max(v for x in samples for v in x['temperature'].values()),sum_rss_kB_max=max(sum(p['Rss_kB'] for p in x['processes']) for x in samples),sum_pss_kB_max=max(sum(p['Pss_kB'] for p in x['processes']) for x in samples),available_kB_min=min(x['memory']['MemAvailable_kB'] for x in samples),system_swap_used_kB_max=max(x['memory']['SwapTotal_kB']-x['memory']['SwapFree_kB'] for x in samples))
result=dict(source=spec,complete=any(e['event']=='complete' for e in events),loaded=[e for e in events if e['event']=='loaded'],closed=[e for e in events if e['event']=='closed'],errors=[e for e in events if e['event']=='error'],natural=natural,performance=perf,per_case=per_case,resources=resource,limits=['same3 semantic questions, small pilot','Intern batch16 plus tail8; Llama scalar prefill','Intern W8 sensitive exceptions, Llama attention/head FP16; not equivalent precision','no original BF16 accuracy oracle for Intern in this matched run'])
(a.result/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
