from pathlib import Path
import json,re,statistics as st
P=Path(__file__).resolve().parent;O=P/'latency-watch-r3/vqa-results';a=json.loads((O/'summary.json').read_text());assert a['status']=='complete' and a['closed']['driver_bytes']==0
events=[json.loads(l) for l in (O/'events.jsonl').read_text().splitlines()]
resources={x['id']:x for x in events if x['event']=='request_resources'}
def vm(s):return {l.split()[0]:int(l.split()[1]) for l in s.splitlines()}
rows=[]
for i,c in enumerate(a['records']):
 m=c['metrics'];r=resources[m['id']];b=r['before'];e=r['after'];vb=vm(b['vmstat']);ve=vm(e['vmstat'])
 sb=b['stat'].rsplit(')',1)[1].split();se=e['stat'].rsplit(')',1)[1].split()
 rows.append({'trial':i//6,'case':c['id'],'answer':c['answer'],'ttft':m['ttft_seconds'],'prefill':m['prefill_seconds'],'api_total':m['api_seconds'],'decode_tps':m['decode_tokens_per_second'],'wall':e['at']-b['at'],'major_faults':int(se[9])-int(sb[9]),'minor_faults':int(se[7])-int(sb[7]),'vm_delta':{k:ve.get(k,0)-vb.get(k,0) for k in ['pswpin','pswpout','pgmajfault','allocstall_normal','pgscan_direct','pgsteal_direct']},'mem_before_kib':int(re.search(r'MemAvailable:\s+(\d+)',b['meminfo']).group(1)),'mem_after_kib':int(re.search(r'MemAvailable:\s+(\d+)',e['meminfo']).group(1)),'psi_available':b['memory_pressure'] is not None})
assert len(rows)==18
result={'records':rows,'ttft_median':st.median(x['ttft'] for x in rows),'ttft_max':max(x['ttft'] for x in rows),'closed':a['closed']}
(P/'latency-watch-r3/AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
