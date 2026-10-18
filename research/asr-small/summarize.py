import collections,json,re,statistics,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/asr-small-20261018'
def norm(s):return ''.join(c.lower() for c in s if not unicodedata.category(c).startswith(('P','Z','C')))
def dist(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  new=[i]
  for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
  row=new
 return row[-1]
def quant(a,q):return sorted(a)[max(0,__import__('math').ceil(len(a)*q)-1)]
def main():
 manifest=json.loads((OUT/'corpus-manifest.json').read_text());samples={r['id']:r for r in manifest['samples']};keys=manifest['acceptance_before_results']['critical_strings'];summary={}
 for path in sorted(OUT.glob('standalone-*/*/result.json')):
  d=json.loads(path.read_text());runs=d['runs'];first={r['id']:r for r in runs};groups={}
  for condition in sorted({samples[r['id']]['condition'] for r in runs}):
   rs=[r for r in first.values() if samples[r['id']]['condition']==condition];nr=sum(len(norm(r['reference'])) for r in rs);errors=sum(dist(norm(r['reference']),norm(r['text'])) for r in rs)
   groups[condition]={'n':len(rs),'reference_chars':nr,'edit_errors':errors,'cer':errors/nr if nr else None,'outputs':[{'id':r['id'],'reference':r['reference'],'hypothesis':r['text'],'edit_errors':dist(norm(r['reference']),norm(r['text']))} for r in rs]}
  hot=[r for r in runs if samples[r['id']]['condition']=='clean'];duration=sum(r['duration_seconds'] for r in hot);seconds=sum(r['seconds'] for r in hot)
  keyfail=[{'id':i,'missing':[x for x in strings if x not in norm(first[i]['text'])],'reference':first[i]['reference'],'hypothesis':first[i]['text']} for i,strings in keys.items() if i in first and any(x not in norm(first[i]['text']) for x in strings)]
  system=json.loads(path.with_name('system.json').read_text());summary[str(path.relative_to(OUT))]={'kind':d['kind'],'load_seconds':d['load_seconds'],'peak_rss_mib':d['final']['peak_rss_kib']/1024,'peak_sampled_pss_mib':max([p.get('Pss_kib',0) for row in system for p in row['processes'].values()] or [0])/1024,'available_min_mib':min(row['mem_available_kib'] for row in system)/1024,'swap_max_mib':max(row['swap_used_kib'] for row in system)/1024,'clean_duration_p50':statistics.median(r['duration_seconds'] for r in hot),'latency_p50':statistics.median(r['seconds'] for r in hot),'latency_p95':quant([r['seconds'] for r in hot],.95),'rtf_aggregate':seconds/duration,'cpu_fraction':sum(r['cpu_seconds'] for r in hot)/seconds,'critical_string_failures':keyfail,'groups':groups,'same_common_subset':{'n':len(first),'ids':list(first)},'repeat_text_equal':all(r['text']==first[r['id']]['text'] for r in runs) if len(runs)>len(first) else None}
 (OUT/'standalone-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 for k,v in summary.items():print(k,{a:b for a,b in v.items() if a not in ('groups','critical_string_failures','same_common_subset')},'cleanCER',v['groups']['clean']['cer'],'criticalfails',len(v['critical_string_failures']))
if __name__=='__main__':main()
