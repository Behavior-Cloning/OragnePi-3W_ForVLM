"""Serial, fixed-feedback NPU latency matrix. Uses existing immutable graph directory."""
import argparse,hashlib,json,os,statistics,subprocess,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--work',required=True);a=p.parse_args()
 root=Path(a.root);work=Path(a.work);config=json.loads((work/'manifest.json').read_text())
 output=work/'results';output.mkdir(exist_ok=False)
 files=sorted(root.glob('layer*.nb'))+sorted(root.glob('prefill-layer*.nb'))+[root/config['runner'],root/'thermal_condition.py',root/'embeddings.f32']
 hashes={x.name:sha(x) for x in files};save(output/'model-hashes.json',hashes)
 state={'status':'running','config':config,'runs':[],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 env={k:v for k,v in os.environ.items() if not k.startswith('OPI_')}
 env.update(OPI_PERFORMANCE='1',OPI_DECODER_GROUP='12',OPI_FAST_GREEDY='1',OPI_GROUP_HEAD='1',OPI_PREFILL_TOKENS='16',OPI_MAX_START_TEMP_MC='68000',OPI_THERMAL_LOW_MC='55000',OPI_THERMAL_HIGH_MC='64999',OPI_FEEDBACK_FILE=str(root/config['feedback']))
 try:
  for phase,repeat in [('warmup',0),('measure',0),('measure',1),('measure',2)]:
   cases=config['cases'] if repeat%2==0 else list(reversed(config['cases']))
   for case in cases:
    name=f"{case['id']}-{phase}-{repeat}";folder=output/name;folder.mkdir()
    env['OPI_CACHED_PREFIX']=str(case['prefix'])
    prompt=root/config['old_prompt'] if case['id']=='p1024-d32-control' else work/(case['id']+'.i32')
    command=['taskset','-c','6,7','./'+config['runner'],'.','24','1280','151674','embeddings.f32',str(prompt),'128',str(folder/'tokens.json'),str(folder/'unused-logits.f32'),'151645']
    state['current']=name;save(output/'summary.json',state);print('START',name,flush=True)
    started=time.monotonic()
    with (folder/'stdout.log').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
     subprocess.run(command,cwd=root,env=env,stdout=stdout,stderr=stderr,check=True,timeout=600)
    d=json.loads((folder/'tokens.json').read_text());d.update(case=case['id'],phase=phase,repeat=repeat,wall_seconds=time.monotonic()-started)
    assert d['generated_tokens']==128 and d['prompt_tokens']==case['prefix']+case['dynamic']
    assert d['cached_prefix_tokens']==case['prefix'] and d['new_prompt_tokens']==case['dynamic']
    assert d['head_on_npu'] and d['driver_after_release_bytes']==0
    if 'response_temperature_mC' in d:assert 55000<=d['response_temperature_mC']<=64999
    d['tokens_sha256']=hashlib.sha256(json.dumps(d['tokens']).encode()).hexdigest()
    prev=[r for r in state['runs'] if r['case']==case['id']]
    assert not prev or prev[0]['tokens_sha256']==d['tokens_sha256'],'same prompt predictions changed'
    state['runs'].append(d);save(output/'summary.json',state)
    print('DONE',name,'ttft',d['ttft_seconds'],'tps',d['decode_tokens_per_second'],flush=True)
  state['medians']={}
  for case in config['cases']:
   rs=[r for r in state['runs'] if r['case']==case['id'] and r['phase']=='measure']
   state['medians'][case['id']]={k:statistics.median(r[k] for r in rs) for k in ['ttft_seconds','decode_tokens_per_second','total_seconds','load_seconds','preparation_seconds','cpu_percent_one_core','max_rss_kib','driver_allocated_bytes','wall_seconds']}
  state['status']='complete'
 except BaseException as exc:
  state.update(status='failed',error=repr(exc));raise
 finally:save(output/'summary.json',state)
if __name__=='__main__':main()
