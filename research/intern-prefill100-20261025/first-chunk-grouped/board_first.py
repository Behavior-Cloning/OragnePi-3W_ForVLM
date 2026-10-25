from pathlib import Path
import json,hashlib,sys,time,traceback
from resident_worker_v1 import ResidentWorker
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 cfg=json.loads((ROOT/'config.json').read_text());out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
 (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
 sys.path.insert(0,cfg['base_path']+'/deps');from tokenizers import Tokenizer
 assert sha(cfg['tokenizer_path'])==cfg['tokenizer_sha256'];tok=Tokenizer.from_file(cfg['tokenizer_path'])
 log=(out/'events.jsonl').open('w',buffering=1);w=None;success=False
 def emit(event,**kw):log.write(json.dumps(dict(event=event,at=time.monotonic(),**kw),ensure_ascii=False)+'\n')
 def req(ids,n,reset=False,fixed=False):return list(w.generate(ids,n,reset=reset,diagnostic=True,feedback=[100662]*n if fixed else None,timeout=90))[-1]
 try:
  assert sha(ROOT/'resident-first-chunk-grouped')==cfg['binary_sha256']
  for variant in cfg['variants']:
   for path,h in {**variant['expected_graph_sha256'],**variant.get('extra_graph_sha256',{})}.items():assert sha(path)==h
   folder=out/variant['name'];folder.mkdir();dumps=folder/'logits';dumps.mkdir()
   command=['env',*[k+'='+v for k,v in variant['env'].items()],'taskset','-c','4,5',str(ROOT/'resident-first-chunk-grouped'),variant['graph'],'24','1280','151674',variant['graph']+'/embeddings.f32','16','151645']
   w=ResidentWorker(command,cfg['base_path'],folder/'process.log',dump_dir=dumps);emit('ready',variant=variant['name'],ready=w.ready,command=command)
   for case in cfg['cases']:
    for kind,n,fixed in [('fixed',4,True),('natural',32,False)]:
     assert int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())<78000
     prep=req(case['prefix_tokens'],1,True,True);done=req(case['tokens'],n,False,fixed)
     assert prep['reused_tokens']==0 and done['reused_tokens']==case['static_tokens'] and done['new_tokens']==case['body_tokens']+8
     f=dumps/(str(done['id'])+'-logits.f32');assert f.stat().st_size==done['generated_tokens']*151674*4
     emit('result',variant=variant['name'],case=case['id'],kind=kind,metrics=done,prepare=prep,logits_file=str(f.relative_to(out)),logits_sha256=sha(f),reply=tok.decode(done['tokens']),timing_eligible=False)
   closed=w.close();w=None;assert closed['driver_bytes']==0
   text=(folder/'process.log').read_text();switches=text.count('FIRST_CHUNK_ORIGINAL block=1 pos=0 rows=16 valid=16')
   assert switches==(len(cfg['cases'])*2 if variant['env'].get('OPI_FIRST_PREFILL_L2') else 0)
   restores=text.count('FIRST_CHUNK_GROUP_RESTORED blocks=12');assert restores==(switches if variant['env'].get('OPI_GROUP_PREFILL') else 0)
   emit('closed',variant=variant['name'],metrics=closed,original_first_chunk_calls=switches,group_restores=restores)
  success=True
 except BaseException:emit('error',traceback=traceback.format_exc());raise
 finally:
  if w:
   try:emit('closed_on_failure',metrics=w.close())
   except BaseException:w.abort()
  (out/'meta.json').write_text(json.dumps(dict(status='complete' if success else 'failed',scope=cfg['scope'])))
  log.close()
if __name__=='__main__':main()
