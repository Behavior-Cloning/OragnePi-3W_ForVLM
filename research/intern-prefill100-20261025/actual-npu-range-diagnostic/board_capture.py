from pathlib import Path
import json,os,hashlib,sys,time,traceback
from resident_worker_v1 import ResidentWorker
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 cfg=json.loads((ROOT/'config.json').read_text());out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
 (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
 log=(out/'events.jsonl').open('w',buffering=1);worker=None;success=False
 def emit(event,**kw):log.write(json.dumps(dict(event=event,at=time.monotonic(),**kw),ensure_ascii=False)+'\n')
 def run(w,tokens,reset=False,n=1):
  result=list(w.generate(tokens,n,reset=reset,diagnostic=True,feedback=[100662]*n,timeout=90))[-1]
  assert result['event']=='done' and result['generated_tokens']==n and result['diagnostic'] and result['fixed_feedback']
  return result
 try:
  assert sha(ROOT/'resident-capture')==cfg['binary_sha256']
  assert sha(cfg['original_binary'])==cfg['original_binary_sha256']
  emit('sources',hashes={x.name:sha(x) for x in ROOT.iterdir() if x.is_file()})
  for variant in cfg['variants']:
   for filename,expected in variant['expected_graph_sha256'].items():assert sha(filename)==expected,filename
   emit('graphs_verified',variant=variant['name'],hashes=variant['expected_graph_sha256'])
   for case in cfg['cases']:
    reference=None
    for capture in [False,True]:
     assert int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())<78000
     name=variant['name']+'-'+case['id']+('-capture' if capture else '-control')
     folder=out/name;folder.mkdir();io=folder/'io';io.mkdir();dumps=folder/'logits';dumps.mkdir()
     env=dict(variant['env'])
     if capture:env['OPI_GROUP_CAPTURE_DIR']=str(io)
     binary=str(ROOT/'resident-capture') if capture else cfg['original_binary']
     cmd=['env',*[k+'='+v for k,v in env.items()],'taskset','-c','4,5',binary,variant['graph'],'24','1280','151674',variant['graph']+'/embeddings.f32','16','151645']
     worker=ResidentWorker(cmd,cfg['base_path'],folder/'process.log',dump_dir=dumps)
     emit('ready',name=name,ready=worker.ready,command=cmd)
     prep=run(worker,case['prefix_tokens'],True,1);done=run(worker,case['tokens'],False,4)
     assert prep['reused_tokens']==0 and done['reused_tokens']==case['static_tokens'] and done['new_tokens']==case['body_tokens']+8
     assert done['scalar_prompt_tokens']==0 and done['tail_rows']==8
     f=dumps/(str(done['id'])+'-logits.f32');assert f.stat().st_size==4*151674*4
     h=sha(f)
     if capture:assert h==reference,'capture modified final logits'
     else:reference=h
     records=[dict(name=x.name,bytes=x.stat().st_size,sha256=sha(x)) for x in io.iterdir()]
     expected_batches=(len(case['tokens'])+15)//16
     assert len(records)==(expected_batches*2 if capture else 0)
     emit('result',name=name,capture=capture,case=case['id'],variant=variant['name'],prepare=prep,metrics=done,logits_sha256=h,capture_control_exact=True if capture else None,captures=records,timing_eligible=False)
     closed=worker.close();worker=None;emit('closed',name=name,metrics=closed);assert closed['driver_bytes']==0
  success=True
 except BaseException:emit('error',traceback=traceback.format_exc());raise
 finally:
  if worker:
   try:emit('closed_on_failure',metrics=worker.close())
   except BaseException:worker.abort()
  (out/'meta.json').write_text(json.dumps(dict(status='complete' if success else 'failed',scope='Actual NPU layer2/3 grouped IO; diagnostic only, no speech/RAG concurrent, no performance claim')))
  log.close()
if __name__=='__main__':main()
