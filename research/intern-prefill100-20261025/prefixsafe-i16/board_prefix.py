from pathlib import Path
import json,hashlib,sys,time,traceback,threading
from resident_worker_v1 import ResidentWorker
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 cfg=json.loads((ROOT/'config.json').read_text());out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
 (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
 sys.path.insert(0,cfg['base_path']+'/deps');from tokenizers import Tokenizer
 import numpy as np
 assert sha(cfg['tokenizer_path'])==cfg['tokenizer_sha256'];tok=Tokenizer.from_file(cfg['tokenizer_path'])
 for c in cfg['speed_cases']+cfg['quality_cases']+cfg['diag_cases']:
  assert tok.encode(c['prompt_text'],add_special_tokens=False).ids==c['tokens']
  assert len(c['tokens'])==c['static_tokens']+c['body_tokens']+8
  assert c['tokens'][:c['static_tokens']]==c['prefix_tokens']
 log=(out/'events.jsonl').open('w',buffering=1);w=None;success=False;stop=threading.Event();errors=[];state={'variant':None,'phase':None}
 def emit(event,**kw):log.write(json.dumps(dict(event=event,at=time.monotonic(),**kw),ensure_ascii=False)+'\n')
 def snap():
  return dict(at=time.monotonic(),**state,temperatures_mC={p.parent.name:int(p.read_text()) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')},meminfo=Path('/proc/meminfo').read_text(),vmstat=Path('/proc/vmstat').read_text(),stat=Path('/proc/stat').read_text(),loadavg=Path('/proc/loadavg').read_text(),frequencies={p.parts[-3]:int(p.read_text()) for p in Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_cur_freq')})
 def monitor():
  try:
   with (out/'resources.jsonl').open('w') as f:
    while not stop.wait(1):
     row=snap();f.write(json.dumps(row)+'\n');f.flush()
     if max(row['temperatures_mC'].values())>=78000:raise RuntimeError('thermal guard')
  except BaseException:errors.append(traceback.format_exc());stop.set()
 thread=threading.Thread(target=monitor,daemon=True);thread.start();(out/'pre-health.json').write_text(json.dumps(snap()))
 def req(ids,n,reset=False,fixed=False,diag=False,prefix=0):
  assert not errors
  assert max(snap()['temperatures_mC'].values())<78000
  return list(w.generate(ids,n,reset=reset,diagnostic=diag,feedback=[100662]*n if fixed else None,timeout=90,original_prefix_tokens=prefix))[-1]
 try:
  assert sha(ROOT/'resident-prefixsafe')==cfg['binary_sha256']
  for variant in cfg['variants']:
   state['variant']=variant['name'];original_calls=0
   for path,h in {**variant['expected_graph_sha256'],**variant.get('extra_graph_sha256',{})}.items():assert sha(path)==h
   folder=out/variant['name'];folder.mkdir();dumps=folder/'logits';dumps.mkdir()
   command=['env',*[k+'='+v for k,v in variant['env'].items()],'taskset','-c','4,5',str(ROOT/'resident-prefixsafe'),variant['graph'],'24','1280','151674',variant['graph']+'/embeddings.f32','16','151645']
   w=ResidentWorker(command,cfg['base_path'],folder/'process.log',dump_dir=dumps);assert w.ready['layers']==24 and w.ready['capacity']==1280 and not w.ready['true_tail8_enabled'];emit('ready',variant=variant['name'],ready=w.ready,command=command)
   state['phase']='prefix_check'
   for static in [64,128]:
    case=next(c for c in cfg['speed_cases'] if c['static_tokens']==static)
    prefix=static if variant.get('whole_system_original') else 0
    done=req(case['prefix_tokens'],1,True,True,True,prefix);assert done['reused_tokens']==0
    if variant['env'].get('OPI_FIRST_PREFILL_L2'):original_calls+=static//16 if prefix else 1
    f=dumps/(str(done['id'])+'-logits.f32');assert f.stat().st_size==151674*4 and np.isfinite(np.fromfile(f,'<f4')).all()
    emit('prefix_check',variant=variant['name'],static_tokens=static,metrics=done,logits_file=str(f.relative_to(out)),logits_sha256=sha(f),timing_eligible=False)
   for phase,cases,repeats,n,fixed,diag in [('speed',cfg['speed_cases'],2,1,False,False),('natural',cfg['quality_cases'],1,32,False,False),('fixed',cfg['diag_cases'],1,4,True,True)]:
    state['phase']=phase
    for repeat in range(repeats):
     for case in cases if repeat%2==0 else list(reversed(cases)):
      prefix=case['static_tokens'] if variant.get('whole_system_original') else 0
      prep=req(case['prefix_tokens'],1,True,True,False,prefix);done=req(case['tokens'],n,False,fixed,diag)
      assert prep['reused_tokens']==0 and done['reused_tokens']==case['static_tokens'] and done['new_tokens']==case['body_tokens']+8
      if variant['env'].get('OPI_FIRST_PREFILL_L2'):original_calls+=case['static_tokens']//16 if prefix else 1
      extra={}
      if diag:
       f=dumps/(str(done['id'])+'-logits.f32');assert f.stat().st_size==done['generated_tokens']*151674*4;assert np.isfinite(np.fromfile(f,'<f4')).all()
       extra={'logits_file':str(f.relative_to(out)),'logits_sha256':sha(f)}
      emit('result',variant=variant['name'],case=case['id'],kind=phase,repeat=repeat,static_tokens=case['static_tokens'],system_body_tokens=case['static_tokens']-5,system_template_tokens=5,dynamic_body_tokens=case['body_tokens'],dynamic_template_tokens=8,total_prompt_tokens=len(case['tokens']),original_prefix_tokens=prefix,metrics=done,prepare=prep,reply=tok.decode(done['tokens']),input=case.get('input',case.get('body')),timing_eligible=not diag,**extra)
   closed=w.close();w=None;assert closed['driver_bytes']==0
   text=(folder/'process.log').read_text();switches=text.count('PREFIX_ORIGINAL block=1 ');assert switches==original_calls,(switches,original_calls)
   assert text.count('PREFIX_SUFFIX_GROUP blocks=10')==original_calls
   emit('closed',variant=variant['name'],metrics=closed,original_prefix_calls=switches)
  assert not errors;success=True
 except BaseException:emit('error',traceback=traceback.format_exc());raise
 finally:
  if w:
   try:emit('closed_on_failure',metrics=w.close())
   except BaseException:w.abort()
  stop.set();thread.join(5)
  if thread.is_alive():errors.append('monitor failed to join')
  (out/'post-health.json').write_text(json.dumps(snap()))
  (out/'meta.json').write_text(json.dumps(dict(status='complete' if success and not errors else 'failed',scope=cfg['scope'],monitor_errors=errors)))
  log.close()
if __name__=='__main__':main()
