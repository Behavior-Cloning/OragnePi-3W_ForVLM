import json,sys,traceback
from pathlib import Path
from resident_worker import ResidentWorker
root=Path(__file__).resolve().parent;cfg=json.loads((root/'config.json').read_text());doc=json.loads((root/'aligned-inputs.json').read_text());case=doc['performance'][0]
for cycle in range(cfg.get('cycles',1)):
 command=['env','ASAN_OPTIONS=detect_leaks=0:abort_on_error=1',*[k+'='+v for k,v in cfg['env'].items()],'taskset','-c','4,5',str(root/'resident-test'),cfg['graph'],'24','256','151674',cfg['graph']+'/embeddings.f32','16','151645']
 worker=ResidentWorker(command,'/opt/opi-vlm/resident-agent-20261019',root/('stderr-'+str(cycle)+'.log'))
 print(json.dumps({'cycle':cycle,'ready':worker.ready}),flush=True)
 if cfg.get('request',True):
  for tokens,count,feedback in [(case['prefix_tokens'],1,None),(case['tokens'],32,[100662]*32)]:
   done=list(worker.generate(tokens,max_new=count,feedback=feedback,reset=count==1,timeout=90))[-1]
   print(json.dumps({'cycle':cycle,'done':done}),flush=True)
 closed=worker.close();print(json.dumps({'cycle':cycle,'closed':closed}),flush=True);assert closed['driver_bytes']==0
