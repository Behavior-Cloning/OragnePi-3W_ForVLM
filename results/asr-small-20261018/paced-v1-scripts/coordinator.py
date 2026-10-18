"""Run under external board flock; children deliberately share this one lock."""
import argparse,json,os,subprocess,sys,threading,time,statistics
from pathlib import Path
sys.path.insert(0,'/opt/opi-vlm/audio-lab-20261018')
from board_trials import monitor,NPU,PREFIX
ROOT=Path(__file__).resolve().parent
ENV=dict(os.environ,PYTHONPATH='/opt/opi-vlm/audio-lab-20261018/deps',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def command(k,out,extra=[]):return ['taskset','-c','2,3',sys.executable,str(ROOT/'worker.py'),'--kind',k,'--out',str(out),*extra]
def standalone(out,k,paced=False):
 folder=out/(k+('-paced' if paced else ''));folder.mkdir();done=threading.Event();rows=[]
 with (folder/'run.log').open('w') as log:
  child=subprocess.Popen(command(k,folder/'result.json',['--repeats','1','--indices','0,2,8,9,16,24,25,32','--paced'] if paced else ['--repeats','1','--indices','0,1,2,3,4,5,6,8,9,14,16,24,25,32'] if k=='qwen' else ['--repeats','2']),env=ENV,stdout=log,stderr=subprocess.STDOUT)
  thread=threading.Thread(target=monitor,args=(done,[child.pid],rows));thread.start()
  try:code=child.wait(timeout=1800)
  finally:
   if child.poll() is None:child.terminate();child.wait(timeout=10)
   done.set();thread.join();save(folder/'system.json',rows)
 assert code==0,(k,code)
 print('DONE',folder.name,flush=True)
def mixed(out,conditions,dynamic=256):
 summary=[]
 for repeat in range(2):
  for k in conditions if repeat==0 else conditions[::-1]:
   folder=out/f'{k}-{repeat}';folder.mkdir();print('START',folder.name,flush=True)
   child=None;done=threading.Event();rows=[];pids=[];thread=threading.Thread(target=monitor,args=(done,pids,rows));thread.start();logs=[]
   try:
    if k!='baseline':
     log=(folder/'audio.log').open('w');logs.append(log)
     child=subprocess.Popen(command(k,folder/'audio.json',['--limit','6','--ready',str(folder/'ready'),'--go',str(folder/'go'),'--stop',str(folder/'stop')]),env=ENV,stdout=log,stderr=subprocess.STDOUT);pids.append(child.pid)
     deadline=time.monotonic()+180
     while not (folder/'ready').exists():
      assert child.poll() is None,'audio exited'
      if time.monotonic()>deadline:raise TimeoutError('audio load')
      time.sleep(.1)
    with (folder/'thermal.log').open('w') as log:subprocess.run([sys.executable,'thermal_condition.py','--min','55000','--max','64999'],cwd=NPU,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=330)
    env={k:v for k,v in ENV.items() if not k.startswith('OPI_')};env.update(OPI_PERFORMANCE='1',OPI_DECODER_GROUP='12',OPI_FAST_GREEDY='1',OPI_GROUP_HEAD='1',OPI_PREFILL_TOKENS='16',OPI_CACHED_PREFIX='64',OPI_MAX_START_TEMP_MC='70000',OPI_FEEDBACK_FILE=str(NPU/(PREFIX+'-feedback.i32')))
    (folder/'go').write_text(str(time.monotonic()))
    with (folder/'llm.log').open('w') as log:
     llm=subprocess.Popen(['taskset','-c','6,7','./'+PREFIX+'-full_model','.','24','1280','151674','embeddings.f32',('/opt/opi-vlm/short-prefix-d64-20261018/p64-d64.i32' if dynamic==64 else '/opt/opi-vlm/short-prefix-20261018/p64-d256.i32'),'128',str(folder/'llm.json'),str(folder/'unused.f32'),'151645'],cwd=NPU,env=env,stdout=log,stderr=subprocess.STDOUT);pids.append(llm.pid)
     try:code=llm.wait(timeout=180)
     finally:
      if llm.poll() is None:llm.terminate();llm.wait(timeout=10)
    (folder/'stop').write_text(str(time.monotonic()))
    if child:child.wait(timeout=180);assert child.returncode==0
    assert code==0
    d=json.loads((folder/'llm.json').read_text());assert d['driver_after_release_bytes']==0 and d['generated_tokens']==128 and d['head_on_npu'] and d['decoder_layers']==24
    assert not summary or d['tokens']==summary[0]['tokens']
    d.update(condition=k,repeat=repeat,dynamic=dynamic)
    if child:
     audio=json.loads((folder/'audio.json').read_text());start=d['response_start_monotonic_seconds'];end=start+d['total_seconds'];runs=[r for r in audio['runs'] if r['end_monotonic']>start and r['start_monotonic']<end]
     overlap=sum(max(0,min(end,r['end_monotonic'])-max(start,r['start_monotonic'])) for r in runs);assert overlap>0
     d['audio_overlap_seconds']=overlap;d['audio_calls']=len(runs);d['audio_rtf_p50']=statistics.median(r['rtf'] for r in runs)
    summary.append(d);save(out/'summary.json',summary);print('DONE',folder.name,d['decode_tokens_per_second'],flush=True)
   finally:
    if child and child.poll() is None:child.terminate();child.wait(timeout=10)
    done.set();thread.join();save(folder/'system.json',rows)
    for log in logs:log.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('--dynamic',type=int,choices=[64,256],default=256);p.add_argument('--mode',choices=['standalone','mixed','paced']);p.add_argument('--out',required=True);p.add_argument('--kinds',default='stream,offline,para,sense');a=p.parse_args();out=ROOT/a.out;out.mkdir(exist_ok=False);kinds=a.kinds.split(',')
 if a.mode=='mixed':mixed(out,['baseline',*kinds],a.dynamic)
 elif a.mode=='paced':standalone(out,'stream',True)
 else:
  for k in kinds:print('START',k,flush=True);standalone(out,k)
if __name__=='__main__':main()
