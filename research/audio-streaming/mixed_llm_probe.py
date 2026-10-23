import json,os,sys,time,shlex,subprocess,threading,statistics
from pathlib import Path
sys.path.insert(0,'/opt/opi-vlm/audio-lab-20261018');import audio_worker as aw
sys.path.insert(0,'/opt/opi-vlm/asr-small-20261018');import worker as sw
sys.path.insert(0,'/opt/opi-vlm/audio-streaming-20261018');import board_trials as bt
ROOT=Path('/opt/opi-vlm/audio-streaming-20261018');OUT=ROOT/'llm-mixed';OUT.mkdir(parents=True,exist_ok=False);NPU=bt.NPU;PREFIX=bt.PREFIX
# The workload workers are reused between D64 and D256 in the mixed condition.
ASR_SCRIPT=ROOT/'mixed_asr_worker.py';KWS_SCRIPT=ROOT/'mixed_kws_worker.py'
ASR_SCRIPT.write_text('''import os,time,json,sys\nfrom pathlib import Path\nsys.path.insert(0,"/opt/opi-vlm/asr-small-20261018");import worker as w\nR=Path("/opt/opi-vlm/asr-small-20261018");m=w.model("para",1);Path(sys.argv[1]).write_text("ready")\nD=json.loads((R/"corpus/manifest.json").read_text());ids=[0,2,8,9,16,20];i=0;stop=Path(sys.argv[2]);rows=[]\nwhile not stop.exists():\n q=D["samples"][ids[i%len(ids)]];x=w.readwav(R/"corpus"/q["file"]);s=m.create_stream();s.accept_waveform(16000,x);t=time.monotonic();m.decode_stream(s);rows.append(time.monotonic()-t);i+=1\nPath(sys.argv[3]).write_text(json.dumps({"calls":i,"seconds":rows}))\n''')
KWS_SCRIPT.write_text('''import os,time,json,sys,wave\nfrom pathlib import Path\nimport numpy as np, sherpa_onnx as s\nD=Path("/opt/opi-vlm/audio-streaming-20261018/kws");M=D/"model";C=D/"candidate";kw=s.KeywordSpotter(tokens=str(M/"tokens.txt"),encoder=str(M/"encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx"),decoder=str(M/"decoder-epoch-13-avg-2-chunk-16-left-64.onnx"),joiner=str(M/"joiner-epoch-13-avg-2-chunk-16-left-64.onnx"),keywords_file=str(C/"keywords.txt"),num_threads=1,keywords_threshold=.25);files=sorted((C/"tts").glob("*.wav"))+sorted((C/"corpus").glob("*.wav"));Path(sys.argv[1]).write_text("ready");stop=Path(sys.argv[2]);calls=0;elapsed=[]\nwhile not stop.exists():\n f=files[calls%len(files)]\n with wave.open(str(f),"rb") as z:x=np.frombuffer(z.readframes(z.getnframes()),"<i2").astype(np.float32)/32768\n st=kw.create_stream();t=time.monotonic()\n for i in range(0,len(x),1280):\n  st.accept_waveform(16000,x[i:i+1280])\n  while kw.is_ready(st):kw.decode_stream(st)\n  kw.get_result(st)\n elapsed.append(time.monotonic()-t);calls+=1\nPath(sys.argv[3]).write_text(json.dumps({"calls":calls,"seconds":elapsed}))\n''')
def temp():return int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())
def run(dynamic,mixed):
 name=('mixed' if mixed else 'baseline')+f'-d{dynamic}';fd=OUT/name;fd.mkdir();starttemp=temp();children=[];logs=[];done=threading.Event();pids=[];resources=[]
 mt=threading.Thread(target=bt.monitor,args=(done,pids,resources));mt.start();ready=[]
 try:
  if mixed:
   for label,script,cpu in [('para',ASR_SCRIPT,2),('kws',KWS_SCRIPT,3)]:
    rdy=fd/(label+'.ready');stp=fd/(label+'.stop');res=fd/(label+'.json');lg=(fd/(label+'.log')).open('w');logs.append(lg)
    c=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,str(script),str(rdy),str(stp),str(res)],stdout=lg,stderr=subprocess.STDOUT);children.append((c,stp,res,label));pids.append(c.pid);ready.append(rdy)
   limit=time.monotonic()+90
   while not all(x.exists() for x in ready):
    if time.monotonic()>limit:raise TimeoutError('audio worker load')
    time.sleep(.05)
  prompt='/opt/opi-vlm/short-prefix-d64-20261018/p64-d64.i32' if dynamic==64 else '/opt/opi-vlm/short-prefix-20261018/p64-d256.i32'
  env={k:v for k,v in os.environ.items() if not k.startswith('OPI_')};env.update(PYTHONPATH='/opt/opi-vlm/audio-lab-20261018/deps',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',OPI_PERFORMANCE='1',OPI_DECODER_GROUP='12',OPI_FAST_GREEDY='1',OPI_GROUP_HEAD='1',OPI_PREFILL_TOKENS='16',OPI_CACHED_PREFIX='64',OPI_MAX_START_TEMP_MC='70000',OPI_FEEDBACK_FILE=str(NPU/(PREFIX+'-feedback.i32')))
  cmd=['taskset','-c','6,7','./'+PREFIX+'-full_model','.','24','1280','151674','embeddings.f32',prompt,'128',str(fd/'llm.json'),str(fd/'unused.f32'),'151645']
  with (fd/'llm.log').open('w') as log:
   p=subprocess.Popen(cmd,cwd=NPU,env=env,stdout=log,stderr=subprocess.STDOUT);pids.append(p.pid);code=p.wait(timeout=120)
  if code:raise RuntimeError('LLM exit '+str(code))
  d=json.loads((fd/'llm.json').read_text());assert d['generated_tokens']==128 and d['head_on_npu'] and d['driver_after_release_bytes']==0
  d.update(condition=name,dynamic=dynamic,start_temp_mC=starttemp,end_temp_mC=temp());
  for c,stp,res,label in children:stp.write_text('stop')
  for c,stp,res,label in children:c.wait(timeout=60);assert c.returncode==0;d[label]=json.loads(res.read_text())
  (fd/'summary.json').write_text(json.dumps(d,indent=2));return {k:d[k] for k in ['condition','dynamic','start_temp_mC','end_temp_mC','ttft_seconds','decode_tokens_per_second','driver_after_release_bytes','para','kws'] if k in d}
 finally:
  for c,stp,res,label in children:
   if c.poll() is None:stp.write_text('stop');c.terminate();c.wait()
  done.set();mt.join()
  for f in logs:f.close()
rows=[run(64,False),run(64,True),run(256,False),run(256,True)]
(OUT/'summary.json').write_text(json.dumps({'status':'complete','runs':rows,'scope':'fresh A55 Paraformer CPU2 + wake-word KWS CPU3 concurrent with host-side NPU LLM CPU6,7; one rep per dynamic size; no thermal preheat; temperatures recorded; baseline serialized before mixed case'},ensure_ascii=False,indent=2)+'\n');print(json.dumps(rows,ensure_ascii=False))
