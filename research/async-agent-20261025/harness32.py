"""Continuous speech -> event-time context -> resident NPU -> preemptible mock RAG.
No camera, sound semantics, hardware motor or TTS is simulated as measured work.
"""
import os,sys,time,json,threading,queue,random,argparse,traceback,wave,hashlib
from pathlib import Path
from collections import deque
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
BASE=Path('/opt/opi-vlm/resident-agent-20261019')
sys.path[:0]=[str(BASE),str(BASE/'deps'),'/opt/opi-vlm/audio-lab-20261018/deps','/opt/opi-vlm/audio-lab-20261018','/opt/opi-vlm/asr-small-20261018']
import numpy as np
import audio_worker as aw
import worker as sw
from tokenizers import Tokenizer
from resident_worker import ResidentWorker
from agent_replay import snapshot
from context_rag import Context,compress,Retriever
RATE=16000

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=900);ap.add_argument('--out',required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 os.sched_setaffinity(0,{7})
 stop=threading.Event();ready=threading.Event();go=threading.Event();guard=threading.Lock();errors=[]
 events=(out/'events.jsonl').open('w');lk=threading.Lock();context=Context();state={'frames':0,'lag_s':0.,'max_lag_s':0.,'overruns':0,'stages':{'denoise':0.,'vad':0.,'asr':0.},'segments':0,'raw_retention_policy':'no application archive; VAD pop/reset at final; library buffer not directly measured'}
 def emit(kind,**kw):
  with lk:events.write(json.dumps({'event':kind,'at':time.monotonic(),**kw},ensure_ascii=False)+'\n');events.flush()
 def failure(where):
  errors.append({'where':where,'traceback':traceback.format_exc()});emit('error',**errors[-1]);stop.set();ready.set()
 def audio():
  try:
   os.sched_setaffinity(0,{6});load=time.monotonic();dn=aw.model('denoise',1);vad=aw.model('vad',1,.5);asr=sw.model('stream',1)
   dn(np.zeros(256,np.float32),RATE);dn.reset();stream=asr.create_stream();pending=np.zeros(0,np.float32)
   emit('audio_ready',load_s=time.monotonic()-load);ready.set();go.wait();start=time.monotonic();n=0;sid=0;f=None;silence=0;last_partial='';last_partial_at=0
   # Read only the next 16ms of prerecorded synthetic input; never queue whole utterances.
   while not stop.is_set():
    if f is None and silence==0:
     sid=sid%12+1;f=wave.open(str(BASE/'speech'/('%02d.wav'%sid)));assert f.getframerate()==RATE
     emit('source_start',source_id=sid,reference_duration_s=f.getnframes()/RATE)
    if f:
     raw=f.readframes(256)
     if not raw:f.close();f=None;silence=125;emit('source_end',source_id=sid)
     x=np.frombuffer(raw,'<i2').astype(np.float32)/32768
     if len(x)<256:x=np.pad(x,(0,256-len(x)))
    else:x=np.zeros(256,np.float32);silence-=1
    n+=1;due=start+n*.016;stop.wait(max(0,due-time.monotonic()));t=time.monotonic()
    clean=np.asarray(dn(x,RATE).samples,np.float32);u=time.monotonic();state['stages']['denoise']+=u-t
    stream.accept_waveform(RATE,clean)
    while asr.is_ready(stream):asr.decode_stream(stream)
    txt=asr.get_result(stream);v=time.monotonic();state['stages']['asr']+=v-u
    if txt!=last_partial:
     last_partial=txt;last_partial_at=v;emit('partial',text=txt)
    with guard:state['partial']=txt;state['partial_at']=last_partial_at
    pending=np.concatenate([pending,clean]);v0=time.monotonic();drain=0
    while len(pending)>=512:
     vad.accept_waveform(pending[:512]);pending=pending[512:]
     while not vad.empty():
      seg=vad.front;seglen=len(seg.samples);segstart=seg.start;vad.pop();del seg
      endpoint=time.monotonic();d=time.monotonic();stream.accept_waveform(RATE,np.zeros(8000,np.float32));stream.input_finished()
      while asr.is_ready(stream):asr.decode_stream(stream)
      text=asr.get_result(stream);done=time.monotonic();drain+=done-d;state['stages']['asr']+=done-d
      with guard:
       context.add(text,endpoint-.5-state.get('lag_s',0));state['partial']='';state['segments']+=1
      emit('asr_final',text=text,endpoint=endpoint,drain_s=done-d,segment_start_sample=segstart,segment_samples=seglen)
      stream=asr.create_stream();vad.reset();last_partial='';del text
    state['stages']['vad']+=time.monotonic()-v0-drain
    elapsed=time.monotonic()-t;lag=max(0,time.monotonic()-due)
    state.update(frames=n,lag_s=lag,max_lag_s=max(state['max_lag_s'],lag),overruns=state['overruns']+int(elapsed>.016))
    # Periodic frame distribution, including real arrival lateness.
    emit('frame',compute_s=elapsed,lag_s=lag,frame=n,application_pending_samples=len(pending))
   if f:f.close()
  except BaseException:failure('audio')
 front=threading.Thread(target=audio,daemon=True);front.start()
 tokenizer=Tokenizer.from_file(str(BASE/'tokenizer.json'));encode=lambda s:tokenizer.encode(s,add_special_tokens=False).ids
 prefix='<|im_start|>system\n你是家庭陪伴机器人的动作规划器。根据对话情景选择一个可执行动作，只输出动作短语。可用部件是头部、双臂、腰、可移动底盘；没有手指、屏幕。允许点头、摇头、转头、摆臂、弯腰、转身。你只规划，不回答用户问题，不拿取物品。没有合适动作输出保持静止。<|im_end|>\n'
 retriever=Retriever();emit('rag_ready',metadata=retriever.metadata)
 assert ready.wait(120) and not errors
 g=Path('/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018')
 w=ResidentWorker(['taskset','-c','4,5',str(BASE/'resident'),str(g),'24','1280','151674',str(g/'embeddings.f32'),'16','151645'],BASE,out/'npu.log')
 warm=list(w.generate(encode(prefix),max_new=1))[-1];emit('npu_ready',ready=w.ready,warm=warm,prefix_tokens=len(encode(prefix)))
 pending_event=queue.Queue(maxsize=1);action_q=queue.Queue(maxsize=1);running=threading.Event()
 def executor():
  active=None;deadline=0;last_tick=0
  while not stop.is_set() or not action_q.empty():
   try:job=action_q.get(timeout=.02)
   except queue.Empty:job=None
   now=time.monotonic()
   if job:
    if active:emit('action_cancel',old_id=active['id'],new_id=job['id'],preempt_s=now-job['ready_at'],reason='new_brain_plan')
    active=job;deadline=now+8;emit('action_start',**job,mock=True,duration_s=8)
   if active and now-last_tick>=.1:emit('action_tick',id=active['id']);last_tick=now
   if active and now>=deadline:emit('action_done',id=active['id']);active=None
  if active:emit('action_cancel',old_id=active['id'],reason='shutdown')
 def brain():
  try:
   os.sched_setaffinity(0,{7})
   while not stop.is_set():
    try:job=pending_event.get(timeout=.1)
    except queue.Empty:continue
    running.set();t=time.monotonic()
    # Snapshot occurs when the brain can start, not when an event joins the bounded queue.
    with guard:rows=context.snapshot(t);partial=state.get('partial','');partial_at=state.get('partial_at',0)
    c0=time.monotonic();packed=compress(rows+([{'text':partial,'at':partial_at}] if partial else []),encode,budget=32);c1=time.monotonic()
    current='' # partial is already included within the single 32-token summary budget
    dynamic=packed['text']
    assert len(encode(dynamic))<=32, 'dynamic abstraction exceeds32tokens'
    prompt=encode(prefix+'<|im_start|>user\n'+dynamic+'<|im_end|>\n<|im_start|>assistant\n')
    # Native worker reuses the longest exact cached prefix; no redundant prefix-only inference.
    prep0=time.monotonic();prepared=None;prep1=prep0
    ids=[];first=None;first_visible=None;decode_start=time.monotonic();metrics=None
    for item in w.generate(prompt,32,timeout=60):
     if item['event']=='token':
      ids.append(item['token'])
      if first is None:first=time.monotonic()
      if first_visible is None and tokenizer.decode(ids).rstrip('\ufffd').strip():first_visible=time.monotonic()
     else:metrics=item
    llm_end=time.monotonic();reply=tokenizer.decode(ids);r0=time.monotonic();actions=retriever.query(reply,k=3);r1=time.monotonic()
    action={'id':job['id'],'ready_at':r1,'action':actions[0] if actions else None}
    invalid=(metrics['finish_reason']!='eos' or not reply.strip() or any(z in reply for z in ['抱歉','无法','不能','拿起','拿取','喝茶','喝咖啡','整理书桌']) or not any(z in reply for z in ['头','臂','腰','底盘','转身','静止','挥手','摆手']))
    if invalid:emit('plan_rejected',id=job['id'],reply=reply,reason='motion_contract_v2')
    if stop.is_set():emit('shutdown_plan_discard',id=job['id'])
    elif not invalid:
     try:action_q.put_nowait(action)
     except queue.Full:
      try:emit('action_replaced_before_start',old=action_q.get_nowait())
      except queue.Empty:pass
      action_q.put_nowait(action)
    emit('brain_done',id=job['id'],trigger_at=job['at'],start=t,queue_s=t-job['at'],context=packed,context_rows=len(rows),partial=current,partial_at=partial_at,compression_s=c1-c0,prefix_restore_s=prep1-prep0,prefix_metrics=prepared,prompt_tokens=len(prompt),dynamic_body_tokens=len(encode(dynamic)),system_prefix_tokens=len(encode(prefix)),template_overhead_tokens=len(prompt)-len(encode(prefix))-len(encode(dynamic)),first_visible_s=None if first_visible is None else first_visible-decode_start,llm_first_s=None if first is None else first-decode_start,event_to_first_s=None if first is None else first-job['at'],llm_s=llm_end-decode_start,reply=reply,metrics=metrics,rag_s=r1-r0,rag_metrics=retriever.last_query,plan_rejected=invalid,top3=actions,event_to_action_s=r1-job['at'],total_s=r1-t)
    running.clear()
  except BaseException:failure('brain')
 def monitor():
  with (out/'resources.jsonl').open('w') as f:
   while not stop.wait(.5):
    row=snapshot([os.getpid(),w.process.pid]);row['frontend']=dict(state);row['event_queue']=pending_event.qsize();row['brain_busy']=running.is_set();f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    if row['temp_mC']>=78000:errors.append({'where':'temperature','value':row['temp_mC']});stop.set()
 threads=[threading.Thread(target=fn,daemon=True) for fn in (brain,executor,monitor)]
 for th in threads:th.start()
 began=time.monotonic();go.set();rng=random.Random(20261025);i=0;intervals=[]
 try:
  while not stop.is_set() and time.monotonic()-began<a.seconds:
   interval=rng.uniform(2,5);intervals.append(interval)
   if stop.wait(min(interval,max(0,a.seconds-(time.monotonic()-began)))):break
   if time.monotonic()-began>=a.seconds:break
   i+=1;job={'id':i,'at':time.monotonic()};emit('trigger',**job)
   try:pending_event.put_nowait(job)
   except queue.Full:
    try:
     old=pending_event.get_nowait();emit('event_coalesced',old_id=old['id'],new_id=i)
    except queue.Empty:pass
    pending_event.put_nowait(job)
 finally:
  stop.set()
  while not pending_event.empty():
   try:emit('shutdown_event_discard',id=pending_event.get_nowait()['id'])
   except queue.Empty:break
  front.join(15)
  for th in threads:th.join(90)
  closed=w.close();emit('closed',driver=closed)
  meta={'status':'failed' if errors or any(th.is_alive() for th in threads+[front]) else 'complete','errors':errors,'elapsed_s':time.monotonic()-began,'requested_s':a.seconds,'triggers':i,'frontend':state,'vision':'text decoder only, no image encoder','audio':'paced synthetic prerecorded Mandarin, no microphone or acoustic semantic model','policy':'latest pending event wins; running brain completes; new RAG top1 replaces mock8s action','summary_budget':32,'npu_closed':closed,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  (out/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));events.close()
if __name__=='__main__':main()
