import json,os,sys,time,statistics,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'/opt/opi-vlm/audio-lab-20261018');import audio_worker as aw
sys.path.insert(0,'/opt/opi-vlm/asr-small-20261018');import worker as sw
ROOT=Path('/opt/opi-vlm/asr-small-20261018');CORP=ROOT/'corpus';M=json.loads((CORP/'manifest.json').read_text());OUT=Path('/opt/opi-vlm/audio-streaming-20261018/vad-chain');OUT.mkdir(parents=True,exist_ok=False)
INDICES=[0,2,8,9,16,20];SILENCE=.5;FRAME=256;RATE=16000
os.sched_setaffinity(0,{2,3})
# All models resident before timing. GTCRN -> independent Silero VAD -> Paraformer / online Zipformer.
d=aw.model('denoise',1);v=aw.model('vad',1,SILENCE);para=sw.model('para',1);online=sw.model('stream',1)
def concat_text(xs):return ''.join(x for x in xs if x)
def play(idx):
 row=M['samples'][idx];x=sw.readwav(CORP/row['file']);x=np.concatenate([x,np.zeros(int(.8*RATE),np.float32)])
 v.reset();d.reset();stream=online.create_stream();vad_buffer=np.zeros(0,np.float32);asr_buffer=np.zeros(0,np.float32);segrows=[];partials=[];frames=[];started=time.monotonic();speech_end_proxy=None;endpoint=None;para_parts=[];stream_parts=[]
 for p in range(0,len(x),FRAME):
  block=x[p:p+FRAME];due=started+(p+len(block))/RATE;time.sleep(max(0,due-time.monotonic()));arrive=time.monotonic();t0=time.perf_counter();y=np.asarray(d(block,RATE).samples,np.float32);done=time.perf_counter();frames.append((done-t0)*1000)
  vad_buffer=np.concatenate([vad_buffer,y]);asr_buffer=np.concatenate([asr_buffer,y])
  while len(asr_buffer)>=1280:
   z=asr_buffer[:1280];asr_buffer=asr_buffer[1280:];stream.accept_waveform(RATE,z)
   while online.is_ready(stream):online.decode_stream(stream)
   txt=online.get_result(stream)
   if txt and (not partials or partials[-1]['text']!=txt):partials.append({'at':time.monotonic(),'text':txt,'audio_seconds':(p+len(block))/RATE})
  while len(vad_buffer)>=512:
   z=vad_buffer[:512];vad_buffer=vad_buffer[512:];v.accept_waveform(z)
   while not v.empty():
    seg=v.front;ss=np.asarray(seg.samples,np.float32).copy();seg_end=(seg.start+len(ss))/RATE;event=time.monotonic();v.pop();speech_end_proxy=seg_end;endpoint=event
    q=para.create_stream();q.accept_waveform(RATE,ss);t=time.perf_counter();para.decode_stream(q);parat=time.perf_counter()-t;para_parts.append(q.result.text)
    stream.input_finished()
    while online.is_ready(stream):online.decode_stream(stream)
    stxt=online.get_result(stream);stream_parts.append(stxt);segrows.append({'start_audio_seconds':seg.start/RATE,'vad_speech_end_proxy_seconds':seg_end,'vad_endpoint_monotonic':event,'vad_endpoint_delay_from_speech_end_proxy_seconds':event-(started+seg_end),'vad_segment_seconds':len(ss)/RATE,'para_text':q.result.text,'para_seconds':parat,'stream_final_text':stxt,'stream_vad_endpoint_to_final_seconds':time.monotonic()-event})
    stream=online.create_stream()
 while len(vad_buffer)>0:
  z=np.pad(vad_buffer[:512],(0,512-len(vad_buffer[:512])));vad_buffer=vad_buffer[512:];v.accept_waveform(z)
  while not v.empty():
   seg=v.front;ss=np.asarray(seg.samples,np.float32).copy();seg_end=(seg.start+len(ss))/RATE;event=time.monotonic();v.pop();speech_end_proxy=seg_end;endpoint=event
   q=para.create_stream();q.accept_waveform(RATE,ss);t=time.perf_counter();para.decode_stream(q);parat=time.perf_counter()-t;para_parts.append(q.result.text)
   stream.input_finished()
   while online.is_ready(stream):online.decode_stream(stream)
   stxt=online.get_result(stream);stream_parts.append(stxt);segrows.append({'start_audio_seconds':seg.start/RATE,'vad_speech_end_proxy_seconds':seg_end,'vad_endpoint_monotonic':event,'vad_endpoint_delay_from_speech_end_proxy_seconds':event-(started+seg_end),'vad_segment_seconds':len(ss)/RATE,'para_text':q.result.text,'para_seconds':parat,'stream_final_text':stxt,'stream_vad_endpoint_to_final_seconds':time.monotonic()-event});stream=online.create_stream()
 v.flush()
 while not v.empty():
  seg=v.front;ss=np.asarray(seg.samples,np.float32).copy();seg_end=(seg.start+len(ss))/RATE;event=time.monotonic();v.pop();speech_end_proxy=seg_end;endpoint=event
  q=para.create_stream();q.accept_waveform(RATE,ss);t=time.perf_counter();para.decode_stream(q);parat=time.perf_counter()-t;para_parts.append(q.result.text)
  stream.input_finished()
  while online.is_ready(stream):online.decode_stream(stream)
  stxt=online.get_result(stream);stream_parts.append(stxt);segrows.append({'start_audio_seconds':seg.start/RATE,'vad_speech_end_proxy_seconds':seg_end,'vad_endpoint_monotonic':event,'vad_endpoint_delay_from_speech_end_proxy_seconds':event-(started+seg_end),'vad_segment_seconds':len(ss)/RATE,'para_text':q.result.text,'para_seconds':parat,'stream_final_text':stxt,'stream_vad_endpoint_to_final_seconds':time.monotonic()-event});stream=online.create_stream()
 if not segrows:raise RuntimeError('no VAD segments '+row['id'])
 endproxy=started+speech_end_proxy;firstp=partials[0] if partials else None;last_para=segrows[-1];last_stream=segrows[-1]
 return {'id':row['id'],'reference':row['text'],'clip_seconds':len(x)/RATE-.8,'vad_silence_threshold_seconds':SILENCE,'speech_end_reference':'VAD extracted speech segment end sample, timestamped on shared 16ms denoised timeline; endpoint callback arrives after trailing silence. This is a model-derived voice-end proxy, not human phonetic annotation.','speech_end_proxy_audio_seconds':speech_end_proxy,'vad_endpoint_audio_seconds':(endpoint-started),'vad_endpoint_delay_seconds':endpoint-endproxy,'segments':segrows,'para_text':concat_text(para_parts),'para_first_result_after_voice_end_seconds':segrows[0]['vad_endpoint_monotonic']-endproxy+segrows[0]['para_seconds'],'para_final_after_voice_end_seconds':last_para['vad_endpoint_monotonic']-endproxy+last_para['para_seconds'],'stream_text':concat_text(stream_parts),'stream_first_partial_after_voice_end_seconds':firstp['at']-endproxy if firstp else None,'stream_first_partial_audio_seconds':firstp['audio_seconds'] if firstp else None,'stream_final_after_voice_end_seconds':last_stream['vad_endpoint_monotonic']-endproxy+last_stream['stream_vad_endpoint_to_final_seconds'],'stream_persisted_partials':partials,'frontend_frame_ms_p50':statistics.median(frames),'frontend_frame_ms_p95':aw.quantile(frames,.95),'frontend_frame_ms_max':max(frames),'frames_over_16ms':sum(t>16 for t in frames)}
rows=[play(i) for i in INDICES]
summary={'status':'complete','runtime':'sherpa-onnx 1.13.8','affinity':sorted(os.sched_getaffinity(0)),'threads':1,'chunking':'GTCRN256 samples/16ms, VAD512/32ms, online ASR1280/80ms','VAD_endpoint':'Silero min silence 500ms; segments finalize Paraformer and flush online Zipformer','scope':'6 public AISHELL-derived Mandarin utterances replayed in real time with 800ms appended silence; no microphone, echo cancel or NPU LLM concurrency','timing_reference':'model-derived VAD speech-segment end sample on shared timeline; endpoint/trigger delayed by silence detection; not manually labeled ground truth','rows':rows}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False));print(json.dumps([{k:r[k] for k in ['id','para_first_result_after_voice_end_seconds','para_final_after_voice_end_seconds','stream_first_partial_after_voice_end_seconds','stream_final_after_voice_end_seconds','vad_endpoint_delay_seconds','para_text','stream_text','reference']} for r in rows],ensure_ascii=False))
