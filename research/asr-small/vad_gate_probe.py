"""Local VAD gate smoke: gate whole clips; do not trim/change ASR inputs."""
import json,os,time
from pathlib import Path
import sherpa_onnx as s
from worker import ROOT,OLD,readwav,model,infer,stats
c=s.VadModelConfig();c.silero_vad.model=str(OLD/'models/silero_vad.onnx');c.silero_vad.min_silence_duration=.5;c.silero_vad.min_speech_duration=.25;c.sample_rate=16000;c.num_threads=1
vad=s.VoiceActivityDetector(c,buffer_size_in_seconds=60);samples=json.loads((ROOT/'corpus/manifest.json').read_text())['samples'];runs=[]
for row in samples:
 x=readwav(ROOT/'corpus'/row['file']);vad.reset();t=time.perf_counter();segments=[]
 for offset in range(0,len(x),512):
  vad.accept_waveform(x[offset:offset+512])
  while not vad.empty():seg=vad.front;segments.append({'start':seg.start,'samples':len(seg.samples)});vad.pop()
 vad.flush()
 while not vad.empty():seg=vad.front;segments.append({'start':seg.start,'samples':len(seg.samples)});vad.pop()
 runs.append({'id':row['id'],'condition':row['condition'],'seconds':time.perf_counter()-t,'pass_to_asr':bool(segments),'segments':segments})
para=model('para');smoke=[]
for i in [0,len(samples)-1]:
 row=samples[i];gate=runs[i]['pass_to_asr'];x=readwav(ROOT/'corpus'/row['file']);result=infer('para',para,x) if gate else {'text':'','seconds':0};smoke.append({'id':row['id'],'asr_called':gate,**result})
assert not runs[-1]['pass_to_asr'] and not smoke[-1]['asr_called'] and smoke[-1]['text']==''
assert all(r['pass_to_asr'] for r in runs if r['condition']=='clean')
Path('vad-gate.json').write_text(json.dumps({'vad_endpoint_silence':.5,'minimum_speech':.25,'affinity':sorted(os.sched_getaffinity(0)),'scope':'whole-clip speech-presence gate, not continuous segmentation accuracy','runs':runs,'para_smoke':smoke,'resources':stats()},ensure_ascii=False,indent=2));print('VAD_GATE_PASS',len(runs),smoke)
