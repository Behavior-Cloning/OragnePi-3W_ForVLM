#!/usr/bin/env python3
"""Versioned streaming dialogue chain: KWS -> GTCRN -> streaming Zipformer -> LLM."""
import json, os, sys, time, urllib.request, wave
from pathlib import Path
import numpy as np
import sherpa_onnx as sherpa
sys.path.insert(0, "/opt/opi-vlm/audio-lab-20261018")
import audio_worker as aw
sys.path.insert(0, "/opt/opi-vlm/asr-small-20261018")
import worker as sw
RATE=16000
BASE=Path("/opt/opi-vlm/audio-streaming-20261018")
ASR_ROOT=Path("/opt/opi-vlm/asr-small-20261018")
CORPUS=ASR_ROOT/"corpus"
MANIFEST=json.loads((CORPUS/"manifest.json").read_text())
SAMPLES=MANIFEST["samples"]
WAKE=BASE/"kws/candidate/tts/pos-alone-normal.wav"
OUT=Path(sys.argv[1]); OUT.mkdir(parents=True,exist_ok=False)

def read_wav(p):
    with wave.open(str(p),'rb') as f:
        assert f.getframerate()==RATE and f.getnchannels()==1 and f.getsampwidth()==2
        return np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768

def kws():
    m=BASE/"kws/model"
    return sherpa.KeywordSpotter(tokens=str(m/"tokens.txt"),encoder=str(m/"encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx"),decoder=str(m/"decoder-epoch-13-avg-2-chunk-16-left-64.onnx"),joiner=str(m/"joiner-epoch-13-avg-2-chunk-16-left-64.onnx"),keywords_file=str(BASE/"kws/candidate/keywords.txt"),num_threads=1,keywords_threshold=.25)

def wake(spotter,x):
    st=spotter.create_stream(); start=time.monotonic()
    for i in range(0,len(x),1280):
        chunk=x[i:i+1280]; due=start+(i+len(chunk))/RATE; time.sleep(max(0,due-time.monotonic())); st.accept_waveform(RATE,chunk)
        while spotter.is_ready(st): spotter.decode_stream(st)
        r=spotter.get_result(st)
        if r:return {'keyword':r,'latency_s':time.monotonic()-start}
    return {'keyword':'','latency_s':time.monotonic()-start}

def llm(text):
    body=json.dumps({'model':'qwen25-15b-q4','messages':[{'role':'system','content':'你是家庭陪伴机器人，请用简洁中文回答。'},{'role':'user','content':text}],'stream':True,'temperature':0,'max_tokens':48},ensure_ascii=False).encode()
    req=urllib.request.Request('http://127.0.0.1:8081/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
    t=time.monotonic(); first=None; parts=[]
    with urllib.request.urlopen(req,timeout=180) as resp:
        for raw in resp:
            line=raw.decode('utf-8','replace').strip()
            if not line.startswith('data: '):continue
            if line[6:]=='[DONE]':break
            o=json.loads(line[6:]); p=o.get('choices',[{}])[0].get('delta',{}).get('content','')
            if p:
                if first is None:first=time.monotonic()-t
                parts.append(p)
    return {'first_s':first,'complete_s':time.monotonic()-t,'text':''.join(parts)}

def resource():
    d={'temp_mC':int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()),'loadavg':Path('/proc/loadavg').read_text().strip()}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if 'llama-server' in (p/'cmdline').read_text(errors='ignore'):
                    for l in (p/'status').read_text().splitlines():
                        if l.startswith(('VmRSS:','VmHWM:')):d['llama_'+l.split(':')[0]]=int(l.split()[1])
                    break
            except OSError:pass
    return d

def main():
    os.sched_setaffinity(0,{6,7}); den=aw.model('denoise',1); asr=sw.model('stream',2); spot=kws(); results=[]
    for turn,sid in enumerate([0,8,16],1):
        w=wake(spot,read_wav(WAKE)); den.reset(); speech=read_wav(CORPUS/SAMPLES[sid]['file']); x=np.concatenate([speech,np.zeros(int(.8*RATE),np.float32)])
        st=asr.create_stream(); start=time.monotonic(); acc=np.zeros(0,np.float32); first_partial=None; last=''; endpoint=None; decodes=0; frame_ms=[]
        for off in range(0,len(x),256):
            b=x[off:off+256]; due=start+(off+len(b))/RATE; time.sleep(max(0,due-time.monotonic())); t0=time.perf_counter(); clean=np.asarray(den(b,RATE).samples,np.float32); frame_ms.append((time.perf_counter()-t0)*1000); acc=np.concatenate([acc,clean])
            while len(acc)>=1280:
                st.accept_waveform(RATE,acc[:1280]); acc=acc[1280:]
                while asr.is_ready(st): asr.decode_stream(st); decodes+=1
                txt=asr.get_result(st)
                if txt!=last:
                    if txt and first_partial is None:first_partial=time.monotonic()-start
                    last=txt
                if asr.is_endpoint(st) and off+256>=len(speech): endpoint=time.monotonic()-start; break
            if endpoint is not None:break
        if acc.size: st.accept_waveform(RATE,np.pad(acc,(0,1280-len(acc))))
        while asr.is_ready(st):asr.decode_stream(st)
        final=asr.get_result(st); endpoint=endpoint or time.monotonic()-start
        a=llm(final); results.append({'turn':turn,'reference':SAMPLES[sid]['text'],'asr_text':final,'wake_s':w['latency_s'],'first_partial_s':first_partial,'endpoint_s':endpoint,'endpoint_to_llm_first_s':a['first_s'],'endpoint_to_llm_complete_s':a['complete_s'],'estimated_speech_end_to_first_s':.1+a['first_s'],'denoise_frame_p95_ms':sorted(frame_ms)[max(0,int(.95*len(frame_ms))-1)],'decode_calls':decodes,'resources':resource(),'reply':a['text']})
        print(json.dumps(results[-1],ensure_ascii=False),flush=True)
    (OUT/'summary.json').write_text(json.dumps({'version':'streaming-dialogue-v1','status':'complete','chain':'KWS->GTCRN->streaming Zipformer Small->LLM','denoise':True,'asr':'sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01','results':results},ensure_ascii=False,indent=2)+'\n')
main()
