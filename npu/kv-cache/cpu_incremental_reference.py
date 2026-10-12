"""Board-only isolated llama.cpp Q8 cached continuation; production stays online.

This measures the historical two-A76 configuration, not an exhaustive CPU search.
Call under the common NPU/CPU benchmark flock.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import statistics
import struct
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--prompt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--cached-prefix',type=int,required=True);p.add_argument('--steps',type=int,default=128)
p.add_argument('--context',type=int,default=1024)
p.add_argument('--capture-nonstream',action='store_true',help='capture full greedy token IDs separately; this server omits UTF-8 continuation IDs from streaming chunks')
p.add_argument('--repeats',type=int,default=5);p.add_argument('--max-start-temp-mc',type=int,default=68000)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
prompt=json.loads(a.prompt.read_text());tokens=prompt['token_ids'];assert 0<a.cached_prefix<len(tokens)
assert a.steps>0 and len(tokens)+a.steps-1<=a.context, 'context must hold prompt and all feedback positions'
(a.output/'prompt.i32').write_bytes(struct.pack('<'+'i'*len(tokens),*tokens))
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));port=18083
def api(path,body=None):
    request=urllib.request.Request(f'http://127.0.0.1:{port}'+path,
        data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with opener.open(request,timeout=120) as r:return json.load(r)
def health():
    with opener.open('http://127.0.0.1:8080/api/health',timeout=5) as r:v=json.load(r)
    assert v['ready'] and not any(v[k] for k in ('busy','loading','maintenance'))
    return v
def temperature():return int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()
binary=Path('/opt/opi-vlm/bin/llama-server')
command=['taskset','-c','6,7',str(binary),'--model',str(a.model),'--host','127.0.0.1','--port',str(port),
         '--ctx-size',str(a.context),'--threads','2','--threads-batch','2','--parallel','1','--n-gpu-layers','0',
         '--no-context-shift','--no-warmup']
report={'scope':'historical two-A76 Q8 configuration remeasured; not best-CPU search completion',
        'command':command,'model_sha256':digest(a.model),'binary_sha256':digest(binary),
        'prompt':prompt,'cached_prefix':a.cached_prefix,'steps':a.steps,'max_start_temperature_mC':a.max_start_temp_mc,'runs':[]}
request_options={'temperature':0,'seed':42,'ignore_eos':True,'repeat_penalty':1.0,'return_tokens':True}
with (a.output/'server.log').open('w') as log:
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('server exited')
            try:
                if api('/health').get('status')=='ok':break
            except (OSError,urllib.error.URLError):pass
            time.sleep(.2)
        else:raise TimeoutError('server readiness')
        assert api('/tokenize',{'content':prompt['rendered_prompt'],'add_special':False,'parse_special':True})['tokens']==tokens
        trajectory=None
        if a.capture_nonstream:
            api('/completion',{**request_options,'prompt':tokens[:a.cached_prefix],'n_predict':1,'cache_prompt':False})
            trajectory=api('/completion',{**request_options,'prompt':tokens,'n_predict':a.steps,'cache_prompt':True,'stream':False})
            assert len(trajectory['tokens'])==trajectory['timings']['predicted_n']==a.steps
            assert trajectory['timings']['cache_n']==a.cached_prefix
            report['nonstream_trajectory']=trajectory
            report['trajectory_scope']='separate same-input greedy nonstream generation; streaming emitted token positions checked, omitted IDs not observed in stream'
        for repeat in range(a.repeats+1):
            deadline=time.monotonic()+300
            while temperature()>a.max_start_temp_mc:
                if time.monotonic()>deadline:raise TimeoutError('cooldown')
                time.sleep(1)
            before=health();temp=temperature()
            prepared=api('/completion',{**request_options,'prompt':tokens[:a.cached_prefix],
                'n_predict':1,'cache_prompt':False})
            connection=http.client.HTTPConnection('127.0.0.1',port,timeout=120)
            start=time.monotonic();ttft=None;answer='';ids=[];positions=[];final=None
            try:
                connection.request('POST','/completion',json.dumps({**request_options,'prompt':tokens,
                    'n_predict':a.steps,'cache_prompt':True,'stream':True}),{'Content-Type':'application/json'})
                response=connection.getresponse();assert response.status==200
                for raw in response:
                    if not raw.startswith(b'data: '):continue
                    data=json.loads(raw[6:]);assert 'error' not in data,data
                    if data.get('content') and ttft is None:ttft=time.monotonic()-start
                    answer+=data.get('content','');ids+=data.get('tokens',[])
                    if data.get('tokens'):
                        end=data['tokens_predicted'];positions.extend(range(end-len(data['tokens']),end))
                    if data.get('stop'):final=data
            finally:connection.close()
            elapsed=time.monotonic()-start
            assert final and final['timings']['predicted_n']==a.steps,(final,len(ids))
            emitted=ids
            if trajectory is not None:
                assert len(set(positions))==len(positions) and all(0<=i<a.steps for i in positions)
                assert emitted==[trajectory['tokens'][i] for i in positions], 'stream/nonstream predictions changed'
                assert trajectory['content'].startswith(answer), 'stream/nonstream text changed'
                ids=trajectory['tokens']
            else:
                assert len(ids)==a.steps,(final,len(ids))
            timing=final['timings'];assert timing['cache_n']==a.cached_prefix and timing['prompt_n']==len(tokens)-a.cached_prefix,timing
            after=health()
            row={'phase':'warmup' if not repeat else 'measure','repeat':repeat,'ttft_seconds':ttft,
                 'total_seconds':elapsed,'timings':timing,'tokens':ids,'answer':answer,
                 'stream_tokens':emitted,'stream_token_positions':positions,
                 'temperature_before_mC':temp,'temperature_after_mC':temperature(),
                 'health_before':before,'health_after':after,'preparation_timings':prepared['timings']}
            report['runs'].append(row);print(json.dumps({k:row[k] for k in ('phase','repeat','ttft_seconds','timings')}),flush=True)
            (a.output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        measured=[r for r in report['runs'] if r['phase']=='measure']
        assert all(r['tokens']==measured[0]['tokens'] for r in measured)
        report['medians']={'ttft_seconds':statistics.median(r['ttft_seconds'] for r in measured),
            'decode_tokens_per_second':statistics.median(r['timings']['predicted_per_second'] for r in measured)}
        report['status']='complete'
        (a.output/'tokens.json').write_text(json.dumps({'tokens':measured[0]['tokens'],'fixed_feedback':False,
            'source':'llama.cpp CPU greedy '+('nonstream reference, stream-emitted positions verified' if trajectory is not None else 'streaming generation')+', EOS ignored for fixed-length work'},indent=2)+'\n')
    finally:
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        (a.output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
