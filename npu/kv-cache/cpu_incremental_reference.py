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
import os
import threading
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--prompt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--cached-prefix',type=int,required=True);p.add_argument('--steps',type=int,default=128)
p.add_argument('--context',type=int,default=1024)
p.add_argument('--capture-nonstream',action='store_true',help='capture full greedy token IDs separately; this server omits UTF-8 continuation IDs from streaming chunks')
p.add_argument('--repeats',type=int,default=5);p.add_argument('--max-start-temp-mc',type=int,default=68000)
p.add_argument('--threads',type=int,default=2)
p.add_argument('--cpu-affinity',default='6,7')
p.add_argument('--max-response-temp-mc',type=int,default=68000)
p.add_argument('--temperature-band-mc',type=int,nargs=2)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
assert 1<=a.threads<=8 and all(c in '01234567,-' for c in a.cpu_affinity)
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
    v['metrics']['web_cached_temperature_c']=v['metrics']['temperature_c']
    v['metrics']['temperature_c']=temperature()/1000
    v['temperature_sample_monotonic_seconds']=time.monotonic()
    v['temperature_source']='sysfs-direct'
    return v
def temperature():return int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())
def process_metrics(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    status=Path(f'/proc/{pid}/status').read_text().splitlines()
    values={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status if ':' in line}
    return {'cpu_seconds':(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
            'rss_kib':int(values['VmRSS'].split()[0]),'peak_rss_kib':int(values['VmHWM'].split()[0])}
def frequencies():
    return {p.name:{name:(p/name).read_text().strip() for name in
            ('scaling_cur_freq','scaling_max_freq','scaling_governor')}
            for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*')}
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()
binary=Path('/opt/opi-vlm/bin/llama-server')
command=['taskset','-c',a.cpu_affinity,str(binary),'--model',str(a.model),'--host','127.0.0.1','--port',str(port),
         '--ctx-size',str(a.context),'--threads',str(a.threads),'--threads-batch',str(a.threads),'--parallel','1','--n-gpu-layers','0',
         '--no-context-shift','--no-warmup']
report={'scope':'explicit Q8 thread/affinity configuration remeasured; not best-CPU search completion',
        'runner_revision':4,'threads':a.threads,'cpu_affinity':a.cpu_affinity,
        'runner_sha256':digest(Path(__file__)),
        'thermal_condition_sha256':digest(Path(__file__).with_name('thermal_condition.py')),
        'thermal_heater_sha256':digest(Path(__file__).with_name('thermal_heater')),
        'max_response_temperature_mC':a.max_response_temp_mc,
        'temperature_band_mC':a.temperature_band_mc,
        'ttft_scope':'client first SSE event reporting a generated token; first visible text separately recorded; buffering can add delay',
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
            observations=[];observation_errors=[];stop=threading.Event();observe_start=time.monotonic()
            def sample():
                observations.append({'elapsed':time.monotonic()-observe_start,'health':health(),
                                     'process':process_metrics(process.pid),'frequencies':frequencies()})
            def observe():
                while not stop.wait(1):
                    try:sample()
                    except Exception as exc:observation_errors.append(str(exc))
            sample();observer=threading.Thread(target=observe,daemon=True);observer.start()
            prepare_start=time.monotonic()
            prepared=api('/completion',{**request_options,'prompt':tokens[:a.cached_prefix],
                'n_predict':1,'cache_prompt':False})
            preparation_seconds=time.monotonic()-prepare_start
            cool_start=time.monotonic()
            thermal=None
            if a.temperature_band_mc:
                from thermal_condition import condition
                thermal=condition(*a.temperature_band_mc)
            else:
                while temperature()>a.max_response_temp_mc:
                    if time.monotonic()-cool_start>300:raise TimeoutError('response cooldown')
                    time.sleep(1)
            response_cooldown_seconds=time.monotonic()-cool_start
            response_start_temp=temperature()
            response_frequency=frequencies()
            process_before=process_metrics(process.pid)
            connection=http.client.HTTPConnection('127.0.0.1',port,timeout=120)
            start=time.monotonic();response_observation_start=start-observe_start
            ttft=None;visible_ttft=None;answer='';ids=[];positions=[];final=None
            try:
                connection.request('POST','/completion',json.dumps({**request_options,'prompt':tokens,
                    'n_predict':a.steps,'cache_prompt':True,'stream':True}),{'Content-Type':'application/json'})
                response=connection.getresponse();assert response.status==200
                for raw in response:
                    if not raw.startswith(b'data: '):continue
                    data=json.loads(raw[6:]);assert 'error' not in data,data
                    if ttft is None and (data.get('tokens_predicted',0)>0 or data.get('tokens') or data.get('content')):ttft=time.monotonic()-start
                    if data.get('content') and visible_ttft is None:visible_ttft=time.monotonic()-start
                    answer+=data.get('content','');ids+=data.get('tokens',[])
                    if data.get('tokens'):
                        end=data['tokens_predicted'];positions.extend(range(end-len(data['tokens']),end))
                    if data.get('stop'):final=data
            finally:
                connection.close();stop.set();observer.join()
            elapsed=time.monotonic()-start
            process_after=process_metrics(process.pid);sample()
            assert not observation_errors,observation_errors
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
                 'first_visible_text_seconds':visible_ttft,'system_samples':observations,
                 'response_start_temperature_mC':response_start_temp,'response_start_frequencies':response_frequency,
                 'response_observation_start_seconds':response_observation_start,
                 'thermal_conditioning':thermal,
                 'response_end_frequencies':frequencies(),'response_cooldown_seconds':response_cooldown_seconds,
                 'preparation_seconds':preparation_seconds,'response_cpu_seconds':process_after['cpu_seconds']-process_before['cpu_seconds'],
                 'process_before':process_before,'process_after':process_after,
                 'total_seconds':elapsed,'timings':timing,'tokens':ids,'answer':answer,
                 'stream_tokens':emitted,'stream_token_positions':positions,
                 'temperature_before_mC':temp,'temperature_after_mC':temperature(),
                 'health_before':before,'health_after':after,'preparation_timings':prepared['timings']}
            report['runs'].append(row);print(json.dumps({k:row[k] for k in ('phase','repeat','ttft_seconds','timings')}),flush=True)
            (a.output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        measured=[r for r in report['runs'] if r['phase']=='measure']
        assert all(r['tokens']==measured[0]['tokens'] for r in measured)
        report['medians']={'ttft_seconds':statistics.median(r['ttft_seconds'] for r in measured),
            'decode_tokens_per_second':statistics.median(r['timings']['predicted_per_second'] for r in measured),
            'total_seconds':statistics.median(r['total_seconds'] for r in measured),
            'response_cpu_seconds':statistics.median(r['response_cpu_seconds'] for r in measured),
            'preparation_seconds':statistics.median(r['preparation_seconds'] for r in measured)}
        report['status']='complete'
        (a.output/'tokens.json').write_text(json.dumps({'tokens':measured[0]['tokens'],'fixed_feedback':False,
            'source':'llama.cpp CPU greedy '+('nonstream reference, stream-emitted positions verified' if trajectory is not None else 'streaming generation')+', EOS ignored for fixed-length work'},indent=2)+'\n')
    finally:
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        (a.output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
