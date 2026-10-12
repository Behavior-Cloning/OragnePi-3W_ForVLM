"""Board-only CPU screening; not a substitute for quality-gated final acceptance.

Run under /run/opi-npu-kv-benchmark.lock. Uses an isolated HTTP server and
fixed output length. Full reports preserve the actual generated token sequence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
p.add_argument('--prompt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--steps',type=int,default=128);p.add_argument('--repeats',type=int,default=3)
p.add_argument('--start-temperature-mc',type=int,default=55000)
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
prompt=json.loads(args.prompt.read_text());tokens=prompt['token_ids']
capacity=max(256,len(tokens)+args.steps+16)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def api(path,body=None,timeout=180):
    request=urllib.request.Request('http://127.0.0.1:18083'+path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type':'application/json'})
    with opener.open(request,timeout=timeout) as response:return json.load(response)
def temperature():return int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())
def frequencies():
    return {path.name:{name:(path/name).read_text().strip() for name in
            ('scaling_cur_freq','scaling_max_freq','scaling_governor')}
            for path in Path('/sys/devices/system/cpu/cpufreq').glob('policy*')}
def cool_down():
    deadline=time.monotonic()+300
    while temperature()>args.start_temperature_mc:
        if time.monotonic()>deadline:raise TimeoutError('board did not reach the common starting temperature')
        time.sleep(1)

configs=[(1,'6'),(2,'6,7'),(4,'4-7'),(6,'2-7'),(8,'0-7'),
         (1,'0-7'),(2,'0-7'),(4,'0-7'),(6,'0-7')]
def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
report={'scope':'CPU thread/affinity screening of one existing model; no quantization or full quality search yet',
        'model':str(args.model),'model_sha256':sha256(args.model),
        'steps':args.steps,'repeats':args.repeats,'start_temperature_limit_mC':args.start_temperature_mc,
        'prompt':prompt,'capacity':capacity,'candidates':[]}
binary='/opt/opi-vlm/bin/llama-server'
report['binary_sha256']=sha256(Path(binary))
for threads,affinity in configs:
    tag=f't{threads}-cpu{affinity.replace(",","_")}'
    command=['taskset','-c',affinity,binary,'--model',str(args.model),'--host','127.0.0.1','--port','18083',
             '--ctx-size',str(capacity),'--threads',str(threads),'--threads-batch',str(threads),
             '--parallel','1','--n-gpu-layers','0','--no-context-shift','--no-warmup']
    candidate={'threads':threads,'affinity':affinity,'command':command,'runs':[]}
    report['candidates'].append(candidate)
    with (args.output/(tag+'.log')).open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('CPU server exited')
                try:
                    if api('/health',timeout=1).get('status')=='ok':break
                except Exception:pass
                time.sleep(.2)
            else:raise TimeoutError('CPU server readiness')
            check=api('/tokenize',{'content':prompt['rendered_prompt'],'add_special':False,'parse_special':True})['tokens']
            if check!=tokens:raise ValueError('independent tokenizer mismatch')
            for repeat in range(args.repeats+1):
                cool_down();temp_before=temperature();frequency_before=frequencies()
                started=time.monotonic()
                response=api('/completion',{'prompt':tokens,'n_predict':args.steps,'temperature':0,'seed':42,
                    'ignore_eos':True,'cache_prompt':False,'return_tokens':True,'repeat_penalty':1.0})
                if response['timings']['predicted_n']!=args.steps:raise ValueError('incomplete fixed-length decode')
                if repeat:
                    candidate['runs'].append({'request_seconds':time.monotonic()-started,
                        'temperature_before_mC':temp_before,'temperature_mC':temperature(),
                        'frequency_before':frequency_before,'frequency_after':frequencies(),'response':response})
            candidate['median_decode_tps']=statistics.median(r['response']['timings']['predicted_per_second'] for r in candidate['runs'])
            print(json.dumps({'config':tag,'median_decode_tps':candidate['median_decode_tps']}),flush=True)
        finally:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            (args.output/'screening.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
