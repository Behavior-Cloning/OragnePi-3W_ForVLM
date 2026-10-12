"""Independent llama.cpp tokenizer and full Q8 generation on the same board."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--model',required=True)
p.add_argument('--folder',type=Path,required=True);p.add_argument('--max-new',type=int,default=32)
args=p.parse_args();prompt=json.loads((args.folder/'prompt.json').read_text())
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def api(path,body=None,timeout=180):
    request=urllib.request.Request('http://127.0.0.1:18083'+path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type':'application/json'})
    with opener.open(request,timeout=timeout) as response:return json.load(response)

command=['taskset','-c','6,7','/opt/opi-vlm/bin/llama-server','--model',args.model,
         '--host','127.0.0.1','--port','18083','--ctx-size','128','--threads','2','--threads-batch','2',
         '--parallel','1','--n-gpu-layers','0','--no-warmup','--no-context-shift']
with (args.folder/'llama-cpu.log').open('w') as log:
    begin=time.monotonic();process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            if process.poll() is not None:raise RuntimeError('llama-server exited')
            try:
                if api('/health',timeout=1).get('status')=='ok':break
            except Exception:pass
            time.sleep(.2)
        else:raise TimeoutError('CPU reference not ready')
        load=time.monotonic()-begin
        tokens=api('/tokenize',{'content':prompt['rendered_prompt'],'add_special':False,'parse_special':True})['tokens']
        match=tokens==prompt['token_ids']
        if not match:
            (args.folder/'tokenizer-mismatch.json').write_text(json.dumps({'llama':tokens,'exporter':prompt['token_ids']},indent=2))
            raise RuntimeError('tokenizer mismatch; inspect before NPU generation')
        start=time.monotonic()
        response=api('/completion',{'prompt':tokens,'n_predict':args.max_new,'temperature':0,
            'seed':42,'cache_prompt':True,'return_tokens':True,'n_probs':5,'repeat_penalty':1.0})
        report={'backend':'llama.cpp CPU Q8_0','threads':2,'affinity':[6,7],
                'prompt_token_ids':tokens,
                'tokenizer_exact_match':match,'load_seconds':load,'request_seconds':time.monotonic()-start,
                'command':command,'response':response}
        (args.folder/'cpu-full-reference.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({'tokenizer_exact_match':match,'answer':response.get('content'),
                          'timings':response.get('timings'),'tokens':response.get('tokens')},ensure_ascii=False),flush=True)
    finally:
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
