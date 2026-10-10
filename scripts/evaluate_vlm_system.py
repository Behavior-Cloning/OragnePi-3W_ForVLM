#!/usr/bin/env python3
"""Board engineering matrix: raw answers, timings and system samples; not a dataset accuracy claim."""
import argparse
import base64
import json
import http.client
import os
from pathlib import Path
import time
import compare_vlm_models as runtime
import benchmark_vlm as bench

ROOT = Path('/opt/opi-vlm')
LOCK = Path('/var/lib/opi-vlm-control/benchmark.lock')


def backend_json(path, payload):
    conn = http.client.HTTPConnection('127.0.0.1',8082,timeout=30)
    try:
        conn.request('POST',path,json.dumps(payload),{'Content-Type':'application/json'})
        response=conn.getresponse()
        value=json.loads(response.read())
        if response.status != 200: raise RuntimeError(str(value))
        return value
    finally: conn.close()


def sized_prompt(system, key, context, occupancy):
    target = min(int(context*occupancy),context-160) if occupancy else 0
    def message(count):
        return {'role':'user','content':f'Test {key} {context} {occupancy}. Secret: BLUE KITE. Background: '+
                ('A bicycle has wheels. The road is quiet. '*count)+
                '\nFirst state the secret, then explain how bicycles work in a detailed paragraph of at least 150 words.'}
    def tokens(msg):
        prompt=backend_json('/apply-template',{'messages':[system,msg],'add_generation_prompt':True})['prompt']
        return len(backend_json('/tokenize',{'content':prompt,'add_special':True})['tokens'])
    low, high = 0,context
    while low < high:
        mid=(low+high+1)//2
        if tokens(message(mid)) <= target: low=mid
        else: high=mid-1
    msg=message(low)
    return msg,tokens(msg),target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['smolvlm-256m','smolvlm-500m','internvl3-1b','qwen25vl-3b-q4'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve previous results; choose a new output')
    manifest = json.loads((ROOT/'vlm-models.json').read_text())
    system = {'role':'system','content':(ROOT/'app/system_prompt.txt').read_text().strip()}
    photo_path = ROOT/'opi-vlm-dog-ui.jpg'
    photo = {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(photo_path.read_bytes()).decode()}}
    report = {'date':time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'uname':bench.command('uname','-a'),
              'photo_sha256':runtime.digest(photo_path), 'system_prompt':system['content'], 'models':[],
              'method':{'text_repetitions':3,'steady_requests':8,'first_image_startup_conditions':['initial','cached_restart','weight_uncached'],
                        'contexts':[1024,4096,8192], 'occupancies':['short',.25,.5,.9],
                        'context_repeats':['first_prefill','cached_1','cached_2'],
                        'max_output':128,'threads':2,'affinity':[6,7],
                        'sampling_seconds':.1,'request_timeout_seconds':1200,'no_power_meter':True,
                        'limits':'Engineering sample matrix; one natural photo plus deterministic shapes and receipt. No standard dataset accuracy; startup conditions are not independent identical repetitions.'}}
    def save():
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open('x') as lock:
        lock.write(str(os.getpid()))
    runtime.run('systemctl','stop','opi-vlm-model')
    try:
        for key in args.models:
            model = manifest[key]
            folder = ROOT/'models' if key == 'smolvlm-256m' else ROOT/'models'/key
            files = [folder/f[0] for f in model['files']]
            entry = {'id':key,'manifest':model,'startups':[],'tests':[],'contexts':[]}
            report['models'].append(entry)
            def ask(case, messages, repetition=1, context=4096, extra=None):
                try:
                    result = bench.infer([system,*messages],port=8082,model=key,timeout=1200)
                except Exception as error:
                    result = {'error':str(error)}
                result.update(case=case,repetition=repetition,context=context)
                result.update(extra or {})
                recorded = json.loads(json.dumps(messages))
                for message in recorded:
                    if isinstance(message['content'],list):
                        for part in message['content']:
                            if part['type']=='image_url':
                                part['image_url']['url']='sha256:'+result.get('fixture_sha256',report['photo_sha256'])
                result['messages'] = recorded
                entry['tests'].append(result)
                save()
                print(json.dumps({'model':key,'case':case,'rep':repetition,'context':context,
                                  **{k:result[k] for k in ['answer','error','total_seconds','ttft_seconds'] if k in result}},ensure_ascii=False),flush=True)
                if 'error' in result:
                    raise RuntimeError(result['error'])
                return result
            try:
                for file,(_,size,sha) in zip(files,model['files']):
                    if file.stat().st_size != size or runtime.digest(file) != sha:
                        raise RuntimeError('Unverified weight: '+str(file))
                for mode in ['initial','cached_restart','weight_uncached']:
                    runtime.stop(); cool=runtime.cooling()
                    cache=None
                    if mode=='weight_uncached':
                        for file in files:
                            with file.open('rb') as handle:
                                os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                        cache=json.loads(bench.command('fincore','--json','--bytes','-o','FILE,SIZE,RES',*map(str,files)))
                        if any(item['res'] != 0 for item in cache['fincore']):
                            raise RuntimeError('Weight cache eviction was not complete')
                    boot=runtime.start(key,files)
                    boot.update(mode=mode,cooldown_seconds=cool,file_cache=cache)
                    entry['startups'].append(boot);save()
                    print(json.dumps({'model':key,'startup':mode,'seconds':boot['ready_seconds']}),flush=True)
                    if mode=='initial':
                        with bench.Monitor() as monitor: time.sleep(10)
                        entry['idle']={'resources':monitor.summary(),'samples':monitor.samples};save()
                        for repetition in range(1,4):
                            for label,prompt in [('text_zh','请用三句话解释自行车是怎样保持前进的。'),('text_en','Explain how a bicycle moves in three sentences.')]:
                                ask(label,[{'role':'user','content':prompt}],repetition)
                        memory=[{'role':'user','content':'我的宠物兔叫露娜，今天吃了黄焖鸡。请记住。'}]
                        answer=ask('memory_setup',memory)
                        memory += [{'role':'assistant','content':answer['answer']},{'role':'user','content':'我的宠物叫什么？我今天吃了什么？'}]
                        ask('memory_recall',memory)
                        ask('unknown',[{'role':'user','content':'我的宠物兔叫露娜。它是什么颜色？如果没有信息请直说不知道。'}])
                        for repetition,prompt in enumerate(['小明有3个苹果，又买了2个，送人1个，还剩几个？只回答数字。','12乘以3再减去5是多少？只回答数字。','盒子里有8个球，拿走一半，又放回3个，还剩几个？只回答数字。'],1):
                            ask('arithmetic',[{'role':'user','content':prompt}],repetition)
                        for repetition in range(1,9):
                            ask('steady_text',[{'role':'user','content':f'第{repetition}次请求：请用约一百字介绍一种常见交通工具及它的优缺点。'}],repetition)
                    image_message={'role':'user','content':[photo,{'type':'text','text':'Describe this image in two short sentences.'}]}
                    result=ask('first_image_'+mode,[image_message])
                    for repetition,question in enumerate(['What color is the bicycle?','请用中文描述图片里的动物，并说明它在自行车前面还是后面。','Can you read a license plate in the image? Do not guess unclear characters.'],1):
                        ask('visual_followup_'+mode,[image_message,{'role':'assistant','content':result['answer']},{'role':'user','content':question}],repetition)
                    if mode=='initial':
                        for fixture,question,expected in [
                            ('shapes','How many red circles and blue squares are there? Which color is above the other?','2 red circles, 2 blue squares; red above blue'),
                            ('receipt','Read the order number and total amount.','A17; $8')]:
                            path=ROOT/(fixture+'.jpg')
                            fixture_image={'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()}}
                            ask('visual_'+fixture,[{'role':'user','content':[fixture_image,{'type':'text','text':question}]}],
                                extra={'fixture_sha256':runtime.digest(path),'expected':expected})
                for context in [1024,4096,8192]:
                    runtime.stop();runtime.cooling()
                    boot=runtime.start(key,files,context=context)
                    with bench.Monitor() as monitor: time.sleep(5)
                    entry['contexts'].append({'context':context,'startup':boot,'idle':monitor.summary()});save()
                    for occupancy in [0,.25,.5,.9]:
                        msg,count,target=sized_prompt(system,key,context,occupancy)
                        for repetition in range(1,4):
                            ask('context_speed',[msg],repetition,context,
                                {'occupancy':occupancy,'tokenizer_input_count':count,'target_input_tokens':target,
                                 'cache_mode':'first_prefill' if repetition==1 else 'cached_repeat'})
            except Exception as error:
                entry['error']=str(error)
                entry['failure_log']=runtime.run('journalctl','-u',bench.SERVICE,'-n','80','--no-pager',check=False).stdout
                save();print(json.dumps({'model':key,'failure':str(error)},ensure_ascii=False),flush=True)
            finally:
                runtime.stop()
    finally:
        runtime.run('systemctl','start','opi-vlm-model')
        bench.ready(port=8081,timeout=180)
        LOCK.unlink(missing_ok=True)
        report['production_restored']=True
        save()
    print('SYSTEM_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
