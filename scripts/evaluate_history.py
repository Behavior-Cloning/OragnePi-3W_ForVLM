#!/usr/bin/env python3
"""Text-only history-length matrix with native token counts and fixed decode length."""
import argparse
import hashlib
import gzip
import http.client
import json
import os
from pathlib import Path
import signal
import time
import benchmark_vlm as bench
import compare_vlm_models as runtime

ROOT = Path('/opt/opi-vlm')
LOCK = Path('/var/lib/opi-vlm-control/benchmark.lock')
SYSTEM = '你是小橙，用户的桌边陪伴机器人。用自然中文回应，记住对话，不编造真实经历。'
TOPICS = ['下班路上的晚霞', '刚修好的程序', '窗边的雨声', '周末想读的书', '准备做的晚饭', '朋友分享的照片', '散步时的小猫', '今天听到的音乐']


def api(path, payload):
    conn = http.client.HTTPConnection('127.0.0.1', runtime.PORT, timeout=60)
    try:
        conn.request('POST', path, json.dumps(payload), {'Content-Type': 'application/json'})
        response = conn.getresponse()
        data = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(str(data))
        return data
    finally:
        conn.close()


def history(count):
    messages = [{'role': 'system', 'content': SYSTEM}]
    for i in range(count):
        topic = TOPICS[i % len(TOPICS)]
        messages += [{'role': 'user', 'content': f'第{i+1}件小事：我想聊聊{topic}，今天忙完之后终于有时间休息了。'},
                     {'role': 'assistant', 'content': f'那就慢慢聊{topic}吧。我在听，先把最想说的那一点告诉我。'}]
    messages.append({'role': 'user', 'content': '接着陪我聊聊，描述一个轻松的傍晚，可以说得详细一些。'})
    return messages


def render(messages):
    text = api('/apply-template', {'messages': messages, 'add_generation_prompt': True})['prompt']
    tokens = api('/tokenize', {'content': text, 'add_special': True, 'parse_special': True})['tokens']
    return text, tokens


def sized(context, occupancy, max_output):
    target = min(int(context * occupancy), context-max_output-8) if occupancy else 0
    low, high = 0, context // 16
    while occupancy and low < high:
        mid = (low+high+1)//2
        if len(render(history(mid))[1]) <= target:
            low = mid
        else:
            high = mid-1
    messages = history(low)
    prompt, tokens = render(messages)
    if len(tokens) + max_output >= context:
        raise ValueError('Base prompt cannot fit in context')
    return {'messages': messages, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'prompt_tokens': tokens, 'input_tokens': len(tokens), 'history_turn_pairs': low,
            'target_input_tokens': target, 'actual_occupancy': len(tokens)/context}


def infer(tokens, cached, max_output):
    # Pre-tokenized prompt means the exact input count is known, including chat template.
    payload = {'prompt': tokens, 'stream': True, 'n_predict': max_output, 'ignore_eos': True,
               'temperature': 0, 'seed': 42, 'cache_prompt': cached, 'repeat_penalty': 1.0}
    result = {'ttft_seconds': None, 'answer': '', 'cache_prompt': cached}
    conn = http.client.HTTPConnection('127.0.0.1', runtime.PORT, timeout=1800)
    with bench.Monitor() as monitor:
        start = time.monotonic()
        try:
            conn.request('POST', '/completion', json.dumps(payload), {'Content-Type': 'application/json'})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f'{response.status}: {response.read().decode()}')
            done = False
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    break
                item = json.loads(data)
                if item.get('error'):
                    raise RuntimeError(str(item['error']))
                if item.get('content'):
                    if result['ttft_seconds'] is None:
                        result['ttft_seconds'] = time.monotonic()-start
                    result['answer'] += item['content']
                if item.get('stop'):
                    result['final'] = item
                    result['timings'] = item.get('timings', {})
                    done = True
            if not done:
                raise RuntimeError('Incomplete native SSE response')
            final = result['final']
            if final.get('tokens_predicted') != max_output:
                raise RuntimeError(f"Expected {max_output} generated tokens, got {final.get('tokens_predicted')}")
            if final.get('tokens_evaluated') != len(tokens):
                raise RuntimeError('Native input count differs from pre-tokenized prompt')
            cache_n = result['timings'].get('cache_n')
            prompt_n = result['timings'].get('prompt_n')
            if not cached and (cache_n != 0 or prompt_n != len(tokens)):
                raise RuntimeError(f'Uncached prefill was not complete: cache={cache_n}, prompt={prompt_n}')
            if cached and (cache_n is None or cache_n <= 0):
                raise RuntimeError('Expected prefix reuse but none was observed')
        except Exception as error:
            result['error'] = str(error)
        finally:
            conn.close()
    result['total_seconds'] = time.monotonic()-start
    result['resources'] = monitor.summary()
    result['samples'] = monitor.samples
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['smolvlm-256m','smolvlm-500m','internvl3-1b','qwen25-15b-q4','qwen3-17b-q4','qwen25vl-3b-q4'])
    parser.add_argument('--contexts', nargs='+', type=int, default=[1024,4096,8192])
    parser.add_argument('--occupancies', nargs='+', type=float, default=[0,.25,.5,.9])
    parser.add_argument('--max-output', type=int, default=128)
    parser.add_argument('--cooling-label', default='unspecified')
    parser.add_argument('--incremental', action='store_true', help='Append actual reply and one new user message after cached replay')
    parser.add_argument('--kv-only', action='store_true', help='Only measure incremental followup; initial history loading is unscored setup')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.kv_only:
        args.incremental = True
    if args.output.exists():
        raise ValueError('Preserve existing results; choose a fresh path')
    manifest = json.loads((ROOT/'vlm-models.json').read_text())
    assert all(k in manifest for k in args.models)
    assert all(c in [1024,2048,4096,8192] for c in args.contexts)
    assert all(0 <= o <= .9 for o in args.occupancies)
    report = {'status':'RUNNING','models':[], 'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'method':{'cooling_condition':args.cooling_label,'incremental_enabled':args.incremental,'contexts':args.contexts,'occupancies':args.occupancies,'fixed_output_tokens':args.max_output,
                        'system_prompt':SYSTEM,'threads':2,'cpu_affinity':[6,7], 'sample_interval_seconds':.1,
                        'cache_modes':['uncached_full_prefill','cached_identical_replay'],
                        'notes':['Synthetic fixed Chinese multi-turn history, not quality evaluation.',
                                 'Select whole turn pairs below target; report actual token count and occupancy.',
                                 'Greedy decoding, ignore_eos=true forces equal output length; answers not scored.',
                                 'Cached replay is optimistic same-request prefix reuse, not a fresh conversation.',
                                 'One observation per cell; no statistical confidence claims.',
                                 'VLM projectors remain loaded as in web deployment; no image requests.',
                                 'CPU only; no electrical power measurement; startup has warm OS file cache.']}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.kv_only:
        report['method']['cache_modes']=['cached_new_user_turn']
        report['method']['notes']=[
            'Fan-installed; synthetic Chinese multi-turn history; CPU cores 6/7.',
            'Initial history and 32 generated tokens are unscored setup, not a prefill benchmark.',
            'Append actual setup response and a fixed new user sentence; measure 128 generated tokens.',
            'History size reserves space for setup response, new question and measured response; actual tokens reported.',
            'Verify positive cache_n. Chat-template changes can require recomputing some suffix tokens.',
            'One observation per cell; no electrical power measurement; no vision requests.']
    def save():
        temp=args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report,ensure_ascii=False,indent=2));temp.replace(args.output)
    def interrupted(signum,frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with LOCK.open('x') as handle:handle.write(str(os.getpid()))
    try:
        runtime.run('systemctl','stop','opi-vlm-model')
        for key in args.models:
            folder=ROOT/'models' if key=='smolvlm-256m' else ROOT/'models'/key
            files=[folder/f[0] for f in manifest[key]['files']]
            model={'id':key,'manifest':manifest[key],'contexts':[]};report['models'].append(model);save()
            try:
                for file,(_,size,sha) in zip(files,manifest[key]['files']):
                    if file.stat().st_size!=size or runtime.digest(file)!=sha:raise RuntimeError('Unverified weight')
                for context in args.contexts:
                    entry={'context':context,'cases':[]};model['contexts'].append(entry);save()
                    try:
                        runtime.stop();entry['cooldown_seconds']=runtime.cooling()
                        entry['startup']=runtime.start(key,files,context)
                        with bench.Monitor() as idle:time.sleep(3)
                        entry['idle']={'resources':idle.summary(),'samples':idle.samples}
                        for occupancy in args.occupancies:
                            case={'requested_occupancy':occupancy,'runs':[]};entry['cases'].append(case);save()
                            case.update(sized(context,occupancy,args.max_output+(160 if args.kv_only else 0)))
                            sample_dir=args.output.with_suffix('.samples');sample_dir.mkdir(exist_ok=True)
                            if args.kv_only:
                                preparation=infer(case['prompt_tokens'],False,32)
                                if preparation.get('error'):raise RuntimeError(preparation['error'])
                                case['setup']={'answer':preparation['answer'],'input_tokens':case['input_tokens'],
                                               'output_tokens':32,'scored':False}
                            for cached in ([] if args.kv_only else [False,True]):
                                result=infer(case['prompt_tokens'],cached,args.max_output)
                                samples=result.pop('samples')
                                freqs=[s['cpu_freq_khz']['policy6'] for s in samples if 'policy6' in s.get('cpu_freq_khz',{})]
                                result['a76_frequency_khz']={'min':min(freqs),'max':max(freqs)} if freqs else None
                                sample_dir=args.output.with_suffix('.samples');sample_dir.mkdir(exist_ok=True)
                                sample_file=sample_dir/f'{key}-{context}-{occupancy}-{int(cached)}.json.gz'
                                with gzip.open(sample_file,'wt',encoding='utf-8') as stream:json.dump(samples,stream)
                                result['samples_file']=str(sample_file)
                                case['runs'].append(result);save()
                                print(json.dumps({'model':key,'context':context,'target':occupancy,'input':case['input_tokens'],
                                                  'cache':cached,'ttft':result['ttft_seconds'],
                                                  'tps':result.get('timings',{}).get('predicted_per_second'),
                                                  'error':result.get('error')},ensure_ascii=False),flush=True)
                                if result.get('error'):raise RuntimeError(result['error'])
                            if args.incremental:
                                messages=case['messages']+[{'role':'assistant','content':case['setup']['answer'] if args.kv_only else case['runs'][-1]['answer']},
                                                          {'role':'user','content':'我想再听一点，接着说吧。'}]
                                prompt,tokens=render(messages)
                                case['incremental']={'messages':messages,'input_tokens':len(tokens),
                                                     'added_input_tokens':len(tokens)-case['input_tokens']}
                                if len(tokens)+args.max_output>=context:
                                    case['incremental']['status']='SKIPPED_NO_ROOM_NO_TRUNCATION'
                                else:
                                    result=infer(tokens,True,args.max_output)
                                    samples=result.pop('samples')
                                    freqs=[s['cpu_freq_khz']['policy6'] for s in samples if 'policy6' in s.get('cpu_freq_khz',{})]
                                    result['a76_frequency_khz']={'min':min(freqs),'max':max(freqs)} if freqs else None
                                    sample_file=sample_dir/f'{key}-{context}-{occupancy}-incremental.json.gz'
                                    with gzip.open(sample_file,'wt',encoding='utf-8') as stream:json.dump(samples,stream)
                                    result['samples_file']=str(sample_file)
                                    case['incremental']['run']=result
                                    case['incremental']['status']='ERROR' if result.get('error') else 'COMPLETE'
                                    print(json.dumps({'model':key,'context':context,'target':occupancy,'incremental':True,
                                                      'input':len(tokens),'timings':result.get('timings'),
                                                      'ttft':result['ttft_seconds'],'error':result.get('error')},ensure_ascii=False),flush=True)
                                save()
                                if case['incremental']['status']=='ERROR':raise RuntimeError(result['error'])
                    except Exception as error:
                        entry['error']=str(error);save()
                    finally:runtime.stop()
            except Exception as error:
                model['error']=str(error);save()
        report['status']='COMPLETE_WITH_ERRORS' if any(m.get('error') or any(c.get('error') for c in m['contexts']) for m in report['models']) else 'COMPLETE'
    except BaseException:
        report['status']='INTERRUPTED';raise
    finally:
        try:
            runtime.stop();runtime.run('systemctl','start','opi-vlm-model');bench.ready(port=8081,timeout=180)
            report['production_restored']=True
        finally:LOCK.unlink(missing_ok=True);save()


if __name__=='__main__':main()
