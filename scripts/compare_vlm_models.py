#!/usr/bin/env python3
"""Run real A733 model comparisons sequentially; restore production service afterwards."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import benchmark_vlm as bench

ROOT = Path('/opt/opi-vlm')
bench.SERVICE = 'opi-vlm-eval.service'
PORT = 8082


def run(*args, check=True):
    return subprocess.run(args, check=check, text=True, capture_output=True)


def digest(file):
    checksum = hashlib.sha256()
    with file.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def cooling(timeout=240):
    start = time.monotonic()
    while int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) > 58000:
        if time.monotonic() - start > timeout:
            raise RuntimeError(f'Board did not cool below 58C within {timeout}s')
        time.sleep(3)
    return time.monotonic() - start


def stop():
    run('systemctl', 'stop', bench.SERVICE, check=False)
    run('systemctl', 'reset-failed', bench.SERVICE, check=False)
    for _ in range(30):
        if run('systemctl', 'show', bench.SERVICE, '-p', 'LoadState', '--value', check=False).stdout.strip() == 'not-found':
            return
        time.sleep(.1)


def start(key, files, context=4096):
    args = [str(ROOT/'bin/llama-server'), '--model', str(files[0]),
            '--alias', key, '--host', '127.0.0.1', '--port', str(PORT), '--ctx-size', str(context),
            '--parallel', '1', '--threads', '2', '--threads-batch', '2', '--n-gpu-layers', '0',
            '--no-mmproj-offload', '--cache-ram', '128', '--no-context-shift']
    if len(files)>1:
        args += ['--mmproj',str(files[1])]
    if key == 'qwen3-17b-q4':
        args += ['--chat-template-kwargs','{"enable_thinking":false}','--reasoning-budget','0']
    begin = time.monotonic()
    result = run('systemd-run', '--unit', bench.SERVICE, '--collect', '--property=User=opi-vlm',
                 '--property=Group=opi-vlm', '--property=CPUAffinity=6 7', '--property=MemoryMax=' + ('5G' if key == 'qwen25vl-3b-q4' else '4G'),
                 '--property=WorkingDirectory=/opt/opi-vlm', *args)
    with bench.Monitor() as monitor:
        bench.ready(timeout=180, port=PORT)
    return {'ready_seconds': time.monotonic()-begin, 'resources': monitor.summary(),
            'samples': monitor.samples, 'command': args, 'systemd': result.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['smolvlm-256m', 'smolvlm-500m', 'internvl3-1b'])
    parser.add_argument('--output', type=Path, default=ROOT/'model-comparison.json')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'vlm-models.json').read_text())
    prompt = (ROOT/'app/system_prompt.txt').read_text().strip()
    photo_path = ROOT/'opi-vlm-dog-ui.jpg'
    photo = 'data:image/jpeg;base64,' + base64.b64encode(photo_path.read_bytes()).decode()
    report = {'date': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'system_prompt': prompt,
              'input_sha256': hashlib.sha256(photo_path.read_bytes()).hexdigest(), 'models': [],
              'notes': ['Identical Q8 language/F16 vision, CPU6/7, 2 threads, 4096 context, 128 max output, seed42.',
                        'Same photo and prompt across models; image tiling and native token count can differ.',
                        'One photo qualitative test, not dataset accuracy. No electrical power meter.',
                        'TTFT includes preprocessing/vision/prefill. Warm follow-up reuses context.',
                        'Initial/cached-restart file cache is not forcibly evicted; only weight_uncached run verifies zero file residency.']}
    if args.output.exists():
        raise RuntimeError('Output already exists; use a new filename to preserve results')

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    run('systemctl', 'stop', 'opi-vlm-model.service')
    try:
        for key in args.models:
            model = manifest[key]
            directory = ROOT/'models' if key == 'smolvlm-256m' else ROOT/'models'/key
            files = [directory/item[0] for item in model['files']]
            entry = {'id': key, 'manifest': model, 'runs': [], 'tests': []}
            report['models'].append(entry)
            try:
                for file, (_, size, expected) in zip(files, model['files']):
                    if file.stat().st_size != size or digest(file) != expected:
                        raise RuntimeError(f'Unverified model: {file}')
                for mode in ['initial', 'cached_restart', 'weight_uncached']:
                    stop()
                    cool = cooling()
                    cache = None
                    if mode == 'weight_uncached':
                        for file in files:
                            with file.open('rb') as handle:
                                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                        cache = json.loads(bench.command('fincore', '--json', '--bytes', '-o', 'FILE,SIZE,RES', *map(str, files)))
                    boot = start(key, files)
                    boot.update(mode=mode, cooldown_seconds=cool, file_cache=cache)
                    entry['runs'].append(boot)
                    save()
                    print(json.dumps({'model': key, 'startup': mode, 'seconds': boot['ready_seconds']}, ensure_ascii=False), flush=True)
                    if mode == 'initial':
                        with bench.Monitor() as idle:
                            time.sleep(5)
                        entry['idle'] = idle.summary()
                        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                        with opener.open(f'http://127.0.0.1:{PORT}/props') as response:
                            entry['props'] = json.load(response)

                    def ask(label, messages):
                        result = bench.infer([{'role':'system', 'content':prompt}, *messages], port=PORT, model=key)
                        recorded = json.loads(json.dumps(messages))
                        for item in recorded:
                            if isinstance(item['content'], list):
                                for part in item['content']:
                                    if part['type'] == 'image_url':
                                        part['image_url']['url'] = 'sha256:' + report['input_sha256']
                        result.update(case=label, startup_mode=mode, messages=recorded)
                        entry['tests'].append(result)
                        save()
                        print(json.dumps({'model':key,'case':label,'mode':mode,'answer':result['answer'],
                                          'ttft':result['ttft_seconds'],'total':result['total_seconds'],
                                          'tps':result.get('timings',{}).get('predicted_per_second'),
                                          'rss':result['resources'].get('peak_rss_mib')},ensure_ascii=False),flush=True)
                        return result

                    if mode == 'initial':
                        conversations = {
                            'zh_chat': ['你好，你是谁？', '我今天吃了黄焖鸡', '心情有点不好', '我刚才说吃了什么？'],
                            'en_chat': ['Hello, good evening.', 'Who are you, baby?', 'What is your favourite food?',
                                        'I ate chicken today, a traditional Chinese food.', 'What did I say I ate?'],
                            'memory': ['我的宠物兔叫露娜。', '我的宠物叫什么？', '我的兔子是什么颜色？'],
                            'reasoning': ['小明有3个苹果，又买了2个，送给朋友1个，还剩几个？请只回答数量。']}
                        for label, questions in conversations.items():
                            messages=[]
                            for question in questions:
                                messages.append({'role':'user','content':question})
                                result=ask(label, messages.copy())
                                messages.append({'role':'assistant','content':result['answer']})
                    message={'role':'user','content':[{'type':'image_url','image_url':{'url':photo}},
                                                    {'type':'text','text':'Describe this image in two short sentences.'}]}
                    result=ask('first_image',[message])
                    ask('visual_followup',[message,{'role':'assistant','content':result['answer']},
                                           {'role':'user','content':'What color is the bicycle?'}])
                    if mode=='initial':
                        ask('zh_image',[{'role':'user','content':[message['content'][0],
                                                               {'type':'text','text':'请用中文描述图片，并说明狗在自行车前面还是后面。'}]}])
                    save()
            except Exception as error:
                entry['error']=str(error)
                entry['failure_log']=run('journalctl','-u',bench.SERVICE,'-n','60','--no-pager',check=False).stdout
                print(json.dumps({'model':key,'error':str(error)},ensure_ascii=False),flush=True)
                save()
            finally:
                stop()
    finally:
        run('systemctl','start','opi-vlm-model.service')
        bench.ready(port=8081)
        report['production_restored']=True
        save()
    print('MODEL_COMPARISON_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
