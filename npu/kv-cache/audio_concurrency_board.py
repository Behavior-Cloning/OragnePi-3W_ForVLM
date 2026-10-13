"""Board-only full-depth CPU/NPU on/off comparison with resident real ASR.

Fixed 128 predictions, medium start, 512 existing + 64 new tokens. ASR runs
back-to-back copies of an 11-second recording; this is sustained compute load,
not real-time microphone scheduling or an audio-quality benchmark.
"""
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

from resident_worker import ResidentWorker
from resource_sampler import DirectSampler, response_summary, process_snapshot
from thermal_condition import condition, temperature


class Audio:
    def __init__(self, config, root):
        self.log = (root / 'asr.log').open('xb')
        self.process = subprocess.Popen(['taskset', '-c', config.get('asr_affinity','0-7'), config['asr_binary'],
            config['asr_model'], config['asr_wav'], '2'], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = bytearray()
        self.records = []
        self.thread = None
        self.stop = threading.Event()
        self.active = threading.Event()
        self.error = None
        try:
            self.ready = self.event()
            assert self.ready == {'event': 'ready', 'audio_seconds': 11}
        except BaseException:
            self.process.kill(); self.process.wait()
            self.selector.close(); self.log.close()
            raise

    def event(self):
        deadline = time.monotonic() + 180
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError('ASR event')
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError('ASR exited')
            self.buffer.extend(data)
            assert len(self.buffer) <= 2**20
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        return json.loads(line)

    def infer(self):
        start = time.monotonic()
        self.process.stdin.write(b'infer\n'); self.process.stdin.flush()
        self.active.set()
        response = self.event()
        assert response['event'] == 'done' and 'country' in response['text'].lower()
        row = {**response, 'start_monotonic': start, 'end_monotonic': time.monotonic()}
        self.records.append(row)
        return row

    def begin(self):
        assert self.thread is None
        self.stop.clear(); self.active.clear(); self.error = None
        def loop():
            try:
                while not self.stop.is_set():
                    self.infer()
            except BaseException as exc:
                self.error = str(exc)
                self.active.set()
        self.thread = threading.Thread(target=loop)
        self.thread.start()
        assert self.active.wait(5) and self.error is None

    def end(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=190)
            assert not self.thread.is_alive(), 'ASR shutdown timeout'
            self.thread = None
        assert self.error is None, self.error

    def close(self):
        try:
            self.end()
            self.process.stdin.write(b'close\n'); self.process.stdin.flush()
            assert self.event()['event'] == 'closed'
            assert self.process.wait(timeout=30) == 0
        finally:
            if self.process.poll() is None:
                self.process.kill(); self.process.wait()
            self.selector.close(); self.process.stdin.close(); self.log.close()


def main():
    c = json.loads(Path(sys.argv[1]).read_text())
    root = Path(c['remote'])
    report = {'status': 'running', 'scope': __doc__, 'config': c, 'runs': [],
              'asr_scope': 'resident whisper.cpp tiny Q5_1; 2 CPU threads on '+c.get('asr_affinity','0-7')+'; back-to-back 11 s WAV',
              'production_cpu_model_stays_resident': True}
    # Waiting on the host lock can span edits. These are the actual files
    # deployed and built for this run, not the host's earlier queue snapshot.
    deployed_files = ('audio_concurrency_board.py', 'resident_worker.py', 'resident_session.c',
                      'full_model.c', 'board_probe.c', 'batch_prefill.h', 'greedy_i16.h',
                      'resource_sampler.py', 'thermal_condition.py', 'thermal_heater.c',
                      'asr-worker/asr_worker.cpp', 'asr-worker/CMakeLists.txt', 'resident')
    report['deployed_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                 for name in deployed_files}
    report['asr_binary_sha256'] = hashlib.sha256(Path(c['asr_binary']).read_bytes()).hexdigest()
    if c.get('expected_asr_binary_sha256'):
        assert report['asr_binary_sha256']==c['expected_asr_binary_sha256'], 'reused ASR binary changed'
    def save():
        tmp = root / 'result.tmp'
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(root / 'result.json')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def api(path, body=None, port=18085):
        req = urllib.request.Request(f'http://127.0.0.1:{port}' + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=180) as response:
            return json.load(response)
    def production():
        h = api('/api/health', port=8080)
        assert h['ready'] and not any(h[k] for k in ('busy', 'loading', 'maintenance'))
    audio = None
    server = None
    worker = None
    guard_stop = threading.Event()
    guard_errors = []
    def guard():
        while not guard_stop.wait(.5):
            value = temperature()
            if value >= 80000:
                guard_errors.append({'temperature_mC': value})
                for p in (server, worker.process if worker else None, audio.process if audio else None):
                    if p and p.poll() is None:
                        p.terminate()
                return
    thread = threading.Thread(target=guard, daemon=True)
    thread.start()
    save()
    try:
        production()
        assert shutil.disk_usage(root).free >= len(c['feedback'])*c['vocab']*4+512*2**20
        audio = Audio(c, root)
        report['asr_warmup'] = audio.infer()
        save()
        prompt, prefix, feedback = c['prompt'], c['cached_prefix'], c['feedback']
        for backend in c.get('backends', ['npu', 'cpu']):
            deadline = time.monotonic()+300
            while temperature() > 55000:
                assert not guard_errors and time.monotonic()<deadline
                time.sleep(.5)
            if backend == 'npu':
                cmd = ['taskset', '-c', '6,7', str(root/'resident'), c['deployed'], str(c['layers']),
                       str(c['capacity']), str(c['vocab']), c['deployed']+'/embeddings.f32',
                       str(c['batch']), str(c['eos'])]
                (root / 'dumps').mkdir()
                worker = ResidentWorker(cmd, root, root/'npu.log', root/'dumps')
                # Full vocabulary equality before accepting a new orchestration.
                list(worker.generate(prompt[:prefix], 1, reset=True))
                diagnostic = list(worker.generate(prompt, 128, feedback, diagnostic=True))[-1]
                assert diagnostic['reused_tokens'] == prefix and diagnostic['tokens'] == c['expected_tokens']
                with (root/'dumps/2-logits.f32').open('rb') as stream:
                    digest = hashlib.sha256()
                    for block in iter(lambda:stream.read(2**20),b''):
                        digest.update(block)
                    assert digest.hexdigest() == c['expected_logits_sha256']
                report['npu_diagnostic'] = diagnostic
                pid = worker.process.pid
            else:
                cmd = list(c['cpu_command'])
                cmd[cmd.index('--port')+1] = '18085'
                with (root/'cpu.log').open('xb') as log:
                    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic()+120
                while time.monotonic()<deadline:
                    assert server.poll() is None
                    try:
                        if api('/health').get('status') == 'ok':
                            break
                    except OSError:
                        pass
                    time.sleep(.2)
                else:
                    raise TimeoutError('CPU ready')
                pid = server.pid
            for repeat in range(6):
                for enabled in ([False, True] if repeat % 2 == 0 else [True, False]):
                    assert not guard_errors
                    production()
                    deadline = time.monotonic()+300
                    while temperature()>55000:
                        assert not guard_errors and time.monotonic()<deadline
                        time.sleep(.5)
                    options = {'temperature':0,'seed':42,'ignore_eos':True,'repeat_penalty':1.,
                               'return_tokens':True,'stream':False}
                    if backend == 'npu':
                        prepared = list(worker.generate(prompt[:prefix],1,reset=True))[-1]
                        assert prepared['kv_tokens'] == prefix
                    else:
                        api('/completion', {**options,'prompt':prompt[:prefix],'n_predict':1,'cache_prompt':False})
                    thermal = condition(55000, 64999)
                    assert not guard_errors
                    asr_index = len(audio.records)
                    if enabled:
                        audio.begin()
                    sampler = DirectSampler()
                    def snapshot():
                        sample=sampler.snapshot(pid)
                        # The ASR model is resident even with compute disabled;
                        # show its own memory instead of attributing it to LLM.
                        try:
                            proc,err=process_snapshot(Path('/proc')/str(audio.process.pid),audio.process.pid,False)
                            sample['asr_process']=proc
                            sample['errors'].update({'asr_'+k:v for k,v in err.items()})
                        except (OSError,KeyError) as exc:
                            sample['errors']['asr_process']=str(exc)
                        sample['sampling_seconds']=time.monotonic()-sample['monotonic_seconds']
                        return sample
                    samples = [snapshot()]
                    errors = []
                    stop = threading.Event()
                    def observe():
                        while not stop.wait(.25):
                            try:
                                samples.append(snapshot())
                            except Exception as exc:
                                errors.append(str(exc))
                    observer = threading.Thread(target=observe)
                    observer.start()
                    start = time.monotonic()
                    start_temp = temperature()
                    try:
                        if backend == 'npu':
                            done = list(worker.generate(prompt,128,feedback))[-1]
                            assert done['tokens'] == c['expected_tokens'] and done['reused_tokens'] == prefix
                            ttft = done['ttft_seconds']; speed = done['decode_tokens_per_second']
                        else:
                            done = api('/completion', {**options,'prompt':prompt,'n_predict':128,'cache_prompt':True})
                            t = done['timings']
                            assert done['tokens'] == feedback, 'CPU continuation changed; re-align before comparison'
                            assert t['cache_n'] == prefix and t['prompt_n'] == len(prompt)-prefix and t['predicted_n'] == 128
                            ttft = t['prompt_ms']/1000; speed = t['predicted_per_second']
                        end = time.monotonic()
                        end_temp = temperature()
                    finally:
                        stop.set(); observer.join(timeout=5)
                        if enabled:
                            audio.end()
                    assert not errors and not guard_errors
                    assert 55000 <= start_temp < 65000, 'response start temperature outside medium band'
                    asr = audio.records[asr_index:]
                    overlap = sum(max(0,min(end,r['end_monotonic'])-max(start,r['start_monotonic'])) for r in asr)
                    assert not enabled or asr and overlap/(end-start) >= .95, 'ASR did not overlap response'
                    report['runs'].append({'backend':backend,'audio':enabled,'phase':'warmup' if repeat==0 else 'measure',
                        'repeat':repeat,'engine_ttft_seconds':ttft,'decode_tokens_per_second':speed,
                        'total_seconds':end-start,'response_start_mC':start_temp,'response_end_mC':end_temp,
                        'thermal_conditioning':thermal,'runtime':done,'resource_samples':samples,
                        'resource_summary':response_summary(samples,start,end),'asr':asr,
                        'asr_overlap_fraction':overlap/(end-start)})
                    save()
                    print(json.dumps({'backend':backend,'asr':enabled,'repeat':repeat,
                          'engine_ttft_seconds':ttft,'decode_tokens_per_second':speed,
                          'asr_overlap_fraction':overlap/(end-start)}),flush=True)
            if worker:
                report['npu_closed'] = worker.close(); worker = None
                assert report['npu_closed']['driver_bytes'] == 0
            if server:
                server.terminate(); server.wait(timeout=20); server = None
        report['asr_all_calls'] = audio.records
        audio.close(); audio = None
        report['medians'] = {}
        for backend in c.get('backends', ['npu', 'cpu']):
            for enabled in (False,True):
                rows = [r for r in report['runs'] if r['backend']==backend and r['audio']==enabled and r['phase']=='measure']
                assert len(rows)==5
                report['medians'][backend+('-asr' if enabled else '-alone')] = {
                    k: statistics.median(r[k] for r in rows) for k in
                    ('engine_ttft_seconds','decode_tokens_per_second','total_seconds')}
        report['status']='complete-paired-concurrency';save()
    except BaseException as exc:
        report.update(status='failed',error=str(exc),thermal_abort=guard_errors)
        save()
        raise
    finally:
        guard_stop.set();thread.join(timeout=2)
        if worker:
            worker.abort()
        if server and server.poll() is None:
            server.terminate();server.wait(timeout=20)
        if audio:
            audio.close()


if __name__ == '__main__':
    main()
