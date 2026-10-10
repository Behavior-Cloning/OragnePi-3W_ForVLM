#!/usr/bin/env python3
"""On-board live benchmark. Reports observed values, not estimated power or accuracy."""
import argparse
import base64
import http.client
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

SERVICE = 'opi-vlm-model.service'


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def pid():
    return int(command('systemctl', 'show', SERVICE, '--property=MainPID', '--value'))


def snapshot(process):
    result = {'time': time.monotonic()}
    try:
        status = dict(line.split(':', 1) for line in Path(f'/proc/{process}/status').read_text().splitlines() if ':' in line)
        result['rss_mib'] = int(status['VmRSS'].split()[0]) / 1024
        result['rss_highwater_mib'] = int(status['VmHWM'].split()[0]) / 1024
        fields = Path(f'/proc/{process}/stat').read_text().split(') ', 1)[1].split()
        result['cpu_seconds'] = (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
        result['major_faults'] = int(fields[9])
        io = dict(line.split(':', 1) for line in Path(f'/proc/{process}/io').read_text().splitlines())
        result['read_bytes'] = int(io['read_bytes'])
        result['write_bytes'] = int(io['write_bytes'])
    except (FileNotFoundError, KeyError, ProcessLookupError):
        pass
    for key, path in [('temperature_c', '/sys/class/thermal/thermal_zone0/temp'),
                      ('cgroup_mib', f'/sys/fs/cgroup/system.slice/{SERVICE}/memory.current')]:
        try:
            result[key] = int(Path(path).read_text()) / (1000 if key == 'temperature_c' else 1024 ** 2)
        except FileNotFoundError:
            pass
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    result['system_available_mib'] = int(memory['MemAvailable'].split()[0]) / 1024
    result['swap_used_mib'] = (int(memory['SwapTotal'].split()[0]) - int(memory['SwapFree'].split()[0])) / 1024
    result['cpu_freq_khz'] = {p.parent.name: int(p.read_text()) for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq')}
    result['load_average'] = list(os.getloadavg())
    cpu = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    result['system_cpu_ticks'] = sum(cpu)
    result['system_idle_ticks'] = cpu[3]
    result['system_iowait_ticks'] = cpu[4]
    result['system_total_mib'] = int(memory['MemTotal'].split()[0]) / 1024
    result['system_used_mib'] = result['system_total_mib'] - result['system_available_mib']
    return result


class Monitor:
    def __init__(self):
        self.samples = []
        self.stopped = threading.Event()
        self.process = pid()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stopped.is_set():
            self.samples.append(snapshot(self.process))
            self.stopped.wait(.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join()
        self.samples.append(snapshot(self.process))

    def summary(self):
        result = {}
        for key in ['rss_mib', 'cgroup_mib', 'temperature_c', 'swap_used_mib', 'system_used_mib']:
            values = [s[key] for s in self.samples if key in s]
            if values:
                result[f'peak_{key}'] = max(values)
        valid = [s for s in self.samples if 'cpu_seconds' in s]
        if len(self.samples) >= 2:
            first, last = self.samples[0], self.samples[-1]
            ticks = last['system_cpu_ticks'] - first['system_cpu_ticks']
            if ticks > 0:
                result['system_cpu_busy_percent'] = 100 * (1 - (last['system_idle_ticks']-first['system_idle_ticks'] + last['system_iowait_ticks']-first['system_iowait_ticks']) / ticks)
                result['system_iowait_percent'] = 100 * (last['system_iowait_ticks']-first['system_iowait_ticks']) / ticks
            result['min_available_mib'] = min(s['system_available_mib'] for s in self.samples)
        if len(valid) >= 2:
            first, last = valid[0], valid[-1]
            result['cpu_seconds'] = last['cpu_seconds'] - first['cpu_seconds']
            result['cpu_percent_one_core_100'] = 100 * result['cpu_seconds'] / (last['time'] - first['time'])
            for key in ['read_bytes', 'write_bytes', 'major_faults']:
                if key in last and key in first:
                    result[key] = last[key] - first[key]
        return result


def ready(timeout=120, port=8081):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
        try:
            connection.request('GET', '/health')
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return time.monotonic() - start
        except OSError:
            pass
        finally:
            connection.close()
        time.sleep(.05)
    raise TimeoutError('Model did not become ready')


def infer(messages, max_tokens=128, port=8081, model='smolvlm-256m', timeout=600, sampling=None):
    payload = {'model': model, 'messages': messages, 'stream': True,
               'max_tokens': max_tokens, 'temperature': 0.2, 'seed': 42,
               'repeat_penalty': 1.1, 'stream_options': {'include_usage': True}, 'timings_per_token': False}
    payload.update(sampling or {})
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    result = {'answer': '', 'reasoning': '', 'ttft_seconds': None}
    cursor_output = command('journalctl', '-u', SERVICE, '-n', '0', '--show-cursor', '--no-pager')
    cursor = cursor_output.split('-- cursor: ')[-1].strip()
    with Monitor() as monitor:
        start = time.monotonic()
        try:
            connection.request('POST', '/api/chat' if port == 8080 else '/v1/chat/completions',
                               body=json.dumps(payload), headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f'{response.status}: {response.read().decode()}')
            done = False
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                item = json.loads(data)
                if item.get('error'):
                    raise RuntimeError(str(item['error']))
                choices = item.get('choices', [])
                text = choices[0].get('delta', {}).get('content', '') if choices else ''
                delta = choices[0].get('delta', {}) if choices else {}
                result['reasoning'] += delta.get('reasoning_content') or delta.get('reasoning') or ''
                if text:
                    if result['ttft_seconds'] is None:
                        result['ttft_seconds'] = time.monotonic() - start
                    result['answer'] += text
                if item.get('timings'):
                    result['timings'] = item['timings']
                if item.get('usage'):
                    result['usage'] = item['usage']
                if choices and choices[0].get('finish_reason'):
                    result['finish_reason'] = choices[0]['finish_reason']
            if not done:
                raise RuntimeError('Incomplete SSE stream')
        except Exception as error:
            result['error'] = str(error)
        finally:
            connection.close()
    result['total_seconds'] = time.monotonic() - start
    result['resources'] = monitor.summary()
    result['samples'] = monitor.samples
    log = command('journalctl', '-u', SERVICE, '--after-cursor', cursor, '--no-pager', '-o', 'cat')
    result['server_log'] = log
    result['vision_encode_ms_logged'] = [int(value) for value in re.findall(r'image slice encoded in (\d+) ms', log)]
    result['image_prefill_ms_logged'] = [int(value) for value in re.findall(r'image decoded \(batch \d+/\d+\) in (\d+) ms', log)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True, help='JPEG resized to the same 768px limit as the UI')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--cold-restarts', type=int, default=3)
    parser.add_argument('--uncached-restarts', type=int, default=3,
                        help='Evict only these two model files while service is stopped; verify with fincore')
    args = parser.parse_args()
    image = 'data:image/jpeg;base64,' + base64.b64encode(args.image.read_bytes()).decode()
    photo = {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image}},
                                       {'type': 'text', 'text': 'Describe this image in two short sentences.'}]}
    report = {'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'uname': command('uname', '-a'),
              'model_service': command('systemctl', 'cat', SERVICE), 'repetitions': args.repetitions,
              'max_tokens': 128, 'tests': [], 'cold_restarts': [], 'uncached_restarts': [], 'quality_checks': [],
              'notes': ['CPU percentage uses 100% per core; 200% means two full cores.',
                        'cold_restarts retain Linux page cache. uncached_restarts evict only model files and verify residency. Neither measures power-on boot.',
                        'RSS includes mapped weights; cgroup includes cache. Do not add these two metrics.',
                        'TTFT includes HTTP, image encoding, prompt prefill and first decode token.',
                        'No electrical energy/power measurement. Small sample; no accuracy benchmark score.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    ready()
    with Monitor() as idle:
        time.sleep(10)
    report['idle'] = idle.summary()
    report['idle_sample'] = idle.samples[-1]
    persist()
    for index in range(args.repetitions):
        cases = [('text', [{'role': 'user', 'content': 'Explain what a bicycle is in two short sentences.'}]),
                 ('image', [photo])]
        for name, messages in cases:
            result = infer(messages)
            result.update(case=name, repetition=index + 1)
            report['tests'].append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ['samples', 'server_log']}, ensure_ascii=False), flush=True)
            persist()
            if name == 'image':
                followup = infer([photo, {'role': 'assistant', 'content': result['answer']},
                                  {'role': 'user', 'content': 'What color is the animal in this image?'}])
                followup.update(case='visual_followup', repetition=index + 1)
                report['tests'].append(followup)
                print(json.dumps({k: v for k, v in followup.items() if k not in ['samples', 'server_log']}, ensure_ascii=False), flush=True)
                persist()
    restart_modes = [('cold_restarts', args.cold_restarts), ('uncached_restarts', args.uncached_restarts)]
    for mode, index in [(mode, index) for mode, count in restart_modes for index in range(count)]:
        subprocess.run(['systemctl', 'stop', SERVICE], check=True)
        unloaded = snapshot(0)
        cache = None
        if mode == 'uncached_restarts':
            weights = sorted(Path('/opt/opi-vlm/models').glob('*.gguf'))
            for weight in weights:
                with weight.open('rb') as source:
                    os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            cache = json.loads(command('fincore', '--json', '--bytes', '--output', 'FILE,SIZE,RES', *map(str, weights)))
        start = time.monotonic()
        subprocess.run(['systemctl', 'start', SERVICE], check=True)
        with Monitor() as startup_monitor:
            ready()
        boot = {'repetition': index + 1, 'ready_seconds': time.monotonic() - start}
        boot['unloaded_system'] = unloaded
        boot['startup_resources'] = startup_monitor.summary()
        boot['startup_samples'] = startup_monitor.samples
        boot['file_cache_before_start'] = cache
        boot['first_image'] = infer([photo])
        report[mode].append(boot)
        print(json.dumps({'mode': mode, 'restart': index + 1, 'ready_seconds': boot['ready_seconds'],
                          'first_image_ttft_seconds': boot['first_image']['ttft_seconds']}, ensure_ascii=False), flush=True)
        persist()
    report['summary'] = {}
    for name in ['text', 'image', 'visual_followup']:
        tests = [r for r in report['tests'] if r['case'] == name]
        report['summary'][name] = {key: {'median': statistics.median(r[key] for r in tests),
                                        'min': min(r[key] for r in tests), 'max': max(r[key] for r in tests)}
                                   for key in ['ttft_seconds', 'total_seconds']}
    for question in ['What animal is in this image?', 'What color is the bicycle?',
                     'What vehicle is in the background?', 'Is the dog in front of or behind the bicycle?',
                     '请用中文简短描述这张图片。']:
        message = {'role': 'user', 'content': [photo['content'][0], {'type': 'text', 'text': question}]}
        result = infer([message])
        result['question'] = question
        report['quality_checks'].append(result)
        print(json.dumps({'quality_question': question, 'answer': result['answer']}, ensure_ascii=False), flush=True)
        persist()
    persist()
    print('BENCHMARK_COMPLETE ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
