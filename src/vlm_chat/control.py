#!/usr/bin/env python3
"""Loopback-only, fixed-operation model service controller. Never executes client commands."""
import hashlib
import http.client
import json
import os
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path('/opt/opi-vlm')
MANIFEST = json.loads((ROOT/'vlm-models.json').read_text())
CONFIG = Path('/etc/systemd/system/opi-vlm-model.service.d/model.conf')
STATE = Path('/var/lib/opi-vlm-control/settings.json')
LOCK = threading.Lock()
CURRENT = {'model_id': 'smolvlm-256m', 'context': 4096}
LOADING = False
ERROR = None
METRICS = {}
CONTEXTS = [1024, 2048, 4096, 8192]
MAINTENANCE = Path('/var/lib/opi-vlm-control/benchmark.lock')


def backend(path):
    conn = http.client.HTTPConnection('127.0.0.1', 8081, timeout=2)
    try:
        conn.request('GET', path)
        response = conn.getresponse()
        data = json.loads(response.read())
        if response.status != 200:
            raise OSError('Model not ready')
        return data
    finally:
        conn.close()


def files(key):
    folder = ROOT/'models' if key == 'smolvlm-256m' else ROOT/'models'/key
    return [folder / entry[0] for entry in MANIFEST[key]['files']]


def available(key):
    return all(p.exists() and p.stat().st_size == item[1]
               for p, item in zip(files(key), MANIFEST[key]['files']))


def status():
    result = {**CURRENT, 'model': MANIFEST[CURRENT['model_id']]['name'],
              'maintenance': MAINTENANCE.exists(),
              'metrics': METRICS,
              'loading': LOADING, 'error': ERROR, 'backend': 'CPU', 'contexts': CONTEXTS,
              'vision':MANIFEST[CURRENT['model_id']].get('vision',True),
              'models': [{'id': k, 'name': v['name'], 'vision':v.get('vision',True), 'available': available(k)} for k,v in MANIFEST.items()]}
    try:
        result['ready'] = not LOADING and backend('/health').get('status') == 'ok'
    except OSError:
        result['ready'] = False
    return result


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(text)
    temporary.chmod(0o644)
    temporary.replace(path)


def monitor():
    global METRICS
    previous = None
    while True:
        try:
            now = time.monotonic()
            memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            cpu = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
            disk = shutil.disk_usage('/')
            metrics = {'ram_total_mib':int(memory['MemTotal'].split()[0])/1024,
                       'ram_available_mib':int(memory['MemAvailable'].split()[0])/1024,
                       'swap_used_mib':(int(memory['SwapTotal'].split()[0])-int(memory['SwapFree'].split()[0]))/1024,
                       'disk_total_gib':disk.total/1024**3,'disk_available_gib':disk.free/1024**3,
                       'temperature_c':int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000,
                       'load_average':list(os.getloadavg())}
            pid = int(subprocess.check_output(['systemctl','show','opi-vlm-model','-p','MainPID','--value'],text=True,timeout=3))
            ticks = 0
            if pid:
                status = dict(line.split(':',1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
                metrics['model_rss_mib'] = int(status['VmRSS'].split()[0])/1024
                fields = Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
                ticks = int(fields[11])+int(fields[12])
            if previous:
                old_time,old_cpu,old_pid,old_ticks = previous
                total = sum(cpu)-sum(old_cpu)
                if total:
                    metrics['system_cpu_percent'] = 100*(1-(cpu[3]+cpu[4]-old_cpu[3]-old_cpu[4])/total)
                if pid and pid == old_pid:
                    metrics['model_cpu_percent'] = 100*(ticks-old_ticks)/os.sysconf('SC_CLK_TCK')/(now-old_time)
            previous = now,cpu,pid,ticks
            METRICS = metrics
        except (OSError,KeyError,ValueError,subprocess.SubprocessError):
            pass
        time.sleep(2)


def restart():
    subprocess.run(['systemctl', 'daemon-reload'], check=True, timeout=30)
    subprocess.run(['systemctl', 'restart', 'opi-vlm-model'], check=True, timeout=90)


def ready(key, context):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            props = backend('/props')
            if props['model_alias'] == key and props['default_generation_settings']['n_ctx'] == context:
                return
        except (OSError, KeyError, ValueError):
            pass
        time.sleep(.25)
    raise RuntimeError('Model startup timed out')


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != '/state':
            return self.reply(404, {})
        self.reply(200, status())

    def do_POST(self):
        global CURRENT, LOADING, ERROR
        if self.path != '/settings' or self.headers.get('X-OPI-Control') != 'local-gateway':
            return self.reply(403, {'error': 'Forbidden'})
        if MAINTENANCE.exists() and self.headers.get('X-OPI-Benchmark') != 'local-benchmark':
            return self.reply(409, {'error': '性能评测进行中，请等待完成'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size < 1024:
                raise ValueError('Invalid body')
            self.connection.settimeout(5)
            data = json.loads(self.rfile.read(size))
            key, context = data['model_id'], data['context']
            if key not in MANIFEST or type(context) is not int or context not in CONTEXTS or not available(key):
                raise ValueError('模型未下载完成或窗口设置不支持')
        except (ValueError, KeyError, TypeError, OSError) as error:
            return self.reply(400, {'error': str(error)})
        if not LOCK.acquire(blocking=False):
            return self.reply(409, {'error': '模型正在切换，请等待'})
        old = CURRENT.copy()
        original = CONFIG.read_text() if CONFIG.exists() else None
        try:
            LOADING, ERROR = True, None
            CURRENT = {'model_id': key, 'context': context}
            for path, item in zip(files(key), MANIFEST[key]['files']):
                checksum = hashlib.sha256()
                with path.open('rb') as source:
                    for block in iter(lambda: source.read(1024*1024), b''):
                        checksum.update(block)
                if checksum.hexdigest() != item[2]:
                    raise RuntimeError('模型文件校验失败')
            selected = files(key)
            command = (f'/opt/opi-vlm/bin/llama-server --model {selected[0]} '
                       f'--alias {key} --host 127.0.0.1 --port 8081 --ctx-size {context} '
                       '--parallel 1 --threads 2 --threads-batch 2 --n-gpu-layers 0 '
                       '--no-mmproj-offload --cache-ram 128 --no-context-shift')
            if len(selected)>1:
                command += f' --mmproj {selected[1]}'
            if key == 'qwen3-17b-q4':
                command += ''' --chat-template-kwargs '{"enable_thinking":false}' --reasoning-budget 0'''
            memory = '5G' if key == 'qwen25vl-3b-q4' else '4G'
            atomic(CONFIG, '[Service]\nMemoryMax=' + memory + '\nExecStart=\nExecStart=' + command + '\n')
            restart()
            ready(key, context)
            atomic(STATE, json.dumps(CURRENT))
        except Exception as error:
            ERROR = '切换失败，恢复上一配置：' + str(error)
            CURRENT = old
            try:
                if original is None:
                    CONFIG.unlink(missing_ok=True)
                else:
                    atomic(CONFIG, original)
                restart()
                ready(old['model_id'], old['context'])
            except Exception as rollback:
                ERROR += '; 恢复失败：' + str(rollback)
        finally:
            LOADING = False
            LOCK.release()
        self.reply(500 if ERROR else 200, status())


if __name__ == '__main__':
    try:
        props = backend('/props')
        CURRENT = {'model_id': props['model_alias'], 'context': props['default_generation_settings']['n_ctx']}
    except (OSError, KeyError):
        if STATE.exists():
            CURRENT = json.loads(STATE.read_text())
    threading.Thread(target=monitor,daemon=True).start()
    ThreadingHTTPServer(('127.0.0.1', 8083), Handler).serve_forever()
