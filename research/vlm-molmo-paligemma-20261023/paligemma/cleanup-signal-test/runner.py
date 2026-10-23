import pathlib, subprocess, time, threading, queue, json, os, signal, shutil
P = pathlib.Path(__file__).parent
out = (P / 'pali-board-results.jsonl').open('w')
samples = []
stop = threading.Event()
q = queue.Queue()
guard = []
stdout = []

def emit(d):
    out.write(json.dumps(d, ensure_ascii=False) + '\n')
    out.flush()

def monitor(proc):
    while not stop.wait(0.5):
        try:
            st = dict((x.split(':', 1) for x in pathlib.Path('/proc/' + str(proc.pid) + '/status').read_text().splitlines() if ':' in x))
            sm = dict((x.split(':', 1) for x in pathlib.Path('/proc/' + str(proc.pid) + '/smaps_rollup').read_text().splitlines() if ':' in x))
            mem = dict((x.split(':', 1) for x in pathlib.Path('/proc/meminfo').read_text().splitlines()))
            temps = {x.parent.name: float(x.read_text()) / 1000 for x in pathlib.Path('/sys/class/thermal').glob('thermal_zone*/temp')}
            stat = pathlib.Path('/proc/' + str(proc.pid) + '/stat').read_text().split()
            freq = {x.parent.parent.name: int(x.read_text()) for x in pathlib.Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_cur_freq')}
            row = dict(t=time.monotonic(), rss_kb=int(st['VmRSS'].split()[0]), pss_kb=int(sm['Pss'].split()[0]), swap_kb=int(sm['Swap'].split()[0]), available_kb=int(mem['MemAvailable'].split()[0]), temps=temps, cpu_ticks=int(stat[13]) + int(stat[14]), freq_khz=freq, loadavg=pathlib.Path('/proc/loadavg').read_text().strip())
            row['thread_affinities'] = {q.name: dict((x.split(':', 1) for x in (q / 'status').read_text().splitlines() if ':' in x)).get('Cpus_allowed_list', '').strip() for q in pathlib.Path('/proc/' + str(proc.pid) + '/task').iterdir()}
            samples.append(row)
            if max(temps.values()) >= 78 or row['available_kb'] < 256 * 1024:
                guard.append(dict(reason='temperature>=78C or available<256MiB', sample=row))
                os.killpg(proc.pid, signal.SIGTERM)
                return
        except (OSError, KeyError, ValueError, ProcessLookupError):
            pass

def readerr(proc):
    with (P / 'pali-board-stderr.log').open('w') as f:
        for line in proc.stderr:
            f.write(line)
            f.flush()
            q.put((time.monotonic(), line))
    q.put((time.monotonic(), 'PROCESS_EOF'))

def readout(proc):
    with (P / 'pali-board-stdout.log').open('w') as f:
        while True:
            c = proc.stdout.read(1)
            if not c:
                break
            stdout.append(c)
            f.write(c)
            f.flush()

def until(marker, timeout):
    deadline = time.monotonic() + timeout
    rows = []
    while time.monotonic() < deadline:
        try:
            (t, line) = q.get(timeout=0.5)
        except queue.Empty:
            if guard:
                raise RuntimeError('resource_guard ' + str(guard[-1]))
            if proc.poll() is not None:
                raise RuntimeError('process exit ' + str(proc.returncode))
            continue
        rows.append(dict(observed_t=t, line=line.rstrip()))
        if marker in line:
            return rows
        if line == 'PROCESS_EOF':
            raise RuntimeError('unexpected EOF ' + str(proc.poll()))
    raise TimeoutError(marker)
cmd = ['python3', '-c', 'import time; time.sleep(120)']

class TaskInterrupted(RuntimeError):
    pass

def interrupted(signum, frame):
    raise TaskInterrupted('signal ' + str(signum))
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
proc = None
threads = []
try:
    assert shutil.disk_usage(P).free >= 1024 ** 3
    start = time.monotonic()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
    threads = [threading.Thread(target=readerr, args=(proc,), daemon=True), threading.Thread(target=readout, args=(proc,), daemon=True), threading.Thread(target=monitor, args=(proc,), daemon=True)]
    for t in threads:
        t.start()
    ready = until('OPI_MODEL_READY', 120)
    affinity = {t.name: dict((x.split(':', 1) for x in (t / 'status').read_text().splitlines() if ':' in x)).get('Cpus_allowed_list', '').strip() for t in pathlib.Path('/proc/' + str(proc.pid) + '/task').iterdir()}
    emit(dict(affinity=affinity, event='ready', process_ready_s=time.monotonic() - start, markers=ready, command=cmd, image_recomputed_each_request=True, note='model resident; OS page cache not flushed; fixed system KV not implemented'))
    cases = [('building_first', 'What type of building is it?'), ('building_hot', 'What type of building is it?'), ('color_hot', 'What color is the church?'), ('caption_hot', 'Describe this image in one sentence.')]
    for (cid, prompt) in cases:
        offset = len(stdout)
        t0 = time.monotonic()
        proc.stdin.write(prompt + '\n')
        proc.stdin.flush()
        rows = until('OPI_REQUEST_END', 240)
        emit(dict(event='measure', case=cid, prompt=prompt, request_sent=t0, total_observed_s=time.monotonic() - t0, markers=rows, stdout=''.join(stdout[offset:]), resident_hot=cid != 'building_first', image_recomputed=True))
    proc.stdin.write('%q\n')
    proc.stdin.flush()
    proc.wait(timeout=20)
    emit(dict(event='complete', exit=proc.returncode))
except Exception as e:
    emit(dict(event='error', error=repr(e)))
    raise
finally:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    stop.set()
    for t in threads:
        t.join(timeout=2)
    (P / 'pali-board-telemetry.json').write_text(json.dumps(dict(samples=samples, guard=guard, clock_ticks=os.sysconf('SC_CLK_TCK'), exit=None if proc is None else proc.returncode)))
    out.close()
