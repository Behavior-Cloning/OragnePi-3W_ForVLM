"""Direct Linux counters for isolated CPU/NPU benchmarks; no web metric cache."""
import os
from pathlib import Path
import re
import time


def counter_delta(before, after):
    """A reset is missing data, never a negative rate or wraparound spike."""
    return {k: v-before[k] for k, v in after.items()
            if k in before and v >= before[k]}


def cpu_usage(before, after):
    result = {}
    for name, ticks in after.items():
        if name not in before or any(b < a for a, b in zip(before[name], ticks)):
            continue
        delta = [b-a for a, b in zip(before[name], ticks)]
        total = sum(delta)  # user/nice already include guest; omit guest columns.
        if total:
            result[name] = {'busy_percent': 100*(total-delta[3]-delta[4])/total,
                            'iowait_percent': 100*delta[4]/total,
                            'total_ticks': total}
    return result


def process_snapshot(root, pid, read_pss):
    """Optional kernel accounting files must not discard available RSS/CPU."""
    fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
    status = dict(line.split(':', 1) for line in (root/'status').read_text().splitlines())
    process = {'pid': pid, 'starttime_ticks': int(fields[19]),
               'cpu_seconds': (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
               'minor_faults': int(fields[7]), 'major_faults': int(fields[9]),
               'rss_kib': int(status['VmRSS'].split()[0]),
               'peak_rss_kib': int(status['VmHWM'].split()[0])}
    errors = {}
    try:
        process['io'] = {k: int(v) for k, v in
                         (line.split(':', 1) for line in (root/'io').read_text().splitlines())}
    except OSError as exc:
        errors['process_io'] = str(exc)
    if read_pss:
        try:
            rollup = root/'smaps_rollup'
            path = rollup if rollup.exists() else root/'smaps'
            values = [int(line.split()[1]) for line in path.read_text().splitlines() if line.startswith('Pss:')]
            if not values:
                raise ValueError('Pss field unavailable')
            process.update(pss_kib=sum(values), pss_source=path.name)
        except (OSError, ValueError) as exc:
            errors['pss'] = str(exc)
    return process, errors


def driver_bytes(profile):
    # Columns 3/4 are decimal allocation counts. Require the first two
    # hexadecimal byte columns, including their 0x prefixes.
    matches=re.findall(r'^\s*\|\s*(0x[0-9a-fA-F]+)\s*\|\s*(0x[0-9a-fA-F]+)\s*\|',profile,re.M)
    if len(matches)!=1:
        raise ValueError('unrecognized mem_profile byte columns')
    return int(matches[0][0],16)


class DirectSampler:
    def __init__(self, proc='/proc', sys='/sys'):
        self.proc, self.sys = Path(proc), Path(sys)
        self.previous = None
        self.last_pss = -float('inf')

    def snapshot(self, pid=None):
        started = time.monotonic()
        sample = {'source': 'proc-sys-direct-v3', 'monotonic_seconds': started,
                  'page_bytes': os.sysconf('SC_PAGE_SIZE'), 'errors': {}}
        cpu = {}
        for line in (self.proc/'stat').read_text().splitlines():
            fields = line.split()
            if re.fullmatch(r'cpu\d*', fields[0]):
                cpu[fields[0]] = list(map(int, fields[1:9]))
        sample['cpu_ticks'] = cpu
        memory = {}
        for line in (self.proc/'meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree', 'Cached', 'SReclaimable', 'Dirty', 'Writeback'):
                memory[key] = int(value.split()[0])
        sample['memory_kib'] = memory
        sample['vm_counters'] = {k: int(v) for k, v in
                                (line.split() for line in (self.proc/'vmstat').read_text().splitlines())
                                if k in ('pswpin', 'pswpout', 'pgmajfault', 'pgfault')}
        disk = {}
        for line in (self.proc/'diskstats').read_text().splitlines():
            f = line.split()
            # Whole physical disks only, never sum partitions with their parent.
            if re.fullmatch(r'mmcblk\d+|sd[a-z]+|nvme\d+n\d+', f[2]):
                disk[f[2]] = {'read_bytes': int(f[5])*512, 'write_bytes': int(f[9])*512,
                              'read_ios': int(f[3]), 'write_ios': int(f[7]),
                              'io_milliseconds': int(f[12])}
        sample['disk_counters'] = disk
        sample['frequencies_khz'] = {p.name: int((p/'scaling_cur_freq').read_text())
                                    for p in (self.sys/'devices/system/cpu/cpufreq').glob('policy*')}
        for name, path in [('temperature_mC', 'class/thermal/thermal_zone0/temp'),
                           ('npu_frequency_hz', 'class/devfreq/3600000.npu/cur_freq')]:
            try: sample[name] = int((self.sys/path).read_text())
            except OSError as exc: sample['errors'][name] = str(exc)
        try:
            profile = (self.sys/'kernel/debug/viplite/mem_profile').read_text()
            sample['npu_driver_bytes'] = driver_bytes(profile)
        except (OSError,ValueError) as exc: sample['errors']['npu_driver_bytes'] = str(exc)
        if pid is not None:
            root = self.proc/str(pid)
            try:
                read_pss = started-self.last_pss >= 5
                process, errors = process_snapshot(root, pid, read_pss)
                sample['errors'].update(errors)
                if read_pss:
                    self.last_pss = started
                sample['process'] = process
            except (OSError, KeyError) as exc:
                sample['errors']['process'] = str(exc)  # Process may have exited.
        if self.previous is not None:
            prev = self.previous
            dt = started-prev['monotonic_seconds']
            sample['interval_start_monotonic_seconds'] = prev['monotonic_seconds']
            sample['interval_seconds'] = dt
            sample['cpu_usage'] = cpu_usage(prev['cpu_ticks'], cpu)
            sample['vm_delta'] = counter_delta(prev['vm_counters'], sample['vm_counters'])
            sample['disk_delta'] = {name: counter_delta(prev['disk_counters'][name], values)
                                    for name, values in disk.items() if name in prev['disk_counters']}
        sample['sampling_seconds'] = time.monotonic()-started
        self.previous = sample
        return sample


def response_summary(samples, start, end):
    """Only intervals wholly inside the response; exclude heater overlap."""
    inside = [s for s in samples if s.get('interval_start_monotonic_seconds', -1) >= start
              and s['monotonic_seconds']+s['sampling_seconds'] <= end]
    cpu = {}
    for name in {k for s in inside for k in s.get('cpu_usage', {})}:
        values = [s['cpu_usage'][name] for s in inside if name in s.get('cpu_usage', {})]
        ticks = sum(v['total_ticks'] for v in values)
        cpu[name] = {key: sum(v[key]*v['total_ticks'] for v in values)/ticks
                     for key in ('busy_percent', 'iowait_percent')}
    return {'scope': 'direct counter intervals wholly within response; boundary gaps excluded',
            'covered_seconds': sum(s['interval_seconds'] for s in inside),
            'response_seconds': end-start, 'intervals': len(inside), 'cpu': cpu,
            'swap_in_bytes': sum(s['vm_delta'].get('pswpin', 0)*s['page_bytes'] for s in inside),
            'swap_out_bytes': sum(s['vm_delta'].get('pswpout', 0)*s['page_bytes'] for s in inside),
            'disk_io_bytes': {name: {key: sum(s.get('disk_delta', {}).get(name, {}).get(key, 0) for s in inside)
                                     for key in ('read_bytes', 'write_bytes')}
                              for name in {k for s in inside for k in s.get('disk_delta', {})}}}
