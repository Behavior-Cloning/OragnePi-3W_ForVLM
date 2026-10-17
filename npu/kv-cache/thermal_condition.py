"""Board thermal preconditioning only; all heater workers stop before timing."""
import argparse
import json
import os
from pathlib import Path
import time
import subprocess
import signal


def interrupted(signum,frame):
    raise InterruptedError(f'thermal conditioning interrupted by signal {signum}')


def temperature():
    value=int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())
    if not 0<value<110000:raise RuntimeError('invalid temperature sensor')
    return value


def condition(low,high,timeout=300):
    assert 35000<=low<high<=80000 and high-low>=5000
    started=time.monotonic();before=temperature();workers=[];heated=False;peak=before
    # Stop inside the admission band, then recheck after reaping the heater.
    # Waiting for high+5C can deadlock on a cooler board that plateaus below it.
    heater_stop=high-1000
    original_affinity=os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0,{6,7})
        while True:
            current=temperature()
            peak=max(peak,current)
            if workers and any(w.poll() is not None for w in workers):
                raise RuntimeError('preconditioning heater exited before reaching target')
            if low+1000<=current<=high-1000 and not workers:break
            if time.monotonic()-started>timeout:raise TimeoutError(f'temperature band {low}-{high} unreachable, current {current}')
            if workers and current>=heater_stop:
                for worker in workers:worker.terminate()
                for worker in workers:worker.wait(timeout=5)
                workers=[]
                continue
            if current<low+1000 and not workers:
                workers=[subprocess.Popen([str(Path(__file__).with_name('thermal_heater'))],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)]
                heated=True
            time.sleep(.02 if workers else .2)
        return {'low_mC':low,'high_mC':high,'before_mC':before,'after_mC':current,
                'seconds':time.monotonic()-started,'heated':heated,'heater_workers_alive':0,
                'preconditioning_peak_sample_mC':peak,'heater_stop_threshold_mC':heater_stop,
                'scope':'preconditioning only; KV resident; temporary 8-core SIMD heater terminated and reaped before return'}
    finally:
        for worker in workers:
            if worker.poll() is None:worker.terminate()
        for worker in workers:worker.wait(timeout=5)
        os.sched_setaffinity(0,original_affinity)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser();p.add_argument('--min',type=int,required=True);p.add_argument('--max',type=int,required=True)
    a=p.parse_args();print(json.dumps(condition(a.min,a.max)),flush=True)
