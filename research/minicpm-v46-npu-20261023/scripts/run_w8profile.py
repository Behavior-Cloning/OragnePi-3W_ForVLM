from pathlib import Path
import sys,subprocess,fcntl,shlex,json
r=Path(__file__).resolve().parents[1];bld=Path('/opt/opi-lab/minicpm-v46-npu-20261023');sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
lock=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);d='/opt/opi-vlm/minicpm-v46-npu-20261023/npd-w8scaled-l0'
def run(cmd,timeout=45):
 z=b.ssh('flock -n /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=timeout);assert z.returncode==0,z.stdout+z.stderr;return z
run('mkdir -p '+d)
b.scp(bld/'w8profile/l0/export_nbg_unify/network_binary.nb',b.HOST+':'+d+'/model.nb',check=True,timeout=120)
b.scp(bld/'w8retry-source/l0/input.f32',b.HOST+':'+d+'/input.f32',check=True,timeout=30)
z=run('cd '+d+' && timeout --kill-after=5s 40s env OPI_PROBE_NPD=1 taskset -c 6,7 ../probe model.nb input.f32 output.f32 1 timing.json',50);(r/'results/w8profile-board.log').write_text(z.stdout+z.stderr)
names=run('find '+d+' -maxdepth 1 -name "*.npd" -type f').stdout.splitlines();assert names
for src in names:
 dest=r/'results'/('w8profile-'+Path(src).name);b.scp(b.HOST+':'+src,dest,check=True,timeout=30)
 cmd=['/opt/opi-lab/export-venv/bin/python',str(r.parents[1]/'npu/kv-cache/profile_npd.py'),'--folder',str(bld/'w8profile/l0'),'--npd',str(dest),'--output',str(r/'results/w8profile-parsed.json')]
 z=subprocess.run(cmd,capture_output=True,text=True);print(z.stdout,z.stderr,flush=True);assert z.returncode==0
print('COMPLETE',flush=True)
