"""Archive final fingerprints/health, remove only enumerated reproducible transfer duplicates."""
import hashlib,json,shlex,shutil,sys,time
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts')
import board_access as b
repo=Path(__file__).resolve().parents[2];out=repo/'results/audio-models-20261018';assets=Path('/mnt/f/Files/OPi-Lab/audio-models-20261018');remote='/opt/opi-vlm/audio-lab-20261018'
for name in ['standalone-v1','mixed-v1','paced-v1','paced-v2']:
 assert json.loads((out/name/'summary.json').read_text())['status']=='complete',name
assert (out/'quality-v1/result.json').is_file()
with b.hostlock():
 command=f'cd {remote} && df -h . && free -m && du -sh . deps models && ps -eo pid,comm,args | grep -E "[a]udio_worker.py|[b]oard_trials.py|[p]aced_chain.py|[t]hermal_heater|[f]ull_model" || true'
 r=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(command),capture_output=True,text=True,check=True,timeout=30)
 (out/'board-final-health.txt').write_text(r.stdout+r.stderr)
 cmd=f'cd {remote} && rm -f speaker-samples.tar standalone-v1.tar.gz quality-v1.tar.gz mixed-v1.tar.gz paced-v1.tar.gz paced-v2.tar.gz && sha256sum audio_worker.py board_trials.py quality_trials.py paced_chain.py models/*.onnx models/sensevoice/model.int8.onnx'
 r=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,check=True,timeout=60);(out/'board-final-sha256.txt').write_text(r.stdout+r.stderr)
cleanup=[]
for target in [assets/'stage',assets/'deployment.tar']:
 assert target.parent==assets and not target.is_symlink()
 if not target.exists():continue
 size=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()) if target.is_dir() else target.stat().st_size
 if target.is_dir():shutil.rmtree(target)
 else:target.unlink()
 cleanup.append({'path':str(target),'bytes_removed':size,'reason':'rebuildable from retained source assets/wheels/scripts; board deploy and result archives verified'})
(out/'cleanup.json').write_text(json.dumps(cleanup,indent=2))
print((out/'board-final-health.txt').read_text(),flush=True);print('reclaimed bytes',sum(x['bytes_removed'] for x in cleanup),flush=True)
manifest={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for base in [repo/'research/audio-models',out] for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact-index.json'}
(out/'artifact-index.json').write_text(json.dumps({'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'sha256':manifest},indent=2))
