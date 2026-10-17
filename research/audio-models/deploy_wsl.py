import hashlib,json,shlex,subprocess,sys,tarfile,zipfile
from pathlib import Path
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts')
import board_access as b
repo=Path(__file__).resolve().parents[2];root=Path('/mnt/f/Files/OPi-Lab/audio-models-20261018');stage=root/'stage';stage.mkdir(exist_ok=True)
(stage/'deps').mkdir(exist_ok=True);(stage/'models').mkdir(exist_ok=True);(stage/'audio').mkdir(exist_ok=True)
for wheel in (root/'wheels').glob('*.whl'):
 with zipfile.ZipFile(wheel) as z:z.extractall(stage/'deps')
for name in ['silero_vad.onnx','gtcrn_simple.onnx','3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx']:
 (stage/'models'/name).write_bytes((root/name).read_bytes())
with tarfile.open(root/'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2') as t:
 for member in t.getmembers():
  if not member.isfile():continue
  name=Path(member.name).name
  if name in ['model.int8.onnx','tokens.txt']:
   dest=stage/'models/sensevoice'/name
  elif name.endswith('.wav'):dest=stage/'audio'/name
  else:continue
  dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(t.extractfile(member).read())
for script in ['audio_worker.py','board_trials.py']:(stage/script).write_bytes((repo/'research/audio-models'/script).read_bytes())
hashes={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.rglob('*') if p.is_file()}
(root/'deployment-files.json').write_text(json.dumps(hashes,indent=2)+'\n')
archive=root/'deployment.tar'
with tarfile.open(archive,'w') as t:
 for p in stage.iterdir():t.add(p,arcname=p.name)
remote='/opt/opi-vlm/audio-lab-20261018';out=repo/'results/audio-models-20261018'
with b.hostlock():
 b.ssh('mkdir -p '+remote,check=True,timeout=30)
 b.scp(archive,'root@192.168.10.11:'+remote+'/deployment.tar',check=True,timeout=600)
 h=hashlib.sha256(archive.read_bytes()).hexdigest()
 cmd='cd '+remote+' && echo '+shlex.quote(h+'  deployment.tar')+' | sha256sum -c - && tar -xf deployment.tar && rm deployment.tar && PYTHONPATH=deps python3 -c '+shlex.quote('import sherpa_onnx as s,numpy as n; print(s.__version__,n.__version__); print(s.onnxruntime_version); print([x for x in dir(s.OnlineSpeechDenoiser) if not x.startswith("_")])')
 r=b.ssh('flock -w 30 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),capture_output=True,text=True,timeout=180);(out/'deploy.log').write_text(r.stdout+r.stderr);print(r.stdout+r.stderr);r.check_returncode()
 (out/'deployment-files.json').write_text(json.dumps(hashes,indent=2)+'\n')
