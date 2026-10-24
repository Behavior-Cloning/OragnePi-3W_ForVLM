import subprocess,json
from pathlib import Path
p=Path(__file__).resolve().parent
for run,variant in [('pair300-v2','v2'),('pair300-32','32')]:
 if (p/(run+'-controller.json')).exists():raise RuntimeError('Existing run: inspect instead of duplicate '+run)
 subprocess.run(['python3','-u',str(p/'controller_detached.py'),run,'300',variant],cwd=p,check=True)
 s=json.loads((p/(run+'-controller.json')).read_text())
 if s.get('returncode')!=0:raise RuntimeError(s)
(p/'pair32-status.json').write_text(json.dumps({'status':'complete'}))
