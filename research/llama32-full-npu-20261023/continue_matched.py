from pathlib import Path
import json,time,subprocess
p=Path(__file__).resolve().parent
while True:
 s=json.loads((p/'q8-probe-controller-status.json').read_text())
 if s['state']=='finished':break
 if s['state']=='error':raise RuntimeError(s)
 time.sleep(10)
cmd=['/opt/opi-lab/export-venv/bin/python','-u',str(p/'intern-match/controller.py'),'--execute']
r=subprocess.run(cmd);(p/'matched-continuation-status.json').write_text(json.dumps({'returncode':r.returncode}))
raise SystemExit(r.returncode)
