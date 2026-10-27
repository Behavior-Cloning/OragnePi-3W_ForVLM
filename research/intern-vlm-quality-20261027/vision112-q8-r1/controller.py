from pathlib import Path
import sys,json,tarfile,hashlib
P=Path(__file__).resolve().parent;R=P.parents[2]
sys.path.insert(0,str(R/'research/intern-prefill100-20261025'))
from controller_down_outputfp16_tmpfs import b,locked,upload
remote='/tmp/opi-vision112-q8-r1'
assert not (P/'status.json').exists()
with b.hostlock():
 z=locked('systemctl show opi-vlm-model.service -p ActiveState --value',capture_output=True,text=True,check=True);assert z.stdout.strip()=='inactive'
 locked('test ! -e '+remote+' && mkdir '+remote,check=True)
 upload(P/'board_bench.py',remote+'/board_bench.py')
 locked('ln -s /tmp/opi-vision-repack-r1/vision-worker '+remote+'/vision-worker',check=True)
 z=locked('sha256sum '+remote+'/vision-worker /opt/opi-vlm/experiments/vision-q8-r1/encoder-q8.gguf /opt/opi-vlm/models/internvl3-1b/mmproj-InternVL3-1B-Instruct-f16.gguf /opt/opi-vlm/experiments/vision-frontmerge-r4/*112.f32',capture_output=True,text=True,check=True,timeout=60)
 (P/'assets.sha256').write_text(z.stdout)
 (P/'status.json').write_text(json.dumps({'state':'running'}))
 z=locked('python3 '+remote+'/board_bench.py',capture_output=True,text=True,timeout=480)
 (P/'controller.log').write_text(z.stdout+z.stderr)
 data=locked('tar czf - -C '+remote+' results',capture_output=True,check=True,timeout=60).stdout
 (P/'raw.tar.gz').write_bytes(data)
 with tarfile.open(P/'raw.tar.gz') as t:
  assert all(not m.issym() and not m.islnk() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(P)
 state=json.loads((P/'results/summary.json').read_text())['state']
 (P/'status.json').write_text(json.dumps({'state':state,'returncode':z.returncode}))
 print((P/'status.json').read_text())
