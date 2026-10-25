#!/usr/bin/env python3
"""Optional paired single-block fixtures, cropped from frozen c1280 data."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;TASK=R.parent;REPO=TASK.parents[1]
META=REPO/'research/intern-nextopt-20261025/prefill8-fixtures.json'
OLD=Path('/opt/opi-lab/intern-nextopt-20261025')
BASE=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export/w8-fused2-c1280-t16-cap1-productwide/l0')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads(META.read_text());rows=[]
for case in d['cases']:
 for n in [16,8]:
  old=BASE if n==16 else OLD/'prefill8/l0'
  new=TASK/'full-stack256/cases/prefill16/l0' if n==16 else R/'cases/l0'
  inp=OLD/'fixtures'/case['id']/f't{n}/input.f32'
  assert sha(inp)==case['variants'][str(n)]['input_sha256']
  out=R/'fixtures'/case['id']/f't{n}'
  subprocess.run(['/opt/opi-lab/export-venv/bin/python',str(TASK/'crop_fixture.py'),'--old',str(old),'--new',str(new),'--input',str(inp),'--output',str(out),'--history',str(case['history'])],check=True,capture_output=True,text=True)
  ref=OLD/'fixtures'/case['id']/f't{n}/reference.f32';assert sha(ref)==case['variants'][str(n)]['reference_sha256']
  (out/'reference.f32').symlink_to(ref)
  rows.append({'id':case['id'],'history':case['history'],'capacity':256,'query_tokens':n,'valid_tail_tokens':8,'input':str(out/'probe-input.f32'),'input_sha256':sha(out/'probe-input.f32'),'reference_sha256':sha(ref),'padding':'t16 uses last-token padding for rows8..15; only compare first8 query rows per output tensor' if n==16 else 'none','source_input_sha256':case['variants'][str(n)]['input_sha256']})
(R/'fixture-manifest.json').write_text(json.dumps({'history_scope':d['history_scope'],'source_manifest':str(META),'source_manifest_sha256':sha(META),'input_manifest_sha256':d['input_manifest_sha256'],'valid_semantics':'same8 rows; c1280->c256 removes masked unused KV only; effective history/positions unchanged','rows':rows},indent=2)+'\n')
print(json.dumps({'prepared':len(rows),'files_bytes':sum(Path(x['input']).stat().st_size for x in rows)}))
