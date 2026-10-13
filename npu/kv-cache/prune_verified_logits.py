"""Remove only byte-identical board diagnostic copies already archived locally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args();assert not a.output.exists()
    result_root=Path(__file__).resolve().parent.parent/'results'
    wanted=[]
    for path in a.report:
        d=json.loads(path.read_text());assert d['status']=='complete'
        c=d['config'];assert c['fixed_feedback']
        assert c['experiment'].replace('-','').isalnum()
        for v in c['variants']:
            diag=[r for r in d['runs'] if r['phase']=='diagnostic' and r['label']==v['label']]
            assert len(diag)==1
            digest=diag[0]['logits_sha256'];assert digest==v['expected_logits_sha256']
            assert v['case'].startswith('board-') and v['label'].replace('-','').isalnum()
            candidates=list(result_root.glob('full-*/*/'+v['case']+'/logits.f32'))
            matching=[local for local in candidates if sha(local)==digest]
            assert matching,'no complete local logits backup for '+v['case']
            local=matching[0].resolve()
            remote=Path(v['remote'])/(c['experiment']+'-'+v['label']+'-diagnostic-0')/'logits.f32'
            wanted.append({'remote':str(remote),'sha256':digest,'bytes':local.stat().st_size,'backup':str(local)})
    assert len({v['remote'] for v in wanted})==len(wanted)
    from board_host_lock import lock_board_from_host
    lock_board_from_host()
    code=r'''
import hashlib,json,pathlib
c=CONFIG
rows=[]
for row in c['files']:
    path=pathlib.Path(row['remote'])
    assert path.is_absolute() and path.name=='logits.f32' and 'diagnostic-0' in path.parent.name
    assert path.resolve()==path and path.is_relative_to('/opt/opi-vlm/npu-kv-probe')
    if not path.exists():
        rows.append({**row,'status':'already-absent'});continue
    assert path.is_file() and path.stat().st_size==row['bytes']
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==row['sha256'],'backup mismatch: '+str(path)
    rows.append({**row,'status':'verified'})
# Every target is validated before any unlink; JSON/model/timing files stay.
for row in rows:
    if c['apply'] and row['status']=='verified':
        pathlib.Path(row['remote']).unlink();row['status']='duplicate-removed'
print(json.dumps({'apply':c['apply'],'files':rows,'removed_bytes':sum(r['bytes'] for r in rows if r['status']=='duplicate-removed')}))
'''.replace('CONFIG',repr({'files':wanted,'apply':a.apply}),1)
    prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
    ssh=prefix+['ssh','-T','-o','ForwardX11=no','-o','ConnectTimeout=8','root@192.168.88.29']
    run=subprocess.run(ssh+['flock -w 1800 /run/opi-npu-kv-benchmark.lock python3 -c '+shlex.quote(code)],
                       text=True,capture_output=True,check=True,timeout=2000)
    result=json.loads(run.stdout);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
