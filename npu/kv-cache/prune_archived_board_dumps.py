"""Free board diagnostic copies only when a complete local SHA256 backup exists."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    root=Path(__file__).resolve().parent.parent/'results'
    names=('logits.f32','hidden-trace.f32','decode-block-trace.f32')
    backups={}
    for name in names:
        for path in sorted(root.glob('full-*/*/board*/'+name)):
            if path.is_file() and path.stat().st_size:
                sha=digest(path)
                backups.setdefault(sha,{'path':str(path.resolve()),'bytes':path.stat().st_size})
    from board_host_lock import lock_board_from_host
    lock_board_from_host()
    code=r'''
import hashlib,json,pathlib,shutil
backups=BACKUPS
root=pathlib.Path('/opt/opi-vlm/npu-kv-probe');before=shutil.disk_usage(root).free
for name in ('logits.f32','hidden-trace.f32','decode-block-trace.f32'):
    for path in sorted(root.glob('*/**/'+name)):
        if path.is_symlink() or not path.is_file() or path.resolve()!=path:continue
        stat=path.stat()
        if not stat.st_size:continue
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        sha=h.hexdigest();local=backups.get(sha)
        if not local or local['bytes']!=stat.st_size:continue
        # The host rechecks its exact local file before authorizing this unlink.
        row={'remote':str(path),'backup':local['path'],'sha256':sha,'bytes':stat.st_size}
        print(json.dumps({'verify':row}),flush=True)
        reply=json.loads(input());assert reply=={'verified_sha256':sha}
        current=path.stat()
        assert (current.st_ino,current.st_size,current.st_mtime_ns)==(stat.st_ino,stat.st_size,stat.st_mtime_ns)
        path.unlink();print(json.dumps({'removed':row}),flush=True)
print(json.dumps({'summary':{'free_before':before,'free_after':shutil.disk_usage(root).free}}),flush=True)
'''.replace('BACKUPS',repr(backups),1)
    prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
    ssh=prefix+['ssh','-T','-o','ForwardX11=no','-o','ConnectTimeout=8','root@192.168.88.29']
    report={'status':'running','files':[],'scope':'only board dumps with byte-identical complete local backups'}
    def save():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output)
    save()
    try:
        with subprocess.Popen(ssh+['flock -w 120 /run/opi-npu-kv-benchmark.lock python3 -c '+shlex.quote(code)],
                              text=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as process:
            for line in process.stdout:
                event=json.loads(line)
                if 'verify' in event:
                    row=event['verify'];local=Path(row['backup'])
                    assert local.is_relative_to(root) and local.stat().st_size==row['bytes']
                    assert digest(local)==row['sha256']
                    process.stdin.write(json.dumps({'verified_sha256':row['sha256']})+'\n');process.stdin.flush()
                elif 'removed' in event:report['files'].append(event['removed']);save()
                else:report.update(event['summary']);save()
            if process.wait():raise RuntimeError('board dump prune failed; consult recorded completed removals')
        report.update(status='complete',removed_bytes=sum(r['bytes'] for r in report['files']));save()
        print(json.dumps({k:v for k,v in report.items() if k!='files'}))
    except BaseException as exc:
        report.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
