"""Preserve all experiment paths, sharing only hash-identical immutable graphs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    from board_host_lock import lock_board_from_host
    lock_board_from_host()
    prefix=[str(Path.home()/'.local/bin/sshpass'),'-e'] if 'SSHPASS' in os.environ else []
    ssh=prefix+['ssh','-T','-o','ForwardX11=no','-o','ConnectTimeout=8','root@192.168.88.29']
    code=r'''
import hashlib,json,os,pathlib,shutil,time
root=pathlib.Path('/opt/opi-vlm/npu-kv-probe')
assert root.is_dir() and not root.is_symlink()
before=shutil.disk_usage(root).free
paths=sorted([p for pattern in ('*/layer*.nb','*/prefill-layer*.nb','*/embeddings.f32') for p in root.glob(pattern)
              if p.is_file() and not p.is_symlink() and p.resolve()==p])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
seen_inodes={};canonical={};rows=[]
for path in paths:
    stat=path.stat();inode=(stat.st_dev,stat.st_ino)
    if inode in seen_inodes:continue
    digest=sha(path);key=(stat.st_dev,stat.st_size,digest);seen_inodes[inode]=digest
    if key not in canonical:
        canonical[key]=path;continue
    original=canonical[key]
    current=path.stat()
    assert (current.st_ino,current.st_size,current.st_mtime_ns)==(stat.st_ino,stat.st_size,stat.st_mtime_ns)
    assert original.stat().st_size==stat.st_size
    temporary=path.with_name('.dedup-'+str(os.getpid())+'-'+path.name)
    assert not temporary.exists()
    os.link(original,temporary);os.replace(temporary,path)
    assert path.stat().st_ino==original.stat().st_ino
    row={'path':str(path),'canonical':str(original),'sha256':digest,'bytes':stat.st_size,
         'reclaimed_bytes':stat.st_size if stat.st_nlink==1 else 0}
    print(json.dumps({'file':row}),flush=True);rows.append(row)
print(json.dumps({'summary':{'scope':'immutable NPU graph/embedding duplicates only; paths and bytes retained',
                            'files_scanned':len(paths),'unique_inodes_hashed':len(seen_inodes),
                            'linked_files':len(rows),'reclaimed_bytes':sum(r['reclaimed_bytes'] for r in rows),
                            'free_before':before,'free_after':shutil.disk_usage(root).free}}),flush=True)
'''
    record={'status':'running','files':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(a.output)
    save()
    try:
        with subprocess.Popen(ssh+['flock -w 120 /run/opi-npu-kv-benchmark.lock python3 -c '+shlex.quote(code)],
                              text=True,stdout=subprocess.PIPE) as process:
            for line in process.stdout:
                row=json.loads(line)
                if 'file' in row:record['files'].append(row['file'])
                else:record.update(row['summary'])
                save()
            if process.wait():raise RuntimeError('board dedup failed; completed links retained')
        record['status']='complete';save()
        print(json.dumps({k:v for k,v in record.items() if k!='files'}))
    except BaseException as exc:
        record.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
