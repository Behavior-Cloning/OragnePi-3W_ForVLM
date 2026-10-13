"""Hard-link byte-identical generated decoder.data within named experiment roots."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args();assert not a.output.exists()
    results=Path(__file__).resolve().parent.parent/'results'
    found={};rows=[];paths=[]
    for root in a.root:
        root=root.resolve();assert root.is_relative_to(results) and root!=results
        for path in sorted(root.glob('l*/decoder.data')):
            if path.is_symlink() or path.parent.is_symlink():continue
            assert path.is_file()
            paths.append(path)
    # Linux protected_hardlinks forbids linking root-owned SDK exports. Prefer
    # a byte-identical user-owned source; replacing a file in our directory is
    # then possible without changing permissions or requiring elevated access.
    for path in sorted(set(paths),key=lambda p:(p.stat().st_uid!=os.getuid(),str(p))):
            stat=path.stat();digest=sha(path);key=(stat.st_dev,stat.st_size,digest)
            if key not in found:found[key]=path;continue
            original=found[key]
            if original.stat().st_ino==stat.st_ino:continue
            row={'path':str(path),'canonical':str(original),'sha256':digest,'bytes':stat.st_size,'status':'duplicate',
                 'reclaimable_bytes':stat.st_size if stat.st_nlink==1 else 0}
            if a.apply:
                # Recheck before atomic replacement. No changes to file bytes.
                current=path.stat();assert (current.st_ino,current.st_size,current.st_mtime_ns)==(stat.st_ino,stat.st_size,stat.st_mtime_ns)
                assert sha(original)==digest
                temporary=path.with_name('.decoder.data-hardlink-'+str(os.getpid()))
                assert not temporary.exists()
                try:
                    os.link(original,temporary)
                except PermissionError:
                    row['status']='skipped-source-link-permission'
                    rows.append(row)
                    continue
                os.replace(temporary,path)
                assert path.stat().st_ino==original.stat().st_ino
                row['status']='hardlinked'
            rows.append(row)
    report={'apply':a.apply,'files':rows,'reclaimed_bytes':sum(r['reclaimable_bytes'] for r in rows if r['status']=='hardlinked')}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('identical files',len(rows),'reclaimed MiB',report['reclaimed_bytes']/2**20)


if __name__=='__main__':main()
