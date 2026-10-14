"""Refresh actual artifact provenance after an explicit mixed-stack assembly."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
path=a.folder/'stack-conversion.json';rows=json.loads(path.read_text())
for r in rows:
    folder=a.folder/f"l{r['layer']}"
    binary=folder/'export_nbg_unify/network_binary.nb'
    r['path']=str(binary.resolve());r['bytes']=binary.stat().st_size
    r['sha256']=hashlib.file_digest(binary.open('rb'),'sha256').hexdigest()
    conversion=json.loads((folder/'conversion.json').read_text())
    for key in ('weight_bits_requested','float_sensitive','float_residual','float_nonlinear'):
        if key in conversion:r[key]=conversion[key]
    r['dfp_linear']=conversion.get('dfp_linear',[])
path.write_text(json.dumps(rows,indent=2)+'\n')
