"""Compose immutable compiled layers into a new full-model experiment."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--base',required=True);p.add_argument('--dtype',required=True)
p.add_argument('--replace',type=int,nargs='+',required=True)
args=p.parse_args();source=args.source.resolve();out=source/args.dtype;out.mkdir(exist_ok=True)
records=json.loads((source/args.base/'stack-conversion.json').read_text())
for record in records:
    i=record['layer'];folder=out/f'l{i}'
    if i not in args.replace:
        target=source/args.base/f'l{i}'
        if folder.is_symlink():assert folder.resolve()==target.resolve()
        else:folder.symlink_to(target,target_is_directory=True)
    else:
        conversion=json.loads((folder/'conversion.json').read_text())
        assert conversion['exit_code']==0 and conversion['binaries'],i
        record['precision']='W8 attention + FP16 MLP' if conversion.get('float_mlp') else 'W8 with selected floating activations'
        record['precision_overrides']={key:conversion.get(key) for key in
            ('float_mlp','float_residual','float_sensitive','float_down_output','activation_headroom_bits')}
    record['path']=str(folder/'export_nbg_unify/network_binary.nb')
    record['composed_from']=args.dtype if i in args.replace else args.base
    assert Path(record['path']).is_file()
(out/'stack-conversion.json').write_text(json.dumps(records,indent=2)+'\n')
