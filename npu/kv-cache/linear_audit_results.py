"""Prepare a shared-input board matrix or evaluate its raw linear outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True)
p.add_argument('--validate',action='store_true');p.add_argument('--variants',nargs='+',default=['int16','w8a16'])
args=p.parse_args();folder=args.folder
fixture=json.loads((folder/'fixture.json').read_text())
records=[{'name':v,'layer':i,'binary':v+'/export_nbg_unify/network_binary.nb','input':'linear-input.f32',
          'count':fixture['input_count'],'output_count':fixture.get('output_count',fixture['input_count']),
          'cases':fixture['cases']} for i,v in enumerate(args.variants)]
if not args.validate:
    (folder/'activation-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
else:
    expected=np.fromfile(folder/'linear-reference.f32',np.float32).reshape(fixture['cases'],-1)
    summary=[]
    for r in records:
        actual=np.fromfile(folder/(r['name']+'-output.f32'),np.float32).reshape(expected.shape)
        cos=(actual*expected).sum(-1)/np.maximum(np.linalg.norm(actual,axis=-1)*np.linalg.norm(expected,axis=-1),1e-12)
        summary.append({'variant':r['name'],'relative_l2':float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
                        'min_nonzero_cosine':float(cos[1:].min()),'max_abs_error':float(abs(actual-expected).max()),
                        'per_case_relative_l2':(np.linalg.norm(actual-expected,axis=-1)/np.maximum(np.linalg.norm(expected,axis=-1),1e-12)).tolist()})
    (folder/'linear-validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
