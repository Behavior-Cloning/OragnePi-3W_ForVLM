"""Archive stale-output diagnostics with hashes and a same-trajectory reference."""
import hashlib
import json
from pathlib import Path

import numpy as np

here=Path(__file__).resolve().parent
root=here.parent/'results/full-intern1b-sharded-v2-c128/w8-fused2-c1280-t1'
archive=here/'evidence-long-context-20261012';archive.mkdir(exist_ok=True)
reference=root/'board-history1053-scalar-fixed'
expected=np.load(reference/'fp32-reference-logits-w8.npy')
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
report={'scope':'diagnostic only; same 1053-history + 20 input + 128 fixed-feedback steps',
    'fp32_reference_sha256':digest(reference/'fp32-reference-logits-w8.npy'),'runs':[]}
for suffix in ('audit1','audit2','capture160','host-hidden','host-outputs','rebind','poison','preflush','preflush-repeat','clear','clear-repeat','scalar-repeat','scalarblock1-v2','scalarblock1-repeat','pin6','pin0'):
    folder=root/('board-history1053-'+suffix)
    if not (folder/'tokens.json').exists() or not (folder/'logits.f32').exists():continue
    if (folder/'logits.f32').stat().st_size!=expected.nbytes:continue
    runtime=json.loads((folder/'tokens.json').read_text())
    assert (folder/'prompt.i32').read_bytes()==(reference/'prompt.i32').read_bytes()
    assert (folder/'feedback.i32').read_bytes()==(reference/'feedback.i32').read_bytes()
    actual=np.fromfile(folder/'logits.f32',np.float32).reshape(expected.shape)
    assert np.isfinite(actual).all()
    cosine=(actual*expected).sum(1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(expected,axis=1))
    row={'case':suffix,'runner':json.loads((folder/'runner-config.json').read_text()),
        'relative_l2':float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
        'min_cosine':float(cosine.min()),'argmax_matches':int((actual.argmax(1)==expected.argmax(1)).sum()),
        'logits_sha256':digest(folder/'logits.f32'),'released_bytes':runtime['driver_after_release_bytes']}
    row['reference_scope']='reuse scalar independent FP32 reference: same graph constants, prompt and fixed feedback'
    row['hashes']={f.name:digest(f) for f in folder.iterdir() if f.name in ('deployed-sha256.txt','capture-input.f32','capture-output.f32','output-audit.tsv')}
    audit=folder/'output-audit.tsv'
    if audit.exists():
        history={};repeated=[]
        for line in audit.read_text().splitlines():
            stage,pos,block,port,count,value=line.split('\t')
            key=(stage,block,port)
            if key in history and value==history[key]['hash'] and stage=='prefill' and port=='0':
                repeated.append({'position':int(pos),'block':int(block),'previous_position':history[key]['position']})
            history[key]={'position':int(pos),'hash':value}
        row['exact_previous_hidden_repetitions']=repeated
    report['runs'].append(row)
mismatch=here.parent/'results/intern1b-history1053-scalarblock1-mismatch'
if (mismatch/'logits.f32').exists():
    runtime=json.loads((mismatch/'tokens.json').read_text())
    assert runtime['feedback_tokens']==json.loads((reference/'tokens.json').read_text())['feedback_tokens']
    actual=np.fromfile(mismatch/'logits.f32',np.float32).reshape(expected.shape)
    cosine=(actual*expected).sum(1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(expected,axis=1))
    report['failed_scalarblock_benchmark']={'host_cpu_affinity':'6,7',
        'relative_l2':float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
        'min_cosine':float(cosine.min()),'argmax_matches':int((actual.argmax(1)==expected.argmax(1)).sum()),
        'logits_sha256':digest(mismatch/'logits.f32'),'released_bytes':runtime['driver_after_release_bytes'],
        'scope':'diagnostic rejected before warmup/measurement; full FP32 reference reused on identical feedback'}
a=root/'board-history1053-audit1/output-audit.tsv';b=root/'board-history1053-audit2/output-audit.tsv'
if a.exists() and b.exists():
    rows_a=a.read_text().splitlines();rows_b=b.read_text().splitlines();assert len(rows_a)==len(rows_b)
    differences=[]
    for x,y in zip(rows_a,rows_b):
        u,v=x.split('\t'),y.split('\t');assert u[:-1]==v[:-1]
        if u[-1]!=v[-1]:differences.append({'stage':u[0],'position':int(u[1]),'block':int(u[2]),'port':int(u[3]),'count':int(u[4]),'hashes':[u[-1],v[-1]]})
    report['comparison']={'total_output_tensors':len(rows_a),'different_output_tensors':len(differences),
        'first_mismatch':differences[0] if differences else None}
(archive/'stale-output-diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({r['case']:{k:r[k] for k in ('relative_l2','min_cosine','argmax_matches')} for r in report['runs']},indent=2))
