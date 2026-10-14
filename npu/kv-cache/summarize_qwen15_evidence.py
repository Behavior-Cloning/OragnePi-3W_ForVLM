"""Export compact, credential-free evidence for the research branch."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'results'
source=root/'full-qwen25-15b-sharded-min-c128'
report={'scope':'full 28-layer Qwen2.5-1.5B experiments; quality NOT accepted','runs':[]}
for variant,case in [('w4-vdfp-sensitive','companion-vdfp'),('w4-vdfp-last2w8','companion-last2w8'),
                     ('w4-vdfp-sensitive','clean-grouphead'),('w4-vdfp-sensitive','clean-group4'),
                     ('w4-vdfp-sensitive','clean-group4-r2'),('w4-vdfp-sensitive','clean-group4-r3')]:
    folder=source/variant/('board-'+case)
    if not (folder/'tokens.json').exists():continue
    d=json.loads((folder/'tokens.json').read_text())
    row={'variant':variant,'case':case,'runtime':d}
    check=folder/'full-validation-w8.json'
    if check.exists():
        v=json.loads(check.read_text())
        row['validation']={k:v[k] for k in ('npu_answer','relative_l2_all_logits','argmax_match_fraction',
            'min_step_cosine','execution_gate_v1','hidden_trace_errors')}
    report['runs'].append(row)
for name in ('qwen15-sensitive-l0-p0','qwen15-sensitive-vdfp-l0-p0','qwen15-sensitive-vdfp-l0-p43',
             'qwen15-l26-p43','qwen15-l26-capture44'):
    d=json.loads((root/name/'board-validation.json').read_text())
    report[name]={k:d[k] for k in ('errors','gate','timing','binary_sha256','input_sha256')}
report['exported_fp32_free_generation']=json.loads((root/'qwen15-exported-greedy.json').read_text())
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
