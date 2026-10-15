"""Preserve complete speed-search evidence without model weights or credentials."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'results'
out=Path(__file__).resolve().parent/'evidence-qwen15-speed-20261015'
out.mkdir(exist_ok=True)
report={'scope':'full 28-layer submission tuning; model quality still fails',
        'screen':json.loads((root/'qwen15-groups-20261015/summary.json').read_text()),'repeat28':[]}
for i in range(2,6):
    report['repeat28'].extend(json.loads((root/f'qwen15-g28-repeat{i}-20261015/summary.json').read_text()))
for name in ('qwen15-l26-baseline-warm-20261015','qwen15-l26-all-dfp4-capture','qwen15-l26-all-dfp8-capture'):
    d=json.loads((root/name/'board-validation.json').read_text())
    report[name]={k:d[k] for k in ('errors','timing','gate','binary_sha256','input_sha256')}
    fixture=root/name/'probe-fixture.json'
    if fixture.exists():report[name]['constants']=json.loads(fixture.read_text())['constants']
report['profile']=json.loads((root/'qwen15-l26-profile-20261015/profile.json').read_text())
report['profile_binary_sha256']='87a25ac4a2f385162d271b246578ef2c30d81d0bca93f2e2438ec1603095e423'
(out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
