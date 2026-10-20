import os,json,gc,time
from pathlib import Path
from board_adapters import Adapter
r=Path(__file__).resolve().parent;tests=json.loads((r/'results/laya-onnx-export.json').read_text())['tests'];os.environ['LAYA_ORT']='int8';a=Adapter('laya',r,2)
for x in tests:
 t=time.perf_counter();x['int8']=a.decide(x['state'],x['options']);x['seconds']=time.perf_counter()-t;x['int8_answer_match']=x['int8']['answer']==x['eager']['answer'];print(x['count'],x['budget'],x['int8_answer_match'],flush=True)
(r/'results/laya-onnx-int8-parity.json').write_text(json.dumps(dict(status='completed',note='Desktop export sanity only; board quality required',tests=tests),ensure_ascii=False,indent=2))
