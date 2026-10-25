from pathlib import Path
import json,hashlib,sys,numpy as np,torch
T=Path(__file__).resolve().parents[2];D=T/'results/down-capture-c512-m32-aba-r1';S=T/'graphs/cases/c512-m32-l2';F=T/'fixtures-canonical/c512-m32-l2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((T/'down-capture-c512-m32-aba-r1-controller.json').read_text())['returncode']==0
assert json.loads((D/'meta.json').read_text())['status']=='complete'
events=[json.loads(x) for x in (D/'events.jsonl').read_text().splitlines()];rs=[x for x in events if x['event']=='result'];assert len(rs)==3
raw={}
for e in rs:
 p=D/(e['case']+'.f32');assert sha(p)==e['output_sha256'] and p.stat().st_size==e['output_bytes'];x=np.fromfile(p,'<f4');assert np.isfinite(x).all();raw[e['case']]=x;assert e['metrics']['released_bytes']==0
normal=32*(896+4*128);assert raw['capture'].size==normal+2*32*4864
base=raw['original-before'];after=raw['original-after'];assert base.size==normal and np.array_equal(base.view(np.uint32),after.view(np.uint32))
cand=raw['capture'][:normal];delta=cand.astype(np.float64)-base;unchanged=bool(np.array_equal(cand.view(np.uint32),base.view(np.uint32)))
sys.path.insert(0,str(T/'graphs/candidates-fp16'));from graph_reference_conv import evaluate
g=json.loads((S/'decoder.json').read_text());data=np.load(S/'decoder.data',allow_pickle=True).item();conv=json.loads((S/'conversion.json').read_text());x=np.fromfile(F/'probe-input.f32','<f4');off=0;feeds={}
for k in conv['input_order']:
 n=g['Layers'][k];shape=n['parameters']['shape'];count=int(np.prod(shape));feeds[n['name']]=torch.from_numpy(x[off:off+count].copy().reshape(shape));off+=count
assert off==x.size;torch.set_num_threads(2)
with torch.inference_mode():_,cache=evaluate(g,data,feeds,return_intermediates=True)
captures=[]
for i in range(2):
 y=raw['capture'][normal+i*32*4864:normal+(i+1)*32*4864].reshape(32,4864);ref=cache[f'b{i}_reshape_115'].numpy();diff=y.astype(np.float64)-ref;mx=np.max(abs(y),axis=0);ix=np.argsort(-mx,kind='stable')[:16]
 captures.append({'layer':i+2,'elements':y.size,'maxabs':float(abs(y).max()),'gt8':int((abs(y)>8).sum()),'gt16':int((abs(y)>16).sum()),'top_channels':[{'channel':int(c),'maxabs':float(mx[c])} for c in ix],'relative_l2_vs_graph_fp32':float(np.linalg.norm(diff)/max(np.linalg.norm(ref),1e-30)),'max_abs_vs_graph_fp32':float(abs(diff).max()),'f32_sha256':hashlib.sha256(y.tobytes()).hexdigest()})
result={'scope':'single canonical FP32-prefix history128 input. Actual down activations inside this NPU fused block; not full-model NPU-history capture, not complete input distribution or system pos0. Diagnostic graph not a speed candidate.','normal_outputs_byte_equal':unchanged,'normal_changed_elements':int((cand.view(np.uint32)!=base.view(np.uint32)).sum()),'normal_relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(base),1e-30)),'normal_max_abs':float(abs(delta).max()),'original_before_after_byte_equal':True,'captures':captures,'timing_ms':{e['case']:{'p50':float(np.median(e['metrics']['api_seconds'])*1000),'p95':float(np.quantile(e['metrics']['api_seconds'],.95)*1000)} for e in rs},'source_sha256':{'analyzer':sha(Path(__file__)),'source_graph':sha(S/'decoder.json'),'source_data':sha(S/'decoder.data'),'input':sha(F/'probe-input.f32')},'gate':'capture same normal outputs on this input only' if unchanged else 'normal output changed: reject unperturbed-observation claim'}
(D/'ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
