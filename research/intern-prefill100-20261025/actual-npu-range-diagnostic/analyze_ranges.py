from pathlib import Path
import json,hashlib,numpy as np,torch,yaml,sys,time,traceback
P=Path(__file__).resolve().parent;R=P.parent;D=P/'results/actual-npu-range-r1'
sys.path.insert(0,str(R/'graphs/candidates-fp16'));from graph_reference_conv import evaluate
torch.set_num_threads(2)
S=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export/w8-fused2-c1280-t16-cap1-productwide/l2')
g=json.loads((S/'decoder.json').read_text());conv=json.loads((S/'conversion.json').read_text());data=np.load(S/'decoder.data',allow_pickle=True).item()
q=yaml.safe_load((R/'graphs/candidates-down-i16-c1280-range16/c1280-m16-l2/decoder.quantize').read_text())['quantize_parameters']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
st={'status':'running','started':time.time(),'scope':'actual NPU block1 captured IO; FP32 source interpreter gives estimated internal ranges, not measured internal quantized activations','rows':[]}
def save():(P/'analysis.json').write_text(json.dumps(st,ensure_ascii=False,indent=2))
def metrics(a,b):
 d=a.astype(np.float64)-b.astype(np.float64);return dict(relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(b.astype(np.float64)),1e-30)),max_abs=float(abs(d).max()),exact=bool(np.array_equal(a,b)))
try:
 meta=json.loads((D/'meta.json').read_text());assert meta['status']=='complete'
 events=[json.loads(s) for s in (D/'events.jsonl').read_text().splitlines()]
 results=[e for e in events if e['event']=='result'];closed=[e for e in events if e['event']=='closed'];assert len(results)==len(closed)==8
 assert all(e['metrics']['driver_bytes']==0 for e in closed)
 captures=[e for e in results if e['capture']];assert len(captures)==4 and all(e['capture_control_exact'] for e in captures)
 for e in results:
  f=D/e['name']/'logits'/str(e['metrics']['id']);f=f.with_name(f.name+'-logits.f32');assert sha(f)==e['logits_sha256'];assert np.isfinite(np.fromfile(f,dtype='<f4')).all()
  for c in e['captures']:
   f=D/e['name']/'io'/c['name'];assert sha(f)==c['sha256'] and f.stat().st_size==c['bytes'];assert np.isfinite(np.fromfile(f,dtype='<f4')).all()
 st['integrity']={'results':8,'driver_zero':8,'capture_control_exact':4,'capture_files':sum(len(e['captures']) for e in captures)};save()
 with torch.inference_mode():
  for e in captures:
   for f in sorted((D/e['name']/'io').glob('*-input.f32')):
    raw=np.fromfile(f,dtype='<f4');offset=0;feeds={}
    for key in conv['input_order']:
     node=g['Layers'][key];shape=node['parameters']['shape'];n=int(np.prod(shape));feeds[node['name']]=torch.from_numpy(raw[offset:offset+n].copy().reshape(shape));offset+=n
    assert offset==raw.size,(offset,raw.size)
    output,cache=evaluate(g,data,feeds,return_intermediates=True);ranges=[]
    for i in range(2):
     for key,qkey in [(f'b{i}_reshape_115',f'@down_i16_{2000+i}:out0'),(f'b{i}_fullconnect_116',f'@b{i}_fullconnect_116:out0')]:
      x=cache[key].numpy();fl=q[qkey]['fl'];codes=np.rint(x.astype(np.float64)*2**fl);sat=(codes>32767)|(codes<-32768)
      ranges.append(dict(node=key,fl=fl,maxabs=float(abs(x).max()),range_abs=32767/2**fl,saturated_elements=int(sat.sum()),elements=int(x.size),affected_rows=int(np.any(sat.reshape(16,-1),axis=1).sum())))
    actual=np.fromfile(f.with_name(f.name.replace('-input','-output')),dtype='<f4');hidden=actual[:16*896]
    row=dict(name=e['name'],case=e['case'],variant=e['variant'],file=f.name,ranges=ranges)
    if e['variant']=='c1280-i16-range16-l2':
     other=D/e['name'].replace('c1280-i16-range16-l2','c1280-original')/'io'
     original=np.fromfile(other/f.name.replace('-input','-output'),dtype='<f4')
     originput=np.fromfile(other/f.name,dtype='<f4')
     row['actual_hidden_vs_original']=metrics(hidden,original[:16*896]);row['actual_input_vs_original']=metrics(raw,originput)
     row['actual_kv_vs_original']=metrics(actual[16*896:],original[16*896:])
    st['rows'].append(row);save();print(e['name'],f.name,[(r['node'],r['saturated_elements'],round(r['maxabs'],3)) for r in ranges],flush=True)
 st.update(status='complete',ended=time.time());save()
except BaseException:st.update(status='failed',error=traceback.format_exc());save();raise
