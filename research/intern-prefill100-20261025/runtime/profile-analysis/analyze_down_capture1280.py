"""Verify original/capture replay parity; characterize real NPU internal values."""
from pathlib import Path
import json,hashlib,tarfile,numpy as np,yaml
T=Path(__file__).resolve().parents[2];RUN='down-capture-c1280-replay-r1';D=T/'results'/RUN
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ctl=json.loads((T/(RUN+'-controller.json')).read_text());assert ctl['status']=='complete' and ctl['returncode']==0
assert json.loads((D/'meta.json').read_text())['status']=='complete';cfg=json.loads((D/'config.json').read_text());events=[json.loads(s) for s in (D/'events.jsonl').read_text().splitlines()];results={e['case']:e for e in events if e['event']=='result'};assert len(results)==70
for name,h in cfg['source_sha256'].items():assert sha(T/name)==h
archive=T/(RUN+'.tar.gz');assert sha(archive)==ctl['archive_sha256']
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  if not m.isfile():continue
  f=D/m.name.removeprefix('./');assert D.resolve() in f.resolve().parents;assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==sha(f)
raw={};ports=[]
for e in results.values():
 f=D/(e['case']+'.f32');assert sha(f)==e['output_sha256'] and f.stat().st_size==e['output_bytes'];x=np.fromfile(f,'<f4');assert np.isfinite(x).all();raw[e['case']]=x;z=e['metrics'];assert z['repeats']==2 and z['warmup_repeats']==1 and z['released_bytes']==0 and z['mismatched_elements']==0
 assert len(z['ports']) in [16,20];ports.append({'run':e['case'],'ports':z['ports']})
base_ports=next(e['metrics']['ports'] for e in results.values() if len(e['metrics']['ports'])==16)
expected_inputs=[16*896]+[2*1280*64]*4+[7*16*(1280+16)]*2+[16*64]*4
expected_outputs=[16*896]+[2*16*64]*4
for e in results.values():
 ps=e['metrics']['ports'];assert ps[:16]==base_ports
 expected=[(0,i,n) for i,n in enumerate(expected_inputs)]+[(1,i,n) for i,n in enumerate(expected_outputs)]
 if len(ps)==20:expected += [(1,5,16*4864),(1,6,16*4864),(1,7,16*896),(1,8,16*896)]
 for port,(flag,idx,count) in zip(ps,expected):
  assert (port['output'],port['index'],port['count'])==(flag,idx,count)
  assert int(np.prod(port['sizes']))==count and port['quant_format'] in [0,1]
  if flag and idx>=5:assert port['data_format']==1 and port['quant_format']==0 and port['dfp']==0
qp=T/'graphs/candidates-down-i16-c1280-range16/c1280-m16-l2/decoder.quantize'
q=yaml.safe_load(qp.read_text())['quantize_parameters'];normal=16*(896+4*128);rows=[];allvals={k:[] for k in ['l2_input','l3_input','l2_output','l3_output']};positionvals={k:[] for k in allvals};offsets=[(normal,16*4864),(normal+16*4864,16*4864),(normal+32*4864,16*896),(normal+32*4864+16*896,16*896)]
for f in cfg['fixtures']:
 base=raw[f['key']+'-original'];cap=raw[f['key']+'-capture'];histp=Path(f['original_output_path']);assert sha(histp)==f['original_output_sha256'];hist=np.fromfile(histp,'<f4');assert hist.size==normal;assert sha(Path(f['input_path']))==f['input_sha256']
 samehist=bool(np.array_equal(base.view(np.uint32),hist.view(np.uint32)));samecap=bool(np.array_equal(base.view(np.uint32),cap[:normal].view(np.uint32)))
 repeat=raw.get(f['key']+'-original-after');sameaa=bool(np.array_equal(base.view(np.uint32),repeat.view(np.uint32))) if repeat is not None else None
 assert cap.size==normal+32*4864+32*896
 rec={**f,'normal_equal_historical':samehist,'normal_equal_capture':samecap,'original_after_equal':sameaa,'normal_changed_vs_capture':int((base.view(np.uint32)!=cap[:normal].view(np.uint32)).sum()),'activations':{}}
 for idx,(key,(start,count)) in enumerate(zip(allvals,offsets)):
  width=4864 if idx<2 else 896;part=cap[start:start+count].reshape(16,width);valid=part[:f['valid_rows']];pad=part[f['valid_rows']:];b=idx%2;qkey=f'@down_i16_{2000+b}:out0' if idx<2 else f'@b{b}_fullconnect_116:out0';fl=q[qkey]['fl'];codes=np.rint(valid.astype(np.float64)*2**fl);sat=(codes>32767)|(codes<-32768);peak=np.unravel_index(np.argmax(abs(valid)),valid.shape)
  rec['activations'][key]={'source_output_port':5+idx,'width':width,'valid_elements':valid.size,'padding_elements':pad.size,'maxabs':float(abs(valid).max()),'maxabs_position':f['position']+int(peak[0]),'maxabs_channel':int(peak[1]),'candidate_fl':fl,'candidate_range_positive':32767/2**fl,'candidate_clipped_elements':int(sat.sum()),'candidate_clipped_rows':int(sat.any(1).sum()),'clip_channels':np.where(sat.any(0))[0].tolist(),'padding_maxabs':float(abs(pad).max()) if pad.size else None,'valid_sha256':hashlib.sha256(valid.tobytes()).hexdigest()}
  allvals[key].append(valid);positionvals[key].extend([{'case':f['case'],'position':f['position']+i} for i in range(f['valid_rows'])])
 rows.append(rec)
aggregates={}
for key,xs in allvals.items():
 x=np.concatenate(xs);assert x.shape[0]==528;mx=np.max(abs(x),axis=0);order=np.argsort(-mx,kind='stable');idx=0 if key.startswith('l2') else 1;isinput=key.endswith('input');qkey=f'@down_i16_{2000+idx}:out0' if isinput else f'@b{idx}_fullconnect_116:out0';fl=q[qkey]['fl'];codes=np.rint(x.astype(np.float64)*2**fl);sat=(codes>32767)|(codes<-32768)
 byposition=[]
 for j in np.where(sat.any(1))[0]:byposition.append({**positionvals[key][j],'clipped_elements':int(sat[j].sum()),'channels':np.where(sat[j])[0].tolist(),'maxabs':float(abs(x[j]).max())})
 aggregates[key]={'shape':list(x.shape),'maxabs':float(abs(x).max()),'candidate_clipped_elements':int(sat.sum()),'candidate_clipped_rows':int(sat.any(1).sum()),'clipped_positions':byposition,'top16':[{'channel':int(c),'maxabs':float(mx[c])} for c in order[:16]],'candidate_fl':fl}
answer={'scope':'direct NPU down input/output observation on34 replayed original full-NPU upstream windows from2 prompts;528 valid rows; historical normal-output and diagnostic noninterference gates; no independent quality corpus or latency qualification','integrity':{'windows':34,'valid_rows':528,'padded_rows':16,'runs':70,'api_calls':140,'warmup_calls':70,'driver_zero':70,'normal_equal_historical':sum(r['normal_equal_historical'] for r in rows),'normal_equal_capture':sum(r['normal_equal_capture'] for r in rows),'aa_equal':sum(r['original_after_equal'] is True for r in rows)},'rows':rows,'aggregate':aggregates,'sdk_ports':ports,'source_sha256':{'analyzer':sha(Path(__file__)),'archive':sha(archive),'config':sha(D/'config.json'),'candidate_quantize':sha(qp),'sdk_header':sha(Path('/opt/opi-lab/restored-slim3-20261017/board-runtime/usr/include/vip_lite.h'))},'candidate_range_definitions':{k:q[k] for k in ['@down_i16_2000:out0','@down_i16_2001:out0','@b0_fullconnect_116:out0','@b1_fullconnect_116:out0']},'gate':'pass_all_current_windows' if all(r['normal_equal_historical'] and r['normal_equal_capture'] and r['original_after_equal'] is not False for r in rows) else 'fail_noninterference_or_historical_replay'}
(D/'ANALYSIS.json').write_text(json.dumps(answer,indent=2)+'\n');print(json.dumps({k:answer[k] for k in ['integrity','aggregate','gate']},indent=2))
