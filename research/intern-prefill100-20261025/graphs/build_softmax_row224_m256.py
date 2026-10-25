#!/usr/bin/env python3
"""Slice query rows only around softmax; full key axis, source weights and grids retained."""
from pathlib import Path
import copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback,uuid,yaml
R=Path(__file__).resolve().parent;T=R.parent;OUT=R/'candidate-softmax-row224-m256';OUT.mkdir(exist_ok=False)
SRC=R/'cases/c512-m256-l0';SDK='sha256:b2447e88c70d334cbadf91d1cfef2874d6a4360f0e2b075606df32547ec8cbe3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
state={'state':'preparing','start':time.time(),'pid':os.getpid(),'scope':'M256/L0 two-layer softmax row partition only; not full model'}
def save():write(OUT/'status.json',state)
save();lock=open('/tmp/opi-prefill-compile.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
try:
 assert shutil.disk_usage('/mnt/f').free>8*1024**3
 assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','ubuntu-npu:v2.0.10.2'],text=True).strip()==SDK
 frozen={n:sha(SRC/n) for n in ['decoder.json','decoder.quantize','decoder.data','tensor-inputmeta.yml','export/decoder.export.data','export_nbg_unify/network_binary.nb']}
 g=json.loads((SRC/'decoder.json').read_text());q=yaml.safe_load((SRC/'decoder.quantize').read_text());original=copy.deepcopy(g);oq=copy.deepcopy(q);quant=q['quantize_parameters'];edits=[]
 for b in range(2):
  key=f'b{b}_Softmax_output_0_50';node=g['Layers'][key];assert node['op']=='softmax' and node['parameters']=={'sf_axis':-1,'beta':1.0}
  inputref=node['inputs'][0];outquant=quant['@'+key+':out0'];inputquant=quant[inputref]
  assert inputquant=={'qtype':'float16','quantizer':'float16'} and outquant['qtype']=='i16' and outquant['fl']==15
  parts=[]
  for j in range(8):
   sl=f'b{b}_softmax_rowslice_{21000+2*j}';sm=f'b{b}_softmax_rowpart_{21001+2*j}'
   g['Layers'][sl]={'name':sl,'op':'slice','parameters':{'begin':[0,0,j*224,0],'size':[1,2,224,768]},'inputs':[inputref],'outputs':['out0']}
   g['Layers'][sm]={'name':sm,'op':'softmax','parameters':copy.deepcopy(node['parameters']),'inputs':['@'+sl+':out0'],'outputs':['out0']}
   quant['@'+sl+':out0']=copy.deepcopy(inputquant);quant['@'+sm+':out0']=copy.deepcopy(outquant);parts.append('@'+sm+':out0')
  g['Layers'][key]={**node,'op':'concat','parameters':{'dim':2},'inputs':parts}
  edits.append({'node':key,'row_axis':2,'key_axis':3,'original_shape':[1,2,1792,768],'chunk_shape':[1,2,224,768],'chunks':8,'original_node':node,'replacement':g['Layers'][key]})
 assert all(g['Layers'][k]==v for k,v in original['Layers'].items() if v['op']!='softmax')
 assert all(quant[k]==v for k,v in oq['quantize_parameters'].items())
 assert len(g['Layers'])==len(original['Layers'])+32 and len(quant)==len(oq['quantize_parameters'])+32
 write(OUT/'decoder.json',g);(OUT/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False));shutil.copy2(SRC/'tensor-inputmeta.yml',OUT/'tensor-inputmeta.yml');(OUT/'decoder.data').symlink_to((SRC/'decoder.data').resolve())
 write(OUT/'graph-audit.json',{'source_sha256':frozen,'edits':edits,'all_other_original_nodes_equal':True,'all_original_quantizers_equal':True,'full_key_axis_preserved':768,'source_weight_file_identical':True,'numerical_parity':'not measured; different scheduling may change numeric output'})
 cmd=json.loads((SRC/'conversion.json').read_text())['compile_command'];cmd=[x.replace(str(SRC),str(OUT)) for x in cmd];cmd[cmd.index('--name')+1]='opi-softmax-'+uuid.uuid4().hex[:10];cmd[3:3]=['-e','VIV_VX_DUMP_NODE_TRACE_DB=1'];cmd[cmd.index('ubuntu-npu:v2.0.10.2')]=SDK
 state.update(state='compiling',command=cmd,source_sha256=frozen,graph_sha256=sha(OUT/'decoder.json'),quantize_sha256=sha(OUT/'decoder.quantize'));save()
 with (OUT/'export.log').open('w') as log:
  try:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
  except subprocess.TimeoutExpired:
   subprocess.run(['docker','stop','-t','2',cmd[cmd.index('--name')+1]],stdout=log,stderr=subprocess.STDOUT,timeout=15);raise
 assert rc==0,rc
 assert all(sha(SRC/n)==h for n,h in frozen.items())
 payload=OUT/'export/decoder.export.data';assert sha(payload)==frozen['export/decoder.export.data'];payload.unlink();payload.symlink_to((SRC/'export/decoder.export.data').resolve())
 sys.path.insert(0,str(T.parents[1]/'npu/kv-cache'));from fused_contract import verify_contract
 ports=verify_contract(OUT,2);old=json.loads((SRC/'conversion.json').read_text())
 assert ports['input_order']==old['input_order'] and ports['output_order']==old['output_order']
 nb=OUT/'export_nbg_unify/network_binary.nb';state.update(state='compiled',end=time.time(),nb_sha256=sha(nb),nb_bytes=nb.stat().st_size,actual_ports_exact=True,ports=ports,payload_byte_equal=True,source_hashes_unchanged=True);save();print(json.dumps({k:v for k,v in state.items() if k not in ['command','ports','source_sha256']}))
except BaseException:state.update(state='failed',end=time.time(),error=traceback.format_exc());save();raise
