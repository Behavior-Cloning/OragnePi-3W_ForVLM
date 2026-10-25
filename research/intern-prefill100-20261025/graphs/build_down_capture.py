"""Diagnostic-only extra down inputs as outputs; preserve production graph."""
from pathlib import Path
import copy,fcntl,hashlib,json,os,re,shutil,subprocess,time,traceback,uuid
G=Path(__file__).resolve().parent;S=G/'cases/c512-m32-l2';D=G/'down-capture-c512-m32-l2';D.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
st={'state':'waiting_sdk_lock','pid':os.getpid(),'start':time.time(),'scope':'diagnostic7output graph, no performance or noninterference claim'}
def save():write(D/'status.json',st)
save();lock=open('/tmp/opi-prefill-compile.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
try:
 sdk='sha256:b2447e88c70d334cbadf91d1cfef2874d6a4360f0e2b075606df32547ec8cbe3';assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','ubuntu-npu:v2.0.10.2'],text=True).strip()==sdk
 frozen={n:sha(S/n) for n in ['decoder.json','decoder.quantize','decoder.data','tensor-inputmeta.yml','export/decoder.export.data','export_nbg_unify/network_binary.nb']};g=json.loads((S/'decoder.json').read_text());original=copy.deepcopy(g);added=[]
 for b in range(2):
  k=f'z{b}_attach_down_input_12000';g['Layers'][k]={'name':f'z{b}_attach_down_input','op':'output','parameters':{},'inputs':[f'@b{b}_reshape_115:out0'],'outputs':['out0']};added.append(k)
 restored=copy.deepcopy(g)
 for k in added:del restored['Layers'][k]
 assert restored==original;write(D/'decoder.json',g)
 for n in ['decoder.quantize','tensor-inputmeta.yml']:shutil.copy2(S/n,D/n)
 (D/'decoder.data').symlink_to((S/'decoder.data').resolve());write(D/'source-audit.json',{'source_hashes':frozen,'added_output_nodes':{k:g['Layers'][k] for k in added},'all_existing_nodes_unchanged':True,'quantize_identical':sha(D/'decoder.quantize')==frozen['decoder.quantize'],'note':'extra outputs may alter optimization; must compare all original five output tensors on board before interpreting capture'})
 prior=json.loads((S/'conversion.json').read_text());cmd=[x.replace(str(S),str(D)) for x in prior['compile_command']];cmd[cmd.index('--name')+1]='opi-capture-'+uuid.uuid4().hex[:8];cmd[cmd.index('ubuntu-npu:v2.0.10.2')]=sdk;st.update(state='compiling',command=cmd);save()
 with (D/'compile.log').open('w') as log:
  try:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
  except subprocess.TimeoutExpired:subprocess.run(['docker','stop','-t','2',cmd[cmd.index('--name')+1]],stdout=log,stderr=subprocess.STDOUT,timeout=15);raise
 assert rc==0,rc
 assert all(sha(S/n)==h for n,h in frozen.items())
 payload=D/'export/decoder.export.data';assert sha(payload)==frozen['export/decoder.export.data'];payload.unlink();payload.symlink_to((S/'export/decoder.export.data').resolve())
 code=(D/'export/vnn_decoder.c').read_text();tensors={}
 for label,body in re.findall(r'/\* @([^*]+?) \*/(.*?)(?=/\*|$)',code,re.S):
  m=re.search(r'NEW_NORM_TENSOR\(norm_tensor\[(\d+)\]',body)
  if m:tensors[int(m[1])]=label.strip().rsplit(':',1)[0]
 def ports(kind):return [tensors[int(t)] for _,t in sorted(re.findall(r'graph->'+kind+r'\.tensors\[(\d+)\] = norm_tensor\[(\d+)\]',code),key=lambda p:int(p[0]))]
 ins,outs=ports('input'),ports('output');assert ins==prior['input_order'];assert len(outs)==7 and set(outs)==set(prior['output_order']+added);assert [x for x in outs if x not in added]==prior['output_order']
 manifest={'scope':'diagnostic; no actual SDK port dtype/shape introspection yet','first_layer':2,'fusion_width':2,'capacity':512,'query_tokens':32,'source_hashes':frozen,'input_order':ins,'output_order':outs,'normal_output_indices':[outs.index(x) for x in prior['output_order']],'capture_output_indices':[outs.index(x) for x in added],'capture_expected_elements':[32*4864,32*4864],'port_order_verified_from_C':True,'payload_identical':True,'nb_sha256':sha(D/'export_nbg_unify/network_binary.nb'),'command':cmd};write(D/'conversion.json',manifest);st.update(state='complete',ended=time.time(),conversion_sha256=sha(D/'conversion.json'));save()
except BaseException:st.update(state='failed',ended=time.time(),error=traceback.format_exc());save();raise
