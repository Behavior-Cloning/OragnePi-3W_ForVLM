from pathlib import Path
import json,hashlib,fcntl,subprocess,copy,time,traceback
P=Path(__file__).resolve().parent;R=P.parent
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
state=json.loads((P/'status.json').read_text());assert state['status']=='complete_pending_actual_ports_and_board' and len(state['cases'])==36
sdk='/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator';ins=R/'graphs/candidates-w16-storage/inspect_nbg'
audit={'status':'running','started':time.time(),'scope':'Capacity-only1280 to512 actual16-port check and payload preservation; no board performance or quality claim','inspector_sha256':sha(ins),'cases':[]}
def save():(P/'port-audit.json').write_text(json.dumps(audit,indent=2))
try:
 with open('/tmp/opi-prefill-compile.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for case in state['cases']:
   assert case['returncode']==0 and case['payload_exact_source'] and case['quantizer_exact_source']
   source=Path(case['source']);dest=Path(case['output']);ports={}
   assert sha(dest/'export_nbg_unify/network_binary.nb')==case['nb_sha256']
   assert sha(dest/'export/decoder.export.data')==case['payload_sha256']==sha(source/'export/decoder.export.data')
   assert sha(dest/'decoder.quantize')==sha(source/'decoder.quantize')
   for name,folder in [('source',source),('candidate',dest)]:
    nb=folder/'export_nbg_unify/network_binary.nb'
    cmd=['docker','run','--rm','--network','none','--cpus','2','--memory','8g','-v',str(R)+':'+str(R),'-v',str(source)+':'+str(source)+':ro','-e','LD_LIBRARY_PATH='+sdk+'/lib:'+sdk+'/../common/lib','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-e','VSIMULATOR_SHADER_CORE_COUNT=1','--entrypoint',str(ins),'ubuntu-npu:v2.0.10.2',str(nb)]
    z=subprocess.run(cmd,capture_output=True,text=True,timeout=30);assert z.returncode==0,(z.returncode,z.stdout,z.stderr)
    ports[name]=json.loads(z.stdout);(dest/(name+'-actual-ports.json')).write_text(z.stdout)
   expected=copy.deepcopy(ports['source']);assert expected['count']==16
   for index in range(1,5):
    dims=expected['ports'][index]['dims'];assert dims==[64,1280,2,1];dims[1]=512
   for index in [5,6]:
    dims=expected['ports'][index]['dims'];assert dims[0]==1280+case['query_tokens'];dims[0]=512+case['query_tokens']
   assert expected==ports['candidate'],(case['kind'],case['first_layer'],'unexpected external port change')
   audit['cases'].append(dict(kind=case['kind'],first_layer=case['first_layer'],query_tokens=case['query_tokens'],nb_sha256=case['nb_sha256'],actual_port_capacity_only=True,payload_unchanged=True,quantizer_unchanged=True));save();print(case['kind'],case['first_layer'],flush=True)
 audit.update(status='complete_actual_ports_not_board_qualified',ended=time.time());save()
except BaseException:audit.update(status='failed',error=traceback.format_exc());save();raise
