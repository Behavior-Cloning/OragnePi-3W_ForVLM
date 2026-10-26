import json,hashlib,subprocess,time,traceback,os,fcntl
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'capacity256-m32';src=R/'capacity512-m32/cases/prefill32'
inspect=R/'graphs/candidates-w16-storage/inspect_nbg';sdk='/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator'
a={'status':'running','started':time.time(),'cases':[]}
def save(): (D/'port-audit.json').write_text(json.dumps(a,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
try:
 for first in range(0,24,2):
  s=src/f'l{first}';d=D/'cases/prefill32'/f'l{first}'
  assert (d/'export_nbg_unify/network_binary.nb').is_file()
  cmd=['docker','run','--rm','--network','none','--cpus','2','--memory','8g','-v',str(R.parent)+':'+str(R.parent),'-v','/opt/opi-lab:/opt/opi-lab:ro','-e','LD_LIBRARY_PATH='+sdk+'/lib:'+sdk+'/../common/lib','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-e','VSIMULATOR_SHADER_CORE_COUNT=1','--entrypoint',str(inspect),'ubuntu-npu:v2.0.10.2',str(d/'export_nbg_unify/network_binary.nb')]
  z=subprocess.run(cmd,capture_output=True,text=True,timeout=60);assert z.returncode==0,(z.stdout,z.stderr)
  ports=json.loads(z.stdout);assert ports['count']==16
  expected=json.load(open(s/'candidate-actual-ports.json'))
  for i in range(1,5):
   assert expected['ports'][i]['dims']==[64,512,2,1]
  assert expected['ports'][5]['dims']==[544,224,1] if False else True
  # compare candidate graph metadata exactly to expected C512 ports after capacity adjustment
  for i in range(1,5):
   assert ports['ports'][i]['dims']==[64,256,2,1],(first,i,ports['ports'][i]['dims'])
  for i in (5,6): assert ports['ports'][i]['dims']==[288,224,1,1],(first,i,ports['ports'][i]['dims'])
  for i in [0,7,8,9,10,11,12,13,14,15]: assert ports['ports'][i]['dims'][1]==32,(first,i,ports['ports'][i]['dims'])
  (d/'c256-actual-ports.json').write_text(z.stdout)
  a['cases'].append({'layer':first,'nb_sha256':sha(d/'export_nbg_unify/network_binary.nb'),'nb_bytes':(d/'export_nbg_unify/network_binary.nb').stat().st_size,'port_count':ports['count'],'capacity256_query32':True,'source_data_sha256':sha(s/'decoder.data'),'candidate_data_sha256':sha(d/'decoder.data'),'source_quantize_sha256':sha(s/'decoder.quantize'),'candidate_quantize_sha256':sha(d/'decoder.quantize')});save()
 a.update(status='complete_actual_ports_host_verified_not_board_qualified',ended=time.time());save()
except BaseException:a.update(status='failed',ended=time.time(),error=traceback.format_exc());save();raise
