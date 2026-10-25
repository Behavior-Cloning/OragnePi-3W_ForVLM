from pathlib import Path
import hashlib,json,subprocess,shutil,time,uuid,traceback,fcntl,os
R=Path(__file__).resolve().parent;OUT=R/'trace-export-m32';OUT.mkdir(exist_ok=True)
SOURCE=Path('/opt/opi-lab/source-validation-20261017/gguf-fresh/export')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(): (OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
state={'state':'waiting_sdk_lock','start':time.time(),'cases':[],'pid':os.getpid()};save()
compile_lock=open('/tmp/opi-prefill-compile.lock','a+');fcntl.flock(compile_lock,fcntl.LOCK_EX)
SDK='sha256:b2447e88c70d334cbadf91d1cfef2874d6a4360f0e2b075606df32547ec8cbe3'
assert subprocess.check_output(['docker','image','inspect','--format','{{.Id}}','ubuntu-npu:v2.0.10.2'],text=True).strip()==SDK
state.update(state='running',sdk_image=SDK);save()
try:
 for first in [0,2]:
  src=R/'cases'/f'c512-m32-l{first}';dst=OUT/f'c512-m32-l{first}';dst.mkdir(exist_ok=False)
  for name in ['decoder.json','decoder.quantize','tensor-inputmeta.yml']:shutil.copy2(src/name,dst/name)
  (dst/'decoder.data').symlink_to((src/'decoder.data').resolve())
  cmd=json.loads((src/'conversion.json').read_text())['compile_command'];cmd=[x.replace(str(src),str(dst)) for x in cmd];cmd[cmd.index('--name')+1]='opi-trace-'+uuid.uuid4().hex[:10];cmd[3:3]=['-e','VIV_VX_DUMP_NODE_TRACE_DB=1']
  cmd[cmd.index('ubuntu-npu:v2.0.10.2')]=SDK
  frozen={name:sha(src/name) for name in ['decoder.json','decoder.quantize','tensor-inputmeta.yml','decoder.data','export/decoder.export.data','export_nbg_unify/network_binary.nb']}
  t=time.monotonic()
  with (dst/'export.log').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
  assert p.returncode==0
  assert all(sha(src/name)==digest for name,digest in frozen.items())
  nb=dst/'export_nbg_unify/network_binary.nb';trace=list(dst.rglob('*trace*'))
  row={'first_layer':first,'source_sha256':frozen,'query_rows':32,'seconds':time.monotonic()-t,'command':cmd,'nb_sha256':sha(nb),'baseline_nb_sha256':sha(src/'export_nbg_unify/network_binary.nb'),'nb_byte_equal':nb.read_bytes()==(src/'export_nbg_unify/network_binary.nb').read_bytes(),'trace_files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in trace if p.is_file()]}
  assert sha(dst/'export/decoder.export.data')==sha(src/'export/decoder.export.data');(dst/'export/decoder.export.data').unlink();(dst/'export/decoder.export.data').symlink_to((src/'export/decoder.export.data').resolve())
  state['cases'].append(row);save();print(row,flush=True)
  assert row['nb_byte_equal'],'Trace NBG changed; never map existing NPD to a different NBG'
 state.update(state='complete',end=time.time());save()
except BaseException:state.update(state='failed',error=traceback.format_exc(),end=time.time());save();raise
