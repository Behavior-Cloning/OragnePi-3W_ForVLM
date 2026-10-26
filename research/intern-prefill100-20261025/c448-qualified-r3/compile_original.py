from pathlib import Path
import json,subprocess,shutil,hashlib,time,yaml
R=Path(__file__).resolve().parent;parent=R.parent
state={'status':'running','pid':__import__('os').getpid(),'cases':[]}
def save():(R/'compile-status.json').write_text(json.dumps(state,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
save()
for n,src in [(32,parent/'capacity512-m32/cases/original32/l2'),(8,parent/'capacity512-original-l2/cases/prefill8/l2')]:
 dst=R/('original'+str(n));dst.mkdir(exist_ok=False)
 g=json.loads((src/'decoder.json').read_text())
 for i in range(2):
  for node,field,index,before,after in [(f'b{i}_mask_104','shape',3,512+n,448+n),(f'b{i}_past_k_102','shape',2,512,448),(f'b{i}_past_v_103','shape',2,512,448),(f'b{i}_split_probs_new_11005','begin',3,512,448),(f'b{i}_split_probs_past_11004','size',3,512,448)]:
   arr=g['Layers'][node]['parameters'][field];assert arr[index]==before;arr[index]=after
 (dst/'decoder.json').write_text(json.dumps(g,indent=2));(dst/'decoder.data').symlink_to((src/'decoder.data').resolve());shutil.copy2(src/'decoder.quantize',dst/'decoder.quantize')
 meta=yaml.safe_load((src/'tensor-inputmeta.yml').read_text())
 for port in meta['input_meta']['databases'][0]['ports']:port['shape']=g['Layers'][port['lid']]['parameters']['shape']
 (dst/'tensor-inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False))
 cmd=['docker','run','--rm','--name','opi-c448-original-'+str(n),'--network','none','--cpus','4','--memory','8g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=4','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(parent.parent)+':'+str(parent.parent),'-v','/opt/opi-lab:/opt/opi-lab:ro','-w',str(dst),'ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
 row={'rows':n,'source':str(src),'data_sha256':sha(src/'decoder.data'),'quantizer_sha256':sha(src/'decoder.quantize')};state['cases'].append(row);save()
 with (dst/'compile.log').open('w') as log:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=500)
 row['rc']=z.returncode;save();assert z.returncode==0
 nb=dst/'export_nbg_unify/network_binary.nb';row.update(nb_sha256=sha(nb),nb_bytes=nb.stat().st_size);save()
state['status']='complete';save()
