"""Bounded all-FP16 control; same imported layer0 and frozen inputs, no board calls."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time,yaml
r=Path(__file__).resolve().parent
root=Path('/opt/opi-lab');base=root/'llama32-compat-20261023/compiled';out=root/'llama32-compat-20261023/all-fp16'
assert shutil.disk_usage('/mnt/f').free>8*1024**3
out.mkdir(exist_ok=True)
for n in ('decoder.data','decoder.json','tensor-inputmeta.yml','dataset.txt'):
    if not (out/n).exists():os.link(base/n,out/n)
    assert os.path.samefile(base/n,out/n)
q=yaml.safe_load((base/'decoder.quantize').read_text())
q['quantize_parameters']={k:{'qtype':'float16','quantizer':'float16'} for k in q['quantize_parameters'] if not k.endswith(':bias')}
(out/'decoder.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
name='llama32-allfp16-control'
cmd=['docker','run','--rm','--name',name,'--network','none','--cpus','2','--memory','5g','-e','TF_CPP_MIN_LOG_LEVEL=3','-e','OMP_NUM_THREADS=2','-e','VSIMULATOR_SHADER_CORE_COUNT=1','-e','VSIMULATOR_CONFIG=VIP9000NANODI_PLUS_PID0X1000003B','-v',str(root)+':/work','-w','/work/llama32-compat-20261023/all-fp16','ubuntu-npu:v2.0.10.2','python3','/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py','export','ovxlib','--model','decoder.json','--model-data','decoder.data','--model-quantize','decoder.quantize','--dtype','quantized','--with-input-meta','tensor-inputmeta.yml','--target-ide-project','linux64','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000003B','--viv-sdk','/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools','--output-path','export/decoder']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
t=time.monotonic();report={'scope':'all-FP16 real layer0 control from dequantized Q4_K_M source; no BF16 full-model claim','source_sha256':{n:sha(base/n) for n in ('decoder.json','decoder.data','decoder.quantize')},'source_base':str(base),'command':cmd,'all_fp16_quantize_entries':len(q['quantize_parameters'])}
with (out/'export.log').open('w') as log:
    try:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300);code=z.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','stop','-t','2',name],stdout=log,stderr=subprocess.STDOUT,timeout=15);code=124
report.update(exit_code=code,seconds=time.monotonic()-t)
files=list(out.rglob('network_binary.nb'));report['binaries']=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in files]
# Shared decoder.data is not new allocation. Account by source inode.
base_ids={(f.stat().st_dev,f.stat().st_ino) for f in base.rglob('*') if f.is_file()};seen=set();new=0
for f in out.rglob('*'):
    if f.is_file():
        st=f.stat();ident=(st.st_dev,st.st_ino)
        if ident not in base_ids and ident not in seen:new+=st.st_size;seen.add(ident)
report['new_bytes_excluding_hardlinks']=new
report['F_free_after']=shutil.disk_usage('/mnt/f').free
(out/'conversion.json').write_text(json.dumps(report,indent=2)+'\n')
(r/'all-fp16-conversion.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copyfile(out/'export.log',r/'all-fp16-compile.log')
print(json.dumps(report,indent=2),flush=True)
assert new<=300*1024**2, 'artifact budget exceeded'
raise SystemExit(code)
