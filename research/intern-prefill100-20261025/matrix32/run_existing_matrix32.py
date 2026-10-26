#!/usr/bin/env python3
"""Run matrix32 using the already staged C512 banks through both locks."""
from pathlib import Path
import json,hashlib,shlex,subprocess,sys,time,tarfile,traceback
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts'); import board_access as b
D=Path(__file__).resolve().parent; sys.path.insert(0,str(D)); T=D.parent; RUN='matrix32-r10'; OLD='/opt/opi-vlm/intern-prefill100-20261025/capacity512-full/runs/c512-m16-m32-aba-r1'; REMOTE='/opt/opi-vlm/intern-prefill100-20261025/capacity512-full/runs/'+RUN; OUT=REMOTE+'/result'; status=D/(RUN+'-controller.json'); log_local=D/(RUN+'-controller.log')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def locked(cmd,**kw): return b.ssh('flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd),**kw)
def put(path,data):
    cmd='cat > '+shlex.quote(str(path))+'.part && mv '+shlex.quote(str(path))+'.part '+shlex.quote(str(path))
    r=locked(cmd,input=data,timeout=60,check=True); return r
def save(st): status.write_text(json.dumps(st,ensure_ascii=False,indent=2)+'\n')
def main():
    st={'run':RUN,'status':'starting','remote':REMOTE,'started':time.time(),'scope':'static32/64 x dynamic32/64/128/256 performance+diagnostic; reuse system only'};save(st)
    try:
      from contract import freeze
      raw=json.loads((D/'config.fulli16.json').read_text());raw['artifacts']=json.loads((D/'artifacts.json').read_text());raw['expected_fixture_sha256']=sha(D/'inputs.json')
      cfg=freeze(raw,REMOTE);cfg['fixture_path']=REMOTE+'/inputs.json';cfg['rag_catalog']=REMOTE+'/atoms.json';cfg['quality']=True;cfg['diagnostic']=True
      files={n:D/n for n in ['board_fulli16.py','resident_worker_v1.py','context_rag.py','contract.py','inputs.json']}
      cfg_path=D/'config.json';cfg_path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n');files['config.json']=cfg_path
      with b.hostlock():
        # New directory only contains small harness/fixtures and symlinks to existing verified banks.
        locked('rm -rf '+shlex.quote(REMOTE)+' && mkdir -p '+shlex.quote(REMOTE))
        for name,target in [('m16',OLD+'/m16'),('m32',OLD+'/m32'),('m8',OLD+'/m8'),('resident-cap512',OLD+'/resident-cap512'),('atoms.json',OLD+'/atoms.json')]:
          locked('ln -s '+shlex.quote(target)+' '+shlex.quote(REMOTE+'/'+name))
        for n,p in files.items(): put(REMOTE+'/'+n,p.read_bytes())
        # Verify remote SHA and key links while both locks are held.
        for n,p in files.items():
          q=locked('sha256sum '+shlex.quote(REMOTE+'/'+n),capture_output=True,text=True,check=True)
          assert q.stdout.split()[0]==sha(p),n
        st.update(status='running',source_sha256={n:sha(p) for n,p in files.items()},fixture_sha256=sha(D/'inputs.json'));save(st)
        cmd='timeout -k 15 3600 /opt/opi-vlm/jev-lab-20261020/venv/bin/python -u '+shlex.quote(REMOTE+'/board_fulli16.py')+' --config '+shlex.quote(REMOTE+'/config.json')+' --out '+shlex.quote(OUT)+' --repeats 1 --phases performance,quality,diagnostic >'+shlex.quote(REMOTE+'/process.log')+' 2>&1; rc=$?; printf "%s" "$rc" > '+shlex.quote(REMOTE+'/exit')
        b.ssh('nohup sh -c '+shlex.quote('flock -w 5 /run/opi-npu-kv-benchmark.lock sh -c '+shlex.quote(cmd))+' </dev/null >/dev/null 2>&1 &',check=True,timeout=20)
        deadline=time.monotonic()+3900
        while True:
          time.sleep(15); q=b.ssh('cat '+shlex.quote(REMOTE+'/exit'),capture_output=True,text=True,timeout=20)
          if q.returncode==0: st['returncode']=int(q.stdout.strip()); break
          if time.monotonic()>deadline: raise TimeoutError('matrix32 board timeout')
        packet=locked('tar -czf - -C '+shlex.quote(OUT)+' .',capture_output=True,timeout=240,check=True).stdout
        dest=D/'results'/RUN;dest.mkdir(parents=True,exist_ok=False)
        arc=D/(RUN+'.tar.gz');arc.write_bytes(packet)
        with tarfile.open(arc) as tar: tar.extractall(dest)
        st.update(status='complete' if st['returncode']==0 else 'failed',ended=time.time(),archive=str(arc),result=str(dest));save(st)
    except BaseException:
      st.update(status='failed',error=traceback.format_exc(),ended=time.time());save(st);raise
if __name__=='__main__':main()
