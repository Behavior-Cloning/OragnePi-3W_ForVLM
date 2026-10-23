import pathlib,sys,subprocess,json,hashlib,shlex
P=pathlib.Path(__file__).parent;sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
B='/opt/opi-vlm/vlm-new-models-20261023';manifest=[]
with b.hostlock():
 for src,name in [(P/'build-arm/gemma','gemma-pali-text'),(P/'pali_text_board_runner.py','pali_text_board_runner.py'),(P/'text-only-cases.json','text-only-cases.json')]:
  sha=hashlib.sha256(src.read_bytes()).hexdigest();b.scp(str(src),b.HOST+':'+B+'/'+name,timeout=60).check_returncode();r=b.ssh('sha256sum '+B+'/'+name,timeout=20,capture_output=True,text=True);r.check_returncode();assert r.stdout.split()[0]==sha;manifest.append(dict(source=str(src),board_name=name,sha256=sha,bytes=src.stat().st_size))
 (P/'pali-text-board-source-manifest.json').write_text(json.dumps(manifest,indent=2));b.ssh('chmod +x '+B+'/gemma-pali-text',timeout=10).check_returncode()
 r=b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock timeout -k 20 500 python3 '+B+'/pali_text_board_runner.py',timeout=530);print('text board exit',r.returncode,flush=True)
 for name in ['pali-text-board-results.jsonl','pali-text-board-telemetry.json','pali-text-board-stderr.log','pali-text-board-stdout.log']:b.scp(b.HOST+':'+B+'/'+name,str(P/name),timeout=60).check_returncode()
 (P/'pali-text-board-controller-exit.json').write_text(json.dumps(dict(exit=r.returncode)))
r=subprocess.run(['python3',str(P/'summarize_pali_text_board.py')]);print('text summary exit',r.returncode,flush=True)
