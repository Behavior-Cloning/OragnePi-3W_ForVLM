import pathlib,subprocess,time,sys,json,hashlib,shlex
P=pathlib.Path(__file__).parent;sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts');import board_access as b
B='/opt/opi-vlm/vlm-new-models-20261023'
while pathlib.Path('/proc/8587').exists():time.sleep(10)
assert (P/'sfp-board-manifest.json').exists(),'transfer failed'
assert json.loads((P/'sfp-board-manifest.json').read_text())['sha256']==json.loads((P/'sfp-manifest.json').read_text())['sha256']
while pathlib.Path('/proc/8604').exists():time.sleep(5)
assert 'BOTH_INSTRUMENTED_BUILDS_READY' in (P/'build-instrumented-runtime.log').read_text(),'build failed'
assert (P/'instrumented-wsl-parity.json').exists(),'instrumented WSL parity must complete first'
assert json.loads((P/'instrumented-wsl-parity.json').read_text())['passed']
with b.hostlock():
 for src,dest in [(P/'build-arm/gemma','gemma-pali'),(P/'gemma.cpp-main/paligemma/testdata/image.ppm','image.ppm'),(P/'pali_board_runner.py','pali_board_runner.py')]:b.scp(str(src),b.HOST+':'+B+'/'+dest,timeout=120).check_returncode()
 b.ssh('chmod +x '+B+'/gemma-pali',timeout=15).check_returncode()
 print('board starting',time.time(),flush=True)
 r=b.ssh('flock -n -E 73 /run/opi-npu-kv-benchmark.lock timeout -k 20 1150 python3 '+B+'/pali_board_runner.py',timeout=1180);print('board exit',r.returncode,flush=True)
 for n in ['pali-board-results.jsonl','pali-board-telemetry.json','pali-board-stderr.log','pali-board-stdout.log']:b.scp(b.HOST+':'+B+'/'+n,str(P/n),timeout=60)
 (P/'pali-board-controller-exit.json').write_text(json.dumps(dict(exit=r.returncode)))
