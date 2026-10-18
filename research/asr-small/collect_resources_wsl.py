import json,shutil,subprocess,time
from pathlib import Path
repo=Path(__file__).resolve().parents[2];assets=Path('/mnt/f/Files/OPi-Lab/asr-small-models-20261018')
mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()}
ps='Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress'
r=subprocess.run(['powershell.exe','-NoProfile','-Command',ps],capture_output=True,text=True,check=True,stdin=subprocess.DEVNULL)
def size(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
result={'timestamp':time.time(),'F_filesystem':shutil.disk_usage('/mnt/f')._asdict(),'WSL_memory_kib':mem,'Windows_memory_kib':json.loads(r.stdout),'ASR_assets_bytes':size(assets),'ASR_models_bytes':{p.name:size(p) for p in (assets/'models').iterdir() if p.is_dir()},'ASR_result_bytes':size(repo/'results/asr-small-20261018'),'ASR_source_bytes':size(repo/'research/asr-small')}
(repo/'results/asr-small-20261018/windows-resources.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
