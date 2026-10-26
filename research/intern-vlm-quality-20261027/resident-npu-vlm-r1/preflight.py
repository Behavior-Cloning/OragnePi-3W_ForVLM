from pathlib import Path
import subprocess,re,json
# Observed driver allocations, full FP32 embedding mapping and 256MiB reserve.
VISION_BYTES=707619272
LANGUAGE_BYTES=1334594552
EMBEDDING_BYTES=151674*896*4
RESERVE_BYTES=256*1024*1024
REQUIRED_BYTES=VISION_BYTES+LANGUAGE_BYTES+EMBEDDING_BYTES+RESERVE_BYTES
def evaluate(meminfo,service_active,driver_bytes):
 values={m.group(1):int(m.group(2))*1024 for m in re.finditer(r'^(\w+):\s+(\d+) kB$',meminfo,re.M)}
 reasons=[]
 if service_active!='inactive':reasons.append('background CPU model not confirmed inactive')
 if driver_bytes!=0:reasons.append('NPU not confirmed idle')
 if values.get('MemAvailable',0)<REQUIRED_BYTES:reasons.append('insufficient available memory for measured models plus embedding/reserve')
 return {'eligible':not reasons,'reasons':reasons,'available_bytes':values.get('MemAvailable'),'required_bytes':REQUIRED_BYTES,'swap_free_bytes':values.get('SwapFree'),'background_service':service_active,'driver_bytes':driver_bytes}
def collect():
 z=subprocess.run(['systemctl','show','opi-vlm-model.service','-p','ActiveState','--value'],capture_output=True,text=True)
 active=z.stdout.strip() if z.returncode==0 else 'unknown'
 try:
  profile=Path('/sys/kernel/debug/viplite/mem_profile').read_text()
  rows=re.findall(r'\|\s*(0x[0-9a-fA-F]+)\s*\|',profile);driver=int(rows[0],16) if rows else None
 except OSError:driver=None
 return evaluate(Path('/proc/meminfo').read_text(),active,driver)
if __name__=='__main__':print(json.dumps(collect(),indent=2))
