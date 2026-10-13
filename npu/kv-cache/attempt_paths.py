"""Select retry paths without overwriting failed or partially uploaded evidence."""
import json
from pathlib import Path


def select_attempt(base, marker=None, log=None, retry=False):
    base=Path(base)
    for attempt in range(1,1000):
        suffix='' if attempt==1 else f'-attempt{attempt}'
        path=(base.with_name(base.stem+suffix+base.suffix) if marker is None
              else base.with_name(base.name+suffix))
        result=path if marker is None else path/marker
        log_path=Path(log).with_name(Path(log).stem+suffix+Path(log).suffix) if log else None
        if result.exists():
            try:data=json.loads(result.read_text())
            except (ValueError,OSError):data={}
            if data.get('status')=='complete':return path,log_path,attempt
            if not retry:return path,log_path,attempt
        if not path.exists() and not (log_path and log_path.exists()):return path,log_path,attempt
        if not retry:return path,log_path,attempt
    raise RuntimeError('too many attempts: '+str(base))
