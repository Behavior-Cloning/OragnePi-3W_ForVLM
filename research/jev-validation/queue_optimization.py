import os,time,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
while Path('/proc/4805').exists():time.sleep(10)
subprocess.run(['python3','-u',str(r/'run_board_optimization.py')],check=True)
