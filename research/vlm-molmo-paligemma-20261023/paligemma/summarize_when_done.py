import pathlib,time,subprocess
P=pathlib.Path(__file__).parent
while pathlib.Path('/proc/9835').exists():time.sleep(10)
if (P/'pali-board-results.jsonl').exists():
 r=subprocess.run(['python3',str(P/'summarize_pali_board.py')]);print('summary_exit',r.returncode,flush=True)
else:print('No board result; inspect run-pali-board.log',flush=True)
