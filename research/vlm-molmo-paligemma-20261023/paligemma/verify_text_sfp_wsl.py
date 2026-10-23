import pathlib,json,subprocess,re,time
P=pathlib.Path(__file__).parent;cases=json.loads((P/'text-only-cases.json').read_text())
cmd=[str(P/'build-x86/gemma'),'--weights',str(P/'paligemma2-3b-mix-224-sfp.sbs'),'--to_bf16','0','--map','1','--num_threads','2','--seq_len','512','--max_generated_tokens','16','--deterministic','1','--temperature','0','--top_k','1','--verbosity','2','--multiturn','0']
r=subprocess.run(cmd,input='\n'.join(c['prompt'] for c in cases)+'\n%q\n',text=True,capture_output=True,timeout=240);(P/'pali-text-sfp-wsl.stdout.log').write_text(r.stdout);(P/'pali-text-sfp-wsl.stderr.log').write_text(r.stderr)
parts=r.stderr.split('OPI_REQUEST_BEGIN')[1:];out=[]
for c,part in zip(cases,parts):out.append(dict(case=c['id'],output_ids=[int(x) for x in re.findall(r'OPI_TOKEN id=(\d+)',part)],prompt_tokens=int(re.search('prompt_tokens=(\d+)',part)[1])))
(P/'pali-text-sfp-wsl-results.json').write_text(json.dumps(dict(exit=r.returncode,cases=out,image_events=r.stderr.count('OPI_IMAGE_BEGIN'),command=cmd),indent=2));print(out,flush=True);assert r.returncode==0 and len(out)==3 and 'OPI_IMAGE_BEGIN' not in r.stderr
