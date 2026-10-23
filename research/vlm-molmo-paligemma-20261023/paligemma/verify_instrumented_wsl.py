import pathlib,subprocess,time,json,re
P=pathlib.Path(__file__).parent
cmd=[str(P/'build-x86/gemma'),'--weights',str(P/'paligemma2-3b-mix-224-sfp.sbs'),'--to_bf16','0','--map','1','--num_threads','2','--seq_len','512','--max_generated_tokens','16','--deterministic','1','--temperature','0','--top_k','1','--image_file',str(P/'gemma.cpp-main/paligemma/testdata/image.ppm'),'--verbosity','2','--multiturn','0']
t=time.monotonic();r=subprocess.run(cmd,input='What type of building is it?\nWhat color is the church?\n%q\n',text=True,capture_output=True,timeout=240)
(P/'pali-instrumented-wsl.stdout.log').write_text(r.stdout);(P/'pali-instrumented-wsl.stderr.log').write_text(r.stderr)
tokens=[int(x) for x in re.findall(r'OPI_TOKEN id=(\d+)',r.stderr)];d=dict(exit=r.returncode,wall_s=time.monotonic()-t,tokens=tokens,reference_tokens=[44201,1,7384,1],image_runs=r.stderr.count('OPI_IMAGE_BEGIN'),requests=r.stderr.count('OPI_REQUEST_END'),command=cmd);d['passed']=r.returncode==0 and tokens==d['reference_tokens'] and d['image_runs']==2 and d['requests']==2
(P/'instrumented-wsl-parity.json').write_text(json.dumps(d,indent=2));print(d,flush=True);assert d['passed']
