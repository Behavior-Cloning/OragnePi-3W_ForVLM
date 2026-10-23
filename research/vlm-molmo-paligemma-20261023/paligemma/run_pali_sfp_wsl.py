import pathlib,subprocess,time,json
P=pathlib.Path(__file__).parent
for cid,prompt in [('building','What type of building is it?'),('color','What color is the church?')]:
 cmd=[str(P/'build-x86/gemma'),'--weights',str(P/'paligemma2-3b-mix-224-sfp.sbs'),'--to_bf16','0','--num_threads','2','--seq_len','512','--max_generated_tokens','16','--deterministic','1','--temperature','0','--top_k','1','--image_file',str(P/'gemma.cpp-main/paligemma/testdata/image.ppm'),'--prompt',prompt,'--verbosity','2']
 t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,timeout=240);(P/('pali-sfp-wsl-'+cid+'.log')).write_text(r.stdout+'\nSTDERR\n'+r.stderr);print(json.dumps(dict(case=cid,exit=r.returncode,wall_s=time.monotonic()-t,command=cmd)),flush=True)
