import pathlib,subprocess
P=pathlib.Path(__file__).parent
for build in ['build-x86','build-arm']:
 r=subprocess.run(['cmake','--build',str(P/build),'--target','gemma','-j','2']);r.check_returncode()
print('BOTH_INSTRUMENTED_BUILDS_READY',flush=True)
