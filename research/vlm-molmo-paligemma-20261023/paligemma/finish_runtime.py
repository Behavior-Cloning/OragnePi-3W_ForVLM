import pathlib,subprocess,json,os,time,shutil
P=pathlib.Path(__file__).parent
# Wait for the initial x86 build to finish; no simultaneous mutation of the same build tree.
while pathlib.Path('/proc/2654').exists():time.sleep(5)
args=['-DFETCHCONTENT_SOURCE_DIR_'+x['name'].upper()+'='+x['source'] for x in json.loads((P/'deps-manifest.json').read_text())]
common=['-DCMAKE_BUILD_TYPE=Release','-DSPM_ENABLE_SHARED=OFF','-DSPM_ABSL_PROVIDER=internal','-DHWY_ENABLE_TESTS=OFF','-DHWY_ENABLE_EXAMPLES=OFF','-DHWY_ENABLE_CONTRIB=ON','-DGEMMA_ENABLE_TESTS=OFF','-DCMAKE_POSITION_INDEPENDENT_CODE=ON']+args
r=subprocess.run(['cmake','-S',str(P/'gemma.cpp-main'),'-B',str(P/'build-x86')]+common+['-DGEMMA_BUILD_TASK_PYTHON=ON','-DTASK_PYTHON=/opt/opi-lab/export-venv/bin/python'])
if r.returncode==0:r=subprocess.run(['cmake','--build',str(P/'build-x86'),'--target','configs','compression','-j1'])
print('python extensions',r.returncode,flush=True)
r=subprocess.run(['cmake','-S',str(P/'gemma.cpp-main'),'-B',str(P/'build-arm')]+common+['-DCMAKE_TOOLCHAIN_FILE='+str(P/'arm-toolchain.cmake'),'-DCMAKE_CXX_FLAGS=-march=armv8.2-a+dotprod -DHWY_COMPILE_ONLY_STATIC=1','-DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=TRUE'])
if r.returncode==0:r=subprocess.run(['cmake','--build',str(P/'build-arm'),'--target','gemma','-j1'])
print('arm static NEON',r.returncode,flush=True)
