from pathlib import Path
import subprocess
r=Path(__file__).resolve().parent;s=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/cpu-tools/source');b=r/'build-arm64'
with (r/'results/build-jev.log').open('w') as log:
 def run(args):subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
 run(['cmake','-S',str(s),'-B',str(b),'-DCMAKE_SYSTEM_NAME=Linux','-DCMAKE_SYSTEM_PROCESSOR=aarch64','-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc','-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++','-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=ON','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DGGML_NATIVE=OFF','-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod','-DLLAMA_CURL=OFF'])
 run(['cmake','--build',str(b),'-j','4','--target','llama'])
 run(['aarch64-linux-gnu-g++','-std=c++17','-O2','-I'+str(s/'include'),'-I'+str(s/'ggml/include'),'-I'+str(s/'vendor'),str(r/'sources/jev_score.cpp'),'-o',str(b/'bin/jev-score'),'-L'+str(b/'bin'),'-lllama','-lggml','-lggml-base','-Wl,-rpath,$ORIGIN'])
print('build complete',flush=True)
