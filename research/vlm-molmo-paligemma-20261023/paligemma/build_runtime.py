import pathlib,urllib.request,tarfile,json,hashlib,subprocess,time
P=pathlib.Path(__file__).parent
D=P/'deps';D.mkdir(exist_ok=True)
deps={'highway':('google/highway','2a16a50ff61071bb25ddef0ce35d92b0e2b9c579'),'sentencepiece':('google/sentencepiece','53de76561cfc149d3c01037f0595669ad32a5e7c'),'json':('nlohmann/json','9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03'),'httplib':('yhirose/cpp-httplib','v0.18.1'),'benchmark':('google/benchmark','v1.8.2')}
manifest=[]
for name,(repo,rev) in deps.items():
 u='https://codeload.github.com/'+repo+'/tar.gz/'+rev; f=D/(name+'.tar.gz')
 if not f.exists():
  for attempt in range(3):
   try:f.write_bytes(urllib.request.urlopen(u,timeout=60).read());break
   except Exception as e:print(name,attempt,str(e),flush=True);time.sleep(2)
 if not f.exists():raise RuntimeError('dependency unavailable: '+name)
 with tarfile.open(f) as t:
  folder=t.getmembers()[0].name.split('/')[0];t.extractall(D,members=[m for m in t.getmembers() if not m.issym() and not m.islnk()],filter='data')
 manifest.append(dict(name=name,repo=repo,revision=rev,url=u,size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),source=str(D/folder)))
 print('dependency ready',name,f.stat().st_size,flush=True)
(P/'deps-manifest.json').write_text(json.dumps(manifest,indent=2))
args=['-DFETCHCONTENT_SOURCE_DIR_'+x['name'].upper()+'='+x['source'] for x in manifest]
args += ['-DCMAKE_BUILD_TYPE=Release','-DSPM_ENABLE_SHARED=OFF','-DSPM_ABSL_PROVIDER=internal','-DHWY_ENABLE_TESTS=OFF','-DHWY_ENABLE_EXAMPLES=OFF','-DHWY_ENABLE_CONTRIB=ON','-DSPM_ENABLE_NFKC_COMPILE=OFF','-DBENCHMARK_ENABLE_TESTING=OFF','-DGEMMA_ENABLE_TESTS=OFF']
(P/'arm-toolchain.cmake').write_text('set(CMAKE_SYSTEM_NAME Linux)\nset(CMAKE_SYSTEM_PROCESSOR aarch64)\nset(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)\nset(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)\nset(CMAKE_CXX_FLAGS_INIT "-march=armv8.2-a+dotprod")\n')
for arch,extra in [('arm',['-DCMAKE_TOOLCHAIN_FILE='+str(P/'arm-toolchain.cmake'),'-DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=TRUE']),('x86',[])]:
 cmd=['cmake','-S',str(P/'gemma.cpp-main'),'-B',str(P/('build-'+arch))]+args+extra
 print('configure',arch,flush=True);r=subprocess.run(cmd)
 if r.returncode:print('configure failed',arch,r.returncode,flush=True);continue
 r=subprocess.run(['cmake','--build',str(P/('build-'+arch)),'--target','gemma','-j1'])
 print('build return',arch,r.returncode,flush=True)
