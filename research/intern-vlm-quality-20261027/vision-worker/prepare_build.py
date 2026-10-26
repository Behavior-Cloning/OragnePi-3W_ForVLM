from pathlib import Path
import shutil,hashlib,json,subprocess,time,traceback
D=Path(__file__).resolve().parent
S=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/cpu-tools/source')
W=Path('/opt/opi-lab/builds/intern-vision-20261027-r1');W.mkdir(parents=True,exist_ok=True)
status={'started':time.time(),'state':'preparing','source_provenance':'restored source tree; parent Git revision is NOT a llama.cpp revision'}
def save(): (D/'build-status.json').write_text(json.dumps(status,indent=2))
try:
 save();dst=W/'source'
 assert not dst.exists(),'build directory must be fresh'
 shutil.copytree(S,dst,ignore=shutil.ignore_patterns('.git','build','build-*','__pycache__'))
 target=dst/'tools/mtmd/models/internvl.cpp';orig=target.read_text()
 patch=r"""    // GGUF stores CLS last. Resize only the patch grid using the official
    // bicubic/align_corners=false convention; leave the 448 graph unchanged.
    ggml_tensor * pos = model.position_embeddings;
    const int old_grid = (int)std::sqrt(pos->ne[1] - 1);
    GGML_ASSERT(old_grid * old_grid + 1 == pos->ne[1]);
    GGML_ASSERT(n_patches_x == n_patches_y && n_patches_x % 2 == 0);
    if (n_patches_x != old_grid) {
        auto * cls_pos = ggml_view_2d(ctx0, pos, n_embd, 1, pos->nb[1], old_grid * old_grid * pos->nb[1]);
        auto * grid = ggml_view_2d(ctx0, pos, n_embd, old_grid * old_grid, pos->nb[1], 0);
        grid = ggml_cont(ctx0, grid);
        grid = ggml_reshape_3d(ctx0, grid, n_embd, old_grid, old_grid);
        grid = ggml_permute(ctx0, grid, 2, 0, 1, 3);
        grid = ggml_interpolate(ctx0, grid, n_patches_x, n_patches_y, n_embd, 1, GGML_SCALE_MODE_BICUBIC);
        grid = ggml_permute(ctx0, grid, 1, 2, 0, 3);
        grid = ggml_cont_2d(ctx0, grid, n_embd, n_patches);
        pos = ggml_concat(ctx0, grid, cls_pos, 1);
    }
"""
 assert orig.count('    ggml_tensor * cur = build_vit(')==1
 modified=orig.replace('    ggml_tensor * cur = build_vit(',patch+'\n    ggml_tensor * cur = build_vit(').replace('                            model.position_embeddings,','                            pos,')
 target.write_text(modified)
 import difflib
 (D/'internvl-resize.patch').write_text(''.join(difflib.unified_diff(orig.splitlines(True),modified.splitlines(True),fromfile='a/tools/mtmd/models/internvl.cpp',tofile='b/tools/mtmd/models/internvl.cpp')))
 files={str(p.relative_to(dst)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.rglob('*') if p.is_file()}
 (D/'dependency-source-sha256.json').write_text(json.dumps(files,sort_keys=True,indent=2))
 status['source_manifest_sha256']=hashlib.sha256((D/'dependency-source-sha256.json').read_bytes()).hexdigest();status['state']='building';save()
 for arch in ['arm','host']:
  build=W/('build-'+arch)
  cmd=['cmake','-S',str(D),'-B',str(build),'-DLLAMA_SOURCE='+str(dst),'-DCMAKE_BUILD_TYPE=Release','-DBUILD_SHARED_LIBS=OFF','-DLLAMA_BUILD_COMMON=OFF','-DLLAMA_BUILD_TOOLS=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_APP=OFF','-DLLAMA_BUILD_MTMD=ON','-DGGML_NATIVE=OFF','-DGGML_OPENMP=OFF','-DGGML_CPU_REPACK=ON','-DGGML_BACKEND_DL=OFF']
  if arch=='arm': cmd+=['-DCMAKE_TOOLCHAIN_FILE='+str(D/'aarch64.cmake'),'-DCMAKE_C_FLAGS=-march=armv8.2-a+dotprod+fp16','-DCMAKE_CXX_FLAGS=-march=armv8.2-a+dotprod+fp16']
  with (D/(arch+'-build.log')).open('w') as log:
   subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
   subprocess.run(['cmake','--build',str(build),'--target','vision-worker','-j','6'],stdout=log,stderr=subprocess.STDOUT,check=True)
  binary=build/'vision-worker';status[arch]={'binary':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'bytes':binary.stat().st_size};save()
 status['state']='compiled_unvalidated';status['ended']=time.time();save()
except Exception:
 status['state']='failed';status['error']=traceback.format_exc();save();raise
