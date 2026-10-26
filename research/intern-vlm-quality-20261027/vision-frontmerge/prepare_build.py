from pathlib import Path
import shutil,hashlib,json,subprocess,time,traceback
D=Path(__file__).resolve().parent
S=Path('/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/cpu-tools/source')
W=Path('/opt/opi-lab/builds/intern-frontmerge-20261027-r1');W.mkdir(parents=True,exist_ok=True)
status={'started':time.time(),'state':'preparing','source_provenance':'restored source tree; parent Git revision is NOT a llama.cpp revision'}
def save(): (D/'build-status.json').write_text(json.dumps(status,indent=2))
try:
 save();dst=W/'source'
 assert not dst.exists(),'build directory must be fresh'
 shutil.copytree(S,dst,ignore=lambda root,names: [n for n in names if (Path(root)/n).is_dir() and (n in ['.git','__pycache__','build'] or n.startswith('build-'))])
 target=dst/'tools/mtmd/models/internvl.cpp';orig=target.read_text()
 helper=r"""
#include <cstdlib>
#include <cstring>
static bool opi_merge_mode(const char * mode) {
    const char * v = std::getenv("OPI_VISION_MERGE");
    return v && std::strcmp(v, mode) == 0;
}
static ggml_tensor * opi_pool_grid(ggml_context * ctx, ggml_tensor * x, int dim, int width, int height, int batch) {
    GGML_ASSERT(width % 2 == 0 && height % 2 == 0);
    x = ggml_reshape_4d(ctx, x, dim, width, height, batch);
    x = ggml_permute(ctx, x, 2, 0, 1, 3);
    x = ggml_cont(ctx, x);
    x = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
    x = ggml_permute(ctx, x, 1, 2, 0, 3);
    return ggml_cont_3d(ctx, x, dim, width * height / 4, batch);
}
"""
 modified=orig.replace('#include "models.h"','#include "models.h"\n'+helper)
 modified=modified.replace('    const int n_pos = n_patches + 1;','    const bool front = opi_merge_mode("front");\n    const bool late = opi_merge_mode("late");\n    GGML_ASSERT(n_patches_x == 32 && n_patches_y == 32);\n    const int used_patches = front ? n_patches / 4 : n_patches;\n    const int n_pos = used_patches + 1;\n    ggml_tensor * pos = model.position_embeddings;')
 modified=modified.replace('    // add CLS token',r"""    if (front) {
        inp = opi_pool_grid(ctx0, inp, n_embd, n_patches_x, n_patches_y, n_batch);
        auto * patch_pos = ggml_view_2d(ctx0, pos, n_embd, n_patches, pos->nb[1], 0);
        auto * cls_pos = ggml_view_2d(ctx0, pos, n_embd, 1, pos->nb[1], n_patches * pos->nb[1]);
        patch_pos = opi_pool_grid(ctx0, ggml_cont(ctx0, patch_pos), n_embd, n_patches_x, n_patches_y, 1);
        patch_pos = ggml_reshape_2d(ctx0, patch_pos, n_embd, used_patches);
        pos = ggml_concat(ctx0, patch_pos, cls_pos, 1);
    }
    // add CLS token""")
 modified=modified.replace('                            model.position_embeddings,','                            pos,').replace('        n_embd, n_patches, n_batch,','        n_embd, used_patches, n_batch,').replace('const int height = n_patches_y;','const int height = front ? n_patches_y / 2 : n_patches_y;').replace('const int width  = n_patches_x;','const int width  = front ? n_patches_x / 2 : n_patches_x;')
 modified=modified.replace('    // build the graph',r"""    if (late) {
        cur = opi_pool_grid(ctx0, cur, n_mmproj_embd, n_patches_x / 2, n_patches_y / 2, n_batch);
    }
    // build the graph""")
 target.write_text(modified)
 import difflib
 patch=''.join(difflib.unified_diff(orig.splitlines(True),modified.splitlines(True),fromfile='a/tools/mtmd/models/internvl.cpp',tofile='b/tools/mtmd/models/internvl.cpp'))
 target=dst/'tools/mtmd/clip.cpp';orig=target.read_text()
 needle='    int n_patches = (img->nx() / patch_size) * (img->ny() / patch_size);'
 assert orig.count(needle)==1
 modified=orig.replace(needle,needle+r"""
    // Fixed-resolution front/late compression experiments; no change to other models.
    const char * merge_mode = std::getenv("OPI_VISION_MERGE");
    if (ctx->proj_type() == PROJECTOR_TYPE_INTERNVL && merge_mode &&
        (std::strcmp(merge_mode, "front") == 0 || std::strcmp(merge_mode, "late") == 0)) {
        GGML_ASSERT(img->nx() == 448 && img->ny() == 448);
        return 64;
    }
""")
 target.write_text(modified)
 patch+=''.join(difflib.unified_diff(orig.splitlines(True),modified.splitlines(True),fromfile='a/tools/mtmd/clip.cpp',tofile='b/tools/mtmd/clip.cpp'))
 (D/'internvl-frontmerge.patch').write_text(patch)
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
