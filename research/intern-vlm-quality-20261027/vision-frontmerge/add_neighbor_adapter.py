from pathlib import Path
import difflib,subprocess,hashlib,json
D=Path(__file__).resolve().parent;W=Path('/opt/opi-lab/builds/intern-frontmerge-20261027-r1');S=W/'source';p=S/'tools/mtmd/models/internvl.cpp';a=p.read_text();b=a
needle='    const bool keep64 = opi_merge_mode("keep64");';assert b.count(needle)==1
b=b.replace(needle,needle+'\n    const bool neighbor = opi_merge_mode("neighbor");')
needle='    if (keep64) {\n        cur = ggml_repeat_4d(ctx0, cur, n_embd * 4, used_patches, n_batch, 1);\n        cur = ggml_reshape_3d(ctx0, cur, n_embd * 4, used_patches, n_batch);\n    } else {';assert needle in b
b=b.replace(needle,r"""    if (neighbor) {
        // UNTRAINED stride-1 2x2 neighborhood adapter: preserve every
        // spatial location, clamp edges, reuse the existing 4*D projector.
        GGML_ASSERT(n_batch == 1);
        const int w = n_patches_x, h = n_patches_y;
        auto * grid = ggml_reshape_3d(ctx0, cur, n_embd, w, h);
        auto shift_right = [&](ggml_tensor * g) {
            auto * body = ggml_view_3d(ctx0, g, n_embd, w-1, h, g->nb[1], g->nb[2], g->nb[1]);
            auto * edge = ggml_view_3d(ctx0, g, n_embd, 1, h, g->nb[1], g->nb[2], (w-1)*g->nb[1]);
            return ggml_concat(ctx0, body, edge, 1);
        };
        auto * body = ggml_view_3d(ctx0, grid, n_embd, w, h-1, grid->nb[1], grid->nb[2], grid->nb[2]);
        auto * edge = ggml_view_3d(ctx0, grid, n_embd, w, 1, grid->nb[1], grid->nb[2], (h-1)*grid->nb[2]);
        auto * down = ggml_concat(ctx0, body, edge, 2);
        auto * upper = ggml_concat(ctx0, grid, shift_right(grid), 0);
        auto * lower = ggml_concat(ctx0, down, shift_right(down), 0);
        cur = ggml_concat(ctx0, upper, lower, 0);
        cur = ggml_cont_3d(ctx0, cur, n_embd * 4, used_patches, 1);
    } else if (keep64) {
        cur = ggml_repeat_4d(ctx0, cur, n_embd * 4, used_patches, n_batch, 1);
        cur = ggml_reshape_3d(ctx0, cur, n_embd * 4, used_patches, n_batch);
    } else {""");p.write_text(b);patch=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/tools/mtmd/models/internvl.cpp',tofile='b/tools/mtmd/models/internvl.cpp'))
p=S/'tools/mtmd/clip.cpp';a=p.read_text();b=a.replace('&& std::strcmp(merge_mode, "keep64") == 0)', '&& (std::strcmp(merge_mode, "keep64") == 0 || std::strcmp(merge_mode, "neighbor") == 0))');assert a!=b;p.write_text(b);patch+=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/tools/mtmd/clip.cpp',tofile='b/tools/mtmd/clip.cpp'));(D/'internvl-neighbor-adapter.patch').write_text(patch)
p=D/'worker.cpp';a=p.read_text();b=a.replace('mode!="keep64")','mode!="keep64"&&mode!="neighbor")').replace('(mode=="keep64"&&side!=112)','((mode=="keep64"||mode=="neighbor")&&side!=112)');assert a!=b;p.write_text(b)
with (D/'neighbor-host-build.log').open('w') as log:subprocess.run(['cmake','--build',str(W/'build-host'),'--target','vision-worker','-j','6'],stdout=log,stderr=subprocess.STDOUT,check=True)
binary=W/'build-host/vision-worker';(D/'neighbor-build-status.json').write_text(json.dumps({'state':'host_compiled_unvalidated','binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'patch_sha256':hashlib.sha256(patch.encode()).hexdigest()},indent=2));print('neighbor host built')
