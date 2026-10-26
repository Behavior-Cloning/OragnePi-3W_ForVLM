from pathlib import Path
import hashlib,json,difflib,subprocess
D=Path(__file__).resolve().parent;S=Path('/opt/opi-lab/builds/intern-frontmerge-20261027-r1/source');W=S.parent
p=S/'tools/mtmd/models/internvl.cpp';a=p.read_text();b=a
b=b.replace('    GGML_ASSERT(n_patches_x == 32 && n_patches_y == 32);',r"""    const bool keep64 = opi_merge_mode("keep64");
    GGML_ASSERT(n_patches_x == n_patches_y && n_patches_x % 2 == 0);
    GGML_ASSERT(!(front || late) || n_patches_x == 32);""")
needle='    ggml_tensor * pos = model.position_embeddings;'
assert b.count(needle)==1
b=b.replace(needle,needle+r"""
    const int old_grid = (int)std::sqrt(pos->ne[1] - 1);
    GGML_ASSERT(old_grid * old_grid + 1 == pos->ne[1]);
    if (n_patches_x != old_grid) {
        auto * cls_pos = ggml_view_2d(ctx0, pos, n_embd, 1, pos->nb[1], old_grid * old_grid * pos->nb[1]);
        auto * grid = ggml_view_2d(ctx0, pos, n_embd, old_grid * old_grid, pos->nb[1], 0);
        grid = ggml_reshape_3d(ctx0, ggml_cont(ctx0, grid), n_embd, old_grid, old_grid);
        grid = ggml_permute(ctx0, grid, 2, 0, 1, 3);
        grid = ggml_interpolate(ctx0, grid, n_patches_x, n_patches_y, n_embd, 1, GGML_SCALE_MODE_BICUBIC);
        grid = ggml_permute(ctx0, grid, 1, 2, 0, 3);
        grid = ggml_cont_2d(ctx0, grid, n_embd, n_patches);
        pos = ggml_concat(ctx0, grid, cls_pos, 1);
    }
""")
b=b.replace('    // pixel shuffle\n    {',r"""    // keep64 is an UNTRAINED weight-reuse adapter: retain all spatial tokens,
    // repeat each token's channels to match the old projector's 4*D input.
    // This changes no token count and is NOT the native trained projector path.
    if (keep64) {
        cur = ggml_repeat_4d(ctx0, cur, n_embd * 4, used_patches, n_batch, 1);
        cur = ggml_reshape_3d(ctx0, cur, n_embd * 4, used_patches, n_batch);
    } else {
    // pixel shuffle""")
assert a!=b;p.write_text(b);patch=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/tools/mtmd/models/internvl.cpp',tofile='b/tools/mtmd/models/internvl.cpp'))
p=S/'tools/mtmd/clip.cpp';a=p.read_text();b=a;needle='    const char * merge_mode = std::getenv("OPI_VISION_MERGE");'
assert b.count(needle)==1
b=b.replace(needle,needle+r"""
    if (ctx->proj_type() == PROJECTOR_TYPE_INTERNVL && merge_mode && std::strcmp(merge_mode, "keep64") == 0) {
        return n_patches; // keep every spatial token; 112x112 => 64
    }
""");p.write_text(b);patch+=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/tools/mtmd/clip.cpp',tofile='b/tools/mtmd/clip.cpp'));(D/'internvl-resolution-keep64.patch').write_text(patch)
p=D/'worker.cpp';a=p.read_text();b=a.replace('native|front|late','native|front|late|keep64').replace('mode!="late")','mode!="late"&&mode!="keep64")')
b=b.replace('if(side!=448||threads<1||threads>8)','if((side!=448&&side!=224&&side!=112)||((mode=="front"||mode=="late")&&side!=448)||(mode=="keep64"&&side!=112)||threads<1||threads>8)')
b=b.replace('std::vector<float> emb(size_t(mode=="native"?256:64)*clip_n_mmproj_embd(c.ctx_v));','const int out_tokens=mode=="native"?(side/28)*(side/28):64;std::vector<float> emb(size_t(out_tokens)*clip_n_mmproj_embd(c.ctx_v));')
b=b.replace('const int tokens=mode=="native"?256:64,','const int tokens=out_tokens,').replace('mode=="front"?256:1024','mode=="front"?256:(side/14)*(side/14)');assert b!=a;p.write_text(b)
s={'state':'building','source_tree':str(S),'worker_source_sha256':hashlib.sha256(b.encode()).hexdigest(),'variants':['native448','front448','late448','native224','keep64-112-untrained-projector-adapter'],'source_patch_sha256':hashlib.sha256(patch.encode()).hexdigest()}
(D/'build-r3-status.json').write_text(json.dumps(s,indent=2))
for arch in ['arm','host']:
 with (D/(arch+'-r3-build.log')).open('w') as log:subprocess.run(['cmake','--build',str(W/('build-'+arch)),'--target','vision-worker','-j','6'],stdout=log,stderr=subprocess.STDOUT,check=True)
 binary=W/('build-'+arch)/'vision-worker';s[arch]={'binary':str(binary),'bytes':binary.stat().st_size,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
s['state']='compiled_unvalidated';(D/'build-r3-status.json').write_text(json.dumps(s,indent=2));print(json.dumps(s),flush=True)
