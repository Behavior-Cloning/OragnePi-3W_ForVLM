#include "models.h"

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


ggml_cgraph * clip_graph_internvl::build() {
    GGML_ASSERT(model.class_embedding != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);

    const bool front = opi_merge_mode("front");
    const bool late = opi_merge_mode("late");
    const bool keep64 = opi_merge_mode("keep64");
    const bool neighbor = opi_merge_mode("neighbor");
    GGML_ASSERT(n_patches_x == n_patches_y && n_patches_x % 2 == 0);
    GGML_ASSERT(!(front || late) || n_patches_x == 32);
    const int used_patches = front ? n_patches / 4 : n_patches;
    const int n_pos = used_patches + 1;
    ggml_tensor * pos = model.position_embeddings;
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

    ggml_tensor * inp = build_inp();

    if (front) {
        inp = opi_pool_grid(ctx0, inp, n_embd, n_patches_x, n_patches_y, n_batch);
        auto * patch_pos = ggml_view_2d(ctx0, pos, n_embd, n_patches, pos->nb[1], 0);
        auto * cls_pos = ggml_view_2d(ctx0, pos, n_embd, 1, pos->nb[1], n_patches * pos->nb[1]);
        patch_pos = opi_pool_grid(ctx0, ggml_cont(ctx0, patch_pos), n_embd, n_patches_x, n_patches_y, 1);
        patch_pos = ggml_reshape_2d(ctx0, patch_pos, n_embd, used_patches);
        pos = ggml_concat(ctx0, patch_pos, cls_pos, 1);
    }
    // add CLS token
    ggml_tensor * cls_repeated = ggml_repeat_4d(ctx0, model.class_embedding,
            model.class_embedding->ne[0], 1, n_batch, 1);
    inp = ggml_concat(ctx0, inp, cls_repeated, 1);

    // The larger models use a different ViT, which uses RMS norm instead of layer norm
    // ref: https://github.com/ggml-org/llama.cpp/pull/13443#issuecomment-2869786188
    norm_type norm_t = (hparams.n_embd == 3200 && hparams.n_layer == 45)
        ? NORM_TYPE_RMS // 6B ViT (Used by InternVL 2.5/3 - 26B, 38B, 78B)
        : NORM_TYPE_NORMAL; // 300M ViT (Used by all smaller InternVL models)

    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            norm_t,
                            hparams.ffn_op,
                            pos,
                            nullptr);

    // remove CLS token
    cur = ggml_view_3d(ctx0, cur,
        n_embd, used_patches, n_batch,
        cur->nb[1], cur->nb[2], 0);
    cur = ggml_cont(ctx0, cur);

    // keep64 is an UNTRAINED weight-reuse adapter: retain all spatial tokens,
    // repeat each token's channels to match the old projector's 4*D input.
    // This changes no token count and is NOT the native trained projector path.
    if (neighbor) {
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
    } else {
    // pixel shuffle
        const int scale_factor = model.hparams.n_merge;
        const int bsz    = n_batch;
        const int height = front ? n_patches_y / 2 : n_patches_y;
        const int width  = front ? n_patches_x / 2 : n_patches_x;
        GGML_ASSERT(scale_factor > 0);
        cur = ggml_reshape_4d(ctx0, cur, n_embd * scale_factor, height / scale_factor, width, bsz);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont_4d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            height / scale_factor,
            width / scale_factor,
            bsz);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        // flatten to 2D
        cur = ggml_cont_3d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            cur->ne[1] * cur->ne[2],
            cur->ne[3]);
    }

    // projector (always using GELU activation)
    {
        // projector LayerNorm uses pytorch's default eps = 1e-5
        // ref: https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct/blob/a34d3e4e129a5856abfd6aa6de79776484caa14e/modeling_internvl_chat.py#L79
        cur = build_norm(cur, model.mm_0_w, model.mm_0_b, NORM_TYPE_NORMAL, 1e-5, -1);
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_3_w, model.mm_3_b,
            FFN_GELU,
            -1);
    }

    if (late) {
        cur = opi_pool_grid(ctx0, cur, n_mmproj_embd, n_patches_x / 2, n_patches_y / 2, n_batch);
    }
    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
