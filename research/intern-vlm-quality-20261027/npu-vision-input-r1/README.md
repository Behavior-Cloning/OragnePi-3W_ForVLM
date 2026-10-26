# NPU visual embedding input candidate r1

Status: ARM compiled; host ASan/UBSan input checks passed. NOT board validated or promoted.

Independent copy of compact-m64-r4 runtime (r6 binary parent); no baseline edits.
Protocol retains OPI1 eight-word header. Flags bit1 enables visual rows, reserved word is count 1..64. Following token IDs and optional fixed feedback: uint32 image_start, uint32 system_reuse_cap, then count×896 little-endian float32 vectors. Bit0 remains full-logits diagnostic. Bounds, finite values and payload length are checked.

Visual vectors replace word embeddings before first decoder layer in main/compact/tail8/padded batches AND scalar prompt. Padded rows repeat last real row. Decode tokens use normal text embeddings. Reuse is capped by both previous request and current request system limits before normal batch alignment: switching images or switching image→text cannot reuse image-dependent KV. Caller must supply actual static prefix length (not image start). Single-thread worker only.

Next required: board text-only full-logits parity with unchanged binary; same-image cold/reuse tests; image A→B and image→text tests against fresh sessions; NPU VQA using frozen 224 embeddings and then resident vision integration. No performance or quality claim from compilation/host tests. Valid input contents/IDs/token budget are controller responsibilities. Vision output must use the matching 896-dimensional projector.
