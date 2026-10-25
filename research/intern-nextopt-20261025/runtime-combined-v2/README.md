# Combined runtime close investigation — v2 candidate

Status: host validation and ARM build complete; board close validation pending. This is a close-only candidate, not a demonstrated root-cause fix yet.

## Observed failure

`intern-nextbench-20261025/results/combined-matrix-r1/combined-tail8.log` ends with `double free or corruption (out)` after the parent reported all requested inference/diagnostic rows returned. No `closed` event was delivered. The log has no original teardown phase markers, so it cannot identify which `free` detected corruption. The parent separately observed the driver returned to zero after process death; that is not a successful graceful close.

The original `free_net` destroys owned input/output buffers before `vip_destroy_network`; `free_prefill` proceeds forward through networks. Alias consumers can therefore remain alive after a predecessor buffer is destroyed. This is a lifecycle hazard, not yet proof that it caused this particular failure. Earlier static confidence in safe ownership was insufficient because ownership alone does not guarantee lifetime.

## Candidate change

`session_release.h` establishes three close barriers: destroy all groups, destroy every network (while all user buffers remain alive), then unmap/destroy each Layer's original owned buffers exactly once. Tensor fields are never overwritten with alias handles; only `vip_set_input` bindings alias decoder KV or previous-layer output. Pointer fields are cleared so a second close is harmless. Host mask templates are freed after VIP release, as before. Close-phase markers aid a minimal smoke test.

No prompt, KV update, mask, argmax, graph, precision, inference loop, or timing boundary changed. `combined-v2-runtime.diff` records the resident entry diff; `session_release.h` is the sole added runtime helper. Original artifacts remain unchanged. The combined experiment still requires capacity256 + batch16, and only an exact final8 invokes the additional graph bank.

## Evidence

- 128 mock-SDK lifecycle sessions passed ASan/UBSan, each closed twice, with zero remaining buffers/networks/groups. The mock deliberately checks that no live network retains a destroyed buffer. It covers 1/12 blocks, one/two prefill banks, grouped/ungrouped and scalar ownership. This checks our contract, not proprietary SDK internals.
- Mask restoration: 1920 tensor comparisons in each of six modes (INT16 and FP16, each original/template/template+validate), ASan/UBSan rc0; repeated prefix rewind, KV reset and cache free checked.
- Single-pass greedy: 221212 cases, zero mismatches under ASan/UBSan.
- ARM optimized binary: 70704 bytes, SHA256 `ff758d8eb013b7ebd05eae1f95634ecef065b92f0bf1fca7b292256351a2900d`.
- ARM ASan v2: 505200 bytes, SHA256 `f35ab852873d663e2f75fe29f00f8549882364711445447598f53158ea4ea0f8`.

An independent original-behavior ASan binary plus close markers is in `../runtime-combined-debug`. Its purpose is to obtain the actual failing allocation/free stack before declaring cause. Neither ASan runtime is suitable for performance claims. Next gate: parent-run minimal request+close smoke, then repeated close and unchanged full-logits comparison. No board access was performed by this subagent.

## Root-run reproduction updates

- `intern-nextbench-20261025/results/close-asan-1`: original release behavior, one system64 + full104 fixed32 request, completed and closed with driver0; no ASan report.
- `intern-nextbench-20261025/results/combined-asan-full-1`: original release behavior, 24 performance +12 natural +12 cost-natural +5 diagnostic rows, followed by `closed` driver0 and `complete`; no ASan report. Independent event read confirmed these counts. This did not reproduce the original fault and is not a v2 result.
- An original O2, non-ASan detailed-close binary is preserved in `../runtime-combined-close-trace`, SHA256 `17b06d118ef79952112caf8ae74c0a6f62a80ff2403e7df64caf20cb5ec0aa62`. Root is replaying the complete sequence with `MALLOC_CHECK_=3`; result pending.
- Read-only SDK disassembly confirms VIPhal uses a custom heap pool for some host allocations. `viphal_free_memory` first asks `viphal_heap_free_mem`; a failure falls back to the OS free function. This means a pool bookkeeping issue could surface as a libc free error; it is only a possible mechanism, not a diagnosed cause. Disassembly and exact local library SHA are saved. ASan does not independently instrument every suballocation inside the proprietary pool.

The original O2 detailed-close replay also completed: 584 buffer destroys used 584 unique handles; 40 network destroys used 40 unique handles, all 40 completed. The driver closed at zero and no warning appeared. `original-close-trace-audit.json` stores this independent parse. The initial failure remains unexplained; this is not evidence that the fault was repaired.

Final candidate delivery is frozen in `FINAL_DELIVERY.json`. The only production differences are the resident close call and new `session_release.h`; inference modules match the frozen original byte-for-byte. Root may next run three minimal close cycles and a same-binary full baseline/tail8/repeat matrix. Qualification requires actual successful closure and unchanged numerical outputs; do not replace the default solely based on these host tests.
