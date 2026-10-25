# Next: isolate the 8-token tail after I16 range repair

Do not alter the running fulli16-range16 A-B-A. Its only graph change is M16 prefill L2. Read its completion and numerical/semantic results before qualification proceeds.

All target bodies64/128/192/256 plus8 chat template tokens give new72/136/200/264. CORRECTION after actual run: all72 performance requests have scalar_prompt_tokens=0 and scalar_prompt_seconds=0. OPI_TTFT_TAIL_MIN=8 uses a padded16-row call for the final8 real rows. A true8 bank targets wasted padded rows, not a scalar tail. Do not infer measured benefit solely from a compiled M8 graph.

Prepared candidate: graphs/candidates-down-i16-c1280-m8-range16/integration-manifest.json. Actual16 external ports match original M8, capacity1280/query8. Export payload exactly equals M16 I16 range16. Existing original12-block M8 stack is cataloged by research/intern-nextopt-20261025/prefill8-stack-manifest.json; historical board bank /opt/opi-vlm/intern-candidates-20261025/tail8 requires renewed SHA verification after the controller releases locks.

Proposed next same-input A-B-A: I16 M16 padded16-tail8 / same I16 M16 plus true8 / I16 M16 padded16-tail8 repeat. Only middle variant receives OPI_PREFILL8_DIR pointing to its private12-link bank with11 original M8 blocks and new range16 L2 M8. Same main29-file graph, runtime/affinity/load/input IDs. Require exactly1 true_tail8_call per target request and0 scalar tail in candidate; no truncated tokens or dynamic-cache reuse. Prefixes64/128 are multiples16 and reuse exactly64/128. Repeat32 fixed-feedback outputs for speed, natural independent24 scenes separately, full-vocabulary diagnostic4 steps each cell.

Quality boundaries: old semantic24 is a development regression set; new independent-household-v2 was frozen before inspecting current range16 full-model outputs. Do not recalibrate on new set before first evaluation. Action compatibility and complete constraint fulfillment separate; RAG Top3 remains mock.

Capacity1280 is retained only to isolate tail effect. Future C512 graph work can cover max static128+new264+31 generated feedback=423 without truncation, but must be measured separately and numerically revalidated. Existing original single-block layer0/1 C512 times do not support blindly increasing query batch: M16 8.423ms, M32 10.920ms, M64 34.615ms, M128 84.208ms, M256 211.133ms; M32 best per-token among those measured. This is one block, not a prediction of full-model TTFT.

Storage: never duplicate full model sets blindly. After full-run evidence is independently recovered and hashed, remove only completed board diagnostic f32 copies if space is needed; keep WSL evidence/production graphs. Inspect current memory and disk under both locks before staging. No concurrent board jobs.

Runtime prerequisite: resident-combined-v2 explicitly rejects OPI_PREFILL8_DIR when capacity is not256. A future512/1280 true8 experiment needs an isolated runtime extension and host/board contract checks, and cannot reuse the current binary unmodified. Current full-model I16 logits have substantial differences; defer promoting or composing this precision candidate until actual NPU trajectory error is diagnosed.
