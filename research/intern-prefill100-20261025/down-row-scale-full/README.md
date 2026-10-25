# Row-normalized down: C512 M32 full-model experiment

Target remains full-model prefill and TTFT <=100 ms for S64/128, D64–256; not met by local block speed.

- A/B/A: original M32, row-normalized INT16 down in fused layers 2/3 M32 graph, original repeat.
- Same frozen IDs, system-only KV, true original M8 tail, unchanged decode/head/embedding and binary.
- S64/128 includes 5 system-template tokens; body 59/123. D64/128/192/256 plus 8 template gives new72/136/200/264.
- 24 cost cases per variant, fixed feedback 32 outputs; 48 natural cases and 8 four-step full-logit diagnostics per variant.
- Synthetic streamed speech frontend and RAG resident; RAG queried only in natural phase. No vision, microphone or motors.
- Candidate is used for system preparation too. System construction excluded from warm TTFT but included in resources. Candidate is not precision-equivalent to original.
- Double locks; new NB and logs in private tmpfs; immutable original banks reused via SHA-checked symlinks. No duplicate launch or automatic retry.
- Analyzer checks 72 cost/144 natural/24 diagnostic results, full file hashes/finite, actual cache counts, all releases; token equality is not quality approval.

Runtime sources are frozen by controller source hashes. Inspect the controller PID plus status to determine liveness. A status file alone is insufficient. Recover terminal logs before modifying/restarting any failed run.
