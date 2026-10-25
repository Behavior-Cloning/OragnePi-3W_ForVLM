# Suffix-group first-chunk source audit

Read-only source review found no new static blocker. Actual runtime qualification remains pending recovered logits and close events.

Only batch_prefill.h differs from the validated first-chunk ungrouped runtime. Binary and all recorded build-source hashes pass. One immutable group contains blocks2–11; block0 and both original/candidate block1 stay standalone throughout their lifetime. Each batch has three submissions. No group mutation occurs in run_prefill_rows; group destruction remains in release code.

Original block1 retains the same eleven input / five output bindings and exact tensor-contract checks. Only position0 selects original block1; later chunks use the candidate. The extra original graph hash is explicit. Release remains groups, all networks, then owned buffers.

The four presentations are unchanged. Each request resets and prepares only system KV. Fixed feedback is four steps, natural max32/EOS; all timings are diagnostic/ineligible. Require all24 results, actual suffix/ungrouped full-logit parity, three successful close/driver0, and expected eight first-chunk/suffix markers. Four development presentations cannot establish general quality or stability.

Legacy variant binary metadata names the old binary, but execution explicitly selects the new ROOT binary and top-level hash. Arbitrary controller SSH/deadline failure would require parent recovery under locks. No board operation was performed by this audit.
