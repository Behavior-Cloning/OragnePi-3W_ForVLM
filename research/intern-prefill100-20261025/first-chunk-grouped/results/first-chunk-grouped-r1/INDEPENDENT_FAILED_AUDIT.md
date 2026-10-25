# Independent failed-run audit — first-chunk-grouped-r1

**Rejected runtime path.** Controller rc=1 and recovered meta=failed. The third variant failed during its first system preparation, before any prediction result. Destroying the group did not make the member networks available for individual execution in this actual SDK run: `batch prefill status=-9`, `network ... is belong to one group`, then commit failure.

- Counts: 3 ready, 16 completed results (8 fixed / 8 natural), 2 successful close/driver0, 1 error. Hybrid-grouped: **0 results**, first logits file 0 bytes, no first-chunk/restoration markers.
- Independently read all **32 complete F32 files** (16 output and 16 system preparation): size, actual SHA, finite values and argmax/token correspondence pass. Actual runtime and all build-source hashes match the recorded build.
- Prompt/new/reused counts match the four frozen input presentations; each request resets and prepares only its system prefix. Fixed-feedback requests use 4 steps; natural requests use max32/EOS. Every request is diagnostic and timing-ineligible.
- All 16 completed output logits and token sequences are **bitwise identical** to the corresponding earlier original-grouped / i16-first-original results. This reproduces prior evidence; it does not create new independent quality evidence.
- The third worker has **no close/driver0 event** in this run. Only a separate parent post-health observation can establish release after failure; this audit performed no board operations.

Do not launch the prepared fulli16-firstfix matrix with this binary. A replacement with immutable network membership needs its own successful controls, actual-logit parity and close evidence. No speech/RAG concurrency, motor execution, speed result, or broad quality claim is supported here.

Machine detail: `INDEPENDENT_FAILED_AUDIT.json`.
