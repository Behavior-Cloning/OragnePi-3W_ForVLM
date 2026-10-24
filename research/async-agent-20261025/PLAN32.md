# 32-token dynamic abstraction follow-up

User confirmed: only the recent60second abstraction belongs in dynamic input; current partial also shares32-token budget. All action/hardware constraints remain in fixed system. No extra dynamic instruction or label. Chat template overhead and actual new-prefill count are measured separately.

Prepared harness32.py/controller32.py without board access or changing running stress60-v2. Same fixed system asv2 for controlled comparison; summary budget32 includes partial. Exacttoken assertion at runtime. Unique board directory async-agent32-20261025.

After stress60-v2 and finish60.py complete and locks release: run a fresh300s v2 baseline followed by300s 32-token variant using same12speech loop and2–5s deterministic eventseed. Record order andthermal differences; this is scenario comparison, not proof of equalquality. Preserve full hour report first. Both share known RAG compatibility limitations and same mock-only execution. Evaluate summary critical/latest clause dropflags, tokens, latency, output validity and memory; do not equate narrower output with betterunderstanding.

This file records preparation only; neither paired run has started.
