# C512/M32 exact original-system preparation candidate

Motivation: row-normalized down full-model improves TTFT but changes static KV and loses two requested action details. Test whether retaining original sensitive FP16 block for the fixed cached system improves output at unchanged hot input sizes.

Running bounded board A/B/C/A; not qualified or promoted. Runtime is capacity512 baseline plus explicit prefix policy previously used at C1280; restored original M8 graph selection, C512 tail eligibility and propagated prefix threshold. Static64/128 is a multiple of batch32; prefix prepare uses original down for all static chunks. Dynamic uses row-normalized candidate, original M8 tail and decode. No extra dynamic KV reuse.

The runner tests one identical new binary in original/all-row/protected-system/original-repeat conditions. Require same S64/128 and D64/128/192/256+8; model warm TTFT excludes system construction for every variant. Include prefix-only logits checks, actual prefix branch counts, original-repeat equality, full logits and natural48 regression, speech/RAG residency, resources and release. Prefix logits equality alone is not proof every KV byte matches. This candidate is not a deployment approval and cannot claim 100ms.

Run down-row-scale-prefixsafe-r1 controller49958; finish50000 waits and analyzes. Runtime/protocol/source audit passed; no result yet at launch. Hardened explicit reset max_new1, continuation capped exactly to S. Independent runner audit verifies expected96/192/32 main records,8 prefix diagnostics and4 repeated-input cache probes. Prefix protected path uses3 submissions perblock; overhead included. Initial observer startup race is preserved as .initial-failed; board process never restarted.
