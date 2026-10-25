# Intern NPU prefill / TTFT 100 ms investigation

2026-10-25. Work in progress. Goal remains unachieved.

Target both complete dynamic-prefill and model TTFT <=100 ms for dynamic body64–256 tokens, static system64/128 including its5 template tokens. System body59/123; dynamic role template adds8; actual new72/136/200/264, only system KV reused. Max prompt392, output32 means423 retained positions: C512 is sufficient, C256 is not. Every report must list all counts.

Freeze common per-cell token IDs and weight/quantization provenance. Timed prefill includes batched and scalar tails; TTFT includes request setup, all new tokens and first full-vocabulary head. Model load/system preparation are separate, never hidden by dynamic cache reuse. Keep natural quality separate from exact-length cost fixtures.

First qualify M16/32/64/128/256 fused two-layer C512 graphs at L0 and sensitive L2 with same real FP32-derived history. These single-block probes are not full-model timing/quality; API cost per row cannot be called full request latency. Start with3 warmups/30 API observations, compare same valid outputs, and distinguish projection from measurement. Then test full-model candidate only if evidence supports admission and resources fit.

Full-model core grid64/128/192/256 body x64/128 static, 3 scenes each. Add bulk boundary cases around actual new128/192/256 (body119/120/121,183/184/185,247/248/249) and255/256. Sampling these lengths does not establish every length in the entire interval. Keep audio/RAG residency contract, logging of CPU/frequency/temperature/memory/swap/driver release, numerical comparison and natural responses. No motors or vision claimed.

Root owns board through Windows/WSL hostlock and board flock. Read controller status while locked; no outside polling. Code authoritative on Windows. Agents own graphs/, runtime/, and audit documents respectively. Preserve prior failed-close evidence; no root cause established.
