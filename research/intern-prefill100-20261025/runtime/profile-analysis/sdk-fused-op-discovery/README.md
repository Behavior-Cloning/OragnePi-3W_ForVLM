# Pinned SDK operator discovery

Read-only introspection inside image sha256:b2447e88c70d334cbadf91d1cfef2874d6a4360f0e2b075606df32547ec8cbe3. Acuity6.30.22 has RMSNormalize op rmsnormalize, parameters axis(default-1),eps(default0.0001),coef scale. Candidate explicitly uses model-derived epsilon≈1e-6, not SDK default. Attention class has two named inputs and use_scale/dropout/score_mode parameters; this inspection does not establish masked causal GQA or fused FlashAttention support. No board execution, no operator speed claim.
