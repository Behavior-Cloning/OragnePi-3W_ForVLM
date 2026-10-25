# C1280 full-model I16 range16 qualification

Only `prefill-layer2.nb` changes. A-B-A original / calibrated I16 / original. No production files are replaced: a new 29-file symlink overlay points one file at the candidate.

System body59/123 + template5 = static64/128; dynamic body64/128/192/256 + template8 = new72/136/200/264. Each measured request resets and prepares only system KV. Performance:24 frozen cost prompts per variant,32 fixed-feedback outputs, no diagnostic IO. Natural:24 old semantic prompts per variant, max32 outputs with EOS; these are development/regression examples despite historical `independent_holdout` fields retained verbatim. Diagnostic:8 cells ×4 full-vocabulary logits per variant, excluded from timings.

All3 variants use identical runtime, env, affinity, C1280/M16 geometry and synthetic streaming speech/RAG residency. RAG queries only follow natural outputs. The frozen LLM inputs are not triggered by ASR. Resources include system-cache reconstruction and diagnostic phases. This is neither a whole-agent1h stability run nor camera/microphone/motor validation.

The new I16 grid was tuned after observing clipping in the single-block development screen. `same_quantizer_bytes` and `same_14_fc_exported_weight_bytes` are explicitly false. Actual16 external ports match; downstream full-model quality remains to be assessed from evidence. New `independent-household-v2` scenes are frozen separately, not substituted into the running benchmark.

Controller has hostlock and board flock, bounded3600s timeout, partial recovery, unique directory, full SHA verification. While live, inspect only WSL controller state/log. Do not launch another board test. Keeper539 must remain live. Analyze only complete rc0 recovered data with `python analyze_fulli16.py fulli16-range16-c1280-aba-r1` under the export environment.

Next candidate: true8 tail bank with range16 L2 (`graphs/candidates-down-i16-c1280-m8-range16`), compiled and actual ports checked, not deployed yet. It must be a separate same-input comparison. Existing original C1280 tail8 source paths are frozen in `research/intern-nextopt-20261025/prefill8-stack-manifest.json`; historically board bank `/opt/opi-vlm/intern-candidates-20261025/tail8`, reverify SHA after current lock releases before reusing.
