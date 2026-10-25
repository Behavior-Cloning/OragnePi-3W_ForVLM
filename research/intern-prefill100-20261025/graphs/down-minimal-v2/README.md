# Isolated down operator diagnostic

Source b0 sensitive down from the frozen decoder, physical query32, input4864/output896. FC and 1x1Conv, FP16 controls and I16 weight fractional lengths16/15/8; input/output quantization fixed from the earlier single-trajectory calibration. fl8 is deliberately coarse only to diagnose SDK support, not a deployable quality baseline. Same weight values before quantization, metadata and source hashes saved.

No system/dynamic tokenizer counts apply to this isolated graph, no board timings, no full-model latency claims. Full target remains fixed system64/128, dynamic body64/128/192/256 plus8 template tokens, only static KV reuse. Numeric node suffixes repaired after the first parser failure. Stop if either FP16 control fails.
