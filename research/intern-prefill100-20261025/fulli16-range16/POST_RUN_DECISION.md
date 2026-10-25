# Full-model result and next decision

Complete rc0 A-B-A,72 performance/72 natural/24 diagnostic requests. Same static64/128 (body59/123+5), dynamic64/128/192/256+8, output32fixed feedback. Candidate TTFT roughly33% lower; decode unchanged. Candidate natural text only2/24 exact; actual full-vocabulary differences are substantial, including recalled cancelled directions. This is not a qualified model. Await independent semantic grading.

Do not infer complete-model accuracy from the earlier~0.3% local hidden error: those48 windows used FP32-derived inputs/history. Next diagnostic captures actual grouped NPU layer2/3 input/output, including system construction and real history, then evaluates range saturation and standalone replay. Captures are diagnostic only, not speed data; production runtime remains unchanged.

Correction: all72 speed requests used padded M16 calls for final8 real rows, zero scalar prompt steps. True8 is future padded-row savings. Existing resident-combined-v2 true8 is capacity256-only; requires an isolated extension for512/1280.

Capacity512 compilation is allowed as a separate capacity-only artifact, not quality approval of I16. No production replacement or physical actions.
