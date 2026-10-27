# 112 native16 NPU validation candidate

Independent run: original 2x2 spatial merge/projector, 112 input -> 16 visual embeddings. 24 ViT blocks INT16 FC with FP16 norm/GELU internals; stem/projector on NPU. Not CPU Q8 and not all-W8. Compile only is not validation.

`build_all.py` exports and compiles calibrated 26 graphs on C scratch; all model source hashes retained. `controller_tower.py` holds both board locks, verifies SHA, measures 1 warmup+6 hot requests, retrieves evidence and deletes only its own archived graph copies to free tmpfs before loading language. `controller_vqa.py` then feeds actual NPU tower outputs into Intern NPU language; this is sequential model residency, not JPEG-to-response simultaneous residency.

No defaults changed. Official architecture at adapted resolution does not imply official112 benchmark accuracy. Compare full vectors to FP32 reference and NPU language answers to WSL reference. No audio/RAG in these measurements.
