# Small Chinese ASR board experiment

Windows/WSL is the source of truth. Board workspace: `/opt/opi-vlm/asr-small-20261018`.
All model inference uses ARM64 CPU / sherpa-onnx 1.13.8, one inference thread; no cloud inference API and no microphone recording.
Runtime is shared read-only from `/opt/opi-vlm/audio-lab-20261018/deps`.

Run from WSL repository root:

```sh
python3 research/asr-small/run_wsl.py --mode standalone --name standalone-v2
python3 research/asr-small/run_wsl.py --mode paced --name paced-v2
python3 research/asr-small/run_wsl.py --mode mixed --name mixed-d64-v1 --dynamic 64
python3 research/asr-small/run_wsl.py --mode mixed --name mixed-d256-v1 --dynamic 256
python3 research/asr-small/summarize.py
```

Use unique result names. `--deploy` transfers model assets and corpus from F disk and verifies file SHA-256. Calls acquire the Windows coordinator lock and board `flock`; intentional child concurrency has one coordinator. Scripts/results are retained under `results/asr-small-20261018`.

ASR affinity is CPU2,3, one inference thread. NPU LLM host affinity is CPU6,7. Mixed trials stress ASR repeatedly rather than pretending continuous speech production load. Two trials per condition, reversed order, P64/D64 or P64/D256. Check actual response temperature and overlap before interpreting relative changes.

Paced mode is an 80 ms wall-clock single-utterance replay with 2 s appended silence, not a microphone/live conversation service. It preserves the utterance stream across internal endpoints to keep its final transcript comparable. Endpoint transitions, partial changes, compute time, arrival lag, and forced finish are separate. File end is not a manually labeled speech end. Retrospective stable text is not a causal early-finalization policy.

CER removes whitespace/punctuation/control characters only. It does not normalize numbers, names, negation, or simplified/traditional variants. A literal critical-string mismatch is a screening signal, not automatically a semantic error. Read the references and hypotheses. AISHELL test subset (24 utterances, six speakers) and synthetic noise cannot prove home robot accuracy.

`cleanup_chunks.py` is an explicit maintenance action, not part of benchmark setup. It compares each 8 MiB-numbered transfer chunk against the retained GGUF byte range and only then unlinks an identical duplicate. Never run unrelated cleanup during timing trials.

Qwen screening is a fixed 14-input subset, single pass; compare CER only on the same 11 clean utterances (190 reference characters). Run `compare_common.py` after Qwen results. Qwen mixed runs use their own matched baseline and the same CPU2,3 affinity. The separate CPU6,7 three-utterance probe is standalone only.
