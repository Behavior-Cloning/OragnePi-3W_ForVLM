---
license: other
license_name: qwen-research
license_link: LICENSE
base_model: Qwen/Qwen2.5-Omni-3B
base_model_relation: adapter
library_name: peft
pipeline_tag: audio-classification
language:
  - en
tags:
  - audio
  - audio-decision
  - qwen2_5_omni
  - lora
  - streaming
  - custom-code
  - research
inference: false
---

<div align="center">
<p align="center">
<img src="https://huggingface.co/TeenageWorks/Toddler-Jev/resolve/main/assets/toddler-jev-banner.png" alt="Toddler-Jev" width="90%" class="mx-auto block rounded-2xl" style="display: block; margin: 0 auto; border-radius: 24px;">
</p>
<p><strong>Decisions directly from audio.</strong></p>
<p>Teenage Works Private Limited &nbsp;·&nbsp; Version 1.0.0 &nbsp;·&nbsp; Open weights</p>
<p>
<a href="https://teenage.works/blog/toddler-jev">Project &amp; results</a> &nbsp;·&nbsp;
<a href="https://huggingface.co/Qwen/Qwen2.5-Omni-3B">Base model</a> &nbsp;·&nbsp;
<a href="https://huggingface.co/TeenageWorks/Toddler-Jev/blob/main/TRAINING.md">Training guide</a> &nbsp;·&nbsp;
<a href="https://huggingface.co/TeenageWorks/Toddler-Jev/blob/main/LICENSE">License</a>
</p>
</div>

---

## Overview

Toddler-Jev is an audio decision model. Give it **an audio recording, a question, and a list of possible answers**. It selects an answer and returns a probability for every candidate.

The question defines the task, and the candidates define the available decisions. The same model can select an intent from a customer recording, identify a sound, assess sentiment, or answer a multiple-choice question about spoken content. You supply the candidate set for each request rather than using one fixed list of class labels.

Toddler-Jev processes audio through a multimodal backbone and scores candidates with a trained decision head. It does not require a separate transcription stage or generate free-form answer tokens. The supplied Python runtime supports individual recordings, batches, and overlapping windows of an audio stream.

| At a glance | |
|---|---|
| **Input** | Audio + text question + text candidates |
| **Output** | Selected candidate, candidate index, probability distribution |
| **Base** | [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) |
| **Adaptation** | Rank-16 attention LoRA + dynamic candidate decision head |
| **Trained parameters** | 8,421,888 |
| **Audio processing** | Mono, 16 kHz; automatic resampling for files and batch arrays |
| **Streaming default** | 8-second window, 1-second step |
| **Language focus** | English |
| **Execution** | Local Python or CLI; downloads cached for offline use |
| **License** | [Qwen Research License](LICENSE), for research/evaluation use |

## Use cases

The following are application patterns to explore with task-specific evaluation and, where needed, additional fine-tuning.

| Application | Example question | Example candidates |
|---|---|---|
| **Spoken intent classification** | What is the customer asking about? | Card payment · Cash withdrawal · Account balance |
| **Sentiment analysis** | What is the speaker's sentiment? | Negative · Neutral · Positive |
| **Emotion recognition** | What emotion does the speaker express? | Joy · Sadness · Anger · Neutral |
| **Sound classification** | What kind of sound is present? | Speech · Music · Environmental sound |
| **Instrument identification** | Which instrument is playing? | Guitar · Piano · Violin |
| **Spoken question answering** | Which answer follows from the recording? | User-supplied answer choices |
| **Assistant routing** | Which action best matches the spoken request? | Search · Set a reminder · Ask a follow-up |
| **Continuous audio analysis** | What is happening in the current window? | Application-specific states |

Routing returns a candidate decision; your application implements any resulting action. For long-form questions, retain the recording context needed to answer them.

## Quick start

### 1. Download and install

Use Python 3.10 or newer and a PyTorch build appropriate for your hardware.

```bash
python -m pip install huggingface_hub
hf download TeenageWorks/Toddler-Jev --revision v1.0.0 --local-dir Toddler-Jev
cd Toddler-Jev
python -m pip install -r requirements.txt
```

The repository contains the trained adapter, decision head, processor, and runtime. The loader downloads the pinned Qwen base automatically on first use. Allow roughly 12 GB for the base download, plus runtime memory. Once the files and dependencies are cached, no hosted endpoint or API key is required.

### 2. Make a decision

```python
from toddler_jev import ToddlerJev

model = ToddlerJev.from_pretrained(".")

result = model.predict(
    audio="recording.wav",
    question="What is the speaker asking about?",
    options=["card payment", "cash withdrawal", "account balance"],
)
print(result)
```

Or run the CLI:

```bash
python inference.py --audio recording.wav \
  --question "What is the speaker asking about?" \
  --options "card payment" "cash withdrawal" "account balance"
```

### 3. Read the result

Illustrative output structure; these numbers are examples:

```json
{
  "model": "TeenageWorks/Toddler-Jev",
  "version": "1.0.0",
  "decision_index": 2,
  "decision": "account balance",
  "probabilities": {
    "card payment": 0.1,
    "cash withdrawal": 0.2,
    "account balance": 0.7
  },
  "calibrated": false,
  "audio_seconds": 5.0
}
```

`decision_index` is zero-based. Probabilities sum to approximately one across the supplied options and are **uncalibrated**: a value of 0.7 is not a measured guarantee of 70% correctness. Changing the candidate set changes the decision problem and its distribution.

## Batched inference

Each item can use a different question and candidate set:

```python
results = model.predict_batch([
    {
        "audio": "customer.wav",
        "question": "What is the customer asking about?",
        "options": ["card payment", "cash withdrawal", "account balance"],
    },
    {
        "audio": "sound.wav",
        "question": "What kind of sound is present?",
        "options": ["speech", "music", "environmental sound"],
    },
])
```

The CLI reads one request per line from UTF-8 JSONL:

```json
{"id":"clip-001","audio":"audio/clip.wav","question":"What is the sentiment?","options":["negative","neutral","positive"]}
```

```bash
python batch_inference.py --input requests.jsonl \
  --output predictions.jsonl --batch-size 4
```

Relative audio paths resolve beside the JSONL file. Results preserve input order, with one JSON record per request. Start with batch size 1 and increase it for your audio lengths and hardware.

## Streaming inference

Feed a contiguous iterator of 16-kHz float PCM chunks into `predict_stream`. The runtime maintains a rolling audio buffer and evaluates overlapping windows.

| Setting | Default | Behavior |
|---|---:|---|
| `window_seconds` | `8.0` | Amount of audio used for each decision |
| `hop_seconds` | `1.0` | Input time between successive decisions |
| `emit_partial` | `False` | Wait for a full window before the first result |

```python
import soundfile as sf

with sf.SoundFile("recording_16khz.wav") as audio:
    assert audio.samplerate == 16000
    chunks = audio.blocks(blocksize=1600, dtype="float32")

    for result in model.predict_stream(
        chunks,
        question="What is the speaker's sentiment?",
        options=["negative", "neutral", "positive"],
        window_seconds=8,
        hop_seconds=1,
    ):
        print(result["window_start_seconds"],
              result["window_end_seconds"],
              result["decision"])
```

```bash
python stream_inference.py --audio recording_16khz.wav \
  --question "What is the sentiment?" \
  --options negative neutral positive --window 8 --hop 1
```

The default windows are `[0, 8]`, `[1, 9]`, `[2, 10]`, and so on. A 500-ms step is available with `hop_seconds=0.5` or `--hop 0.5`. Partial startup windows can be enabled with `emit_partial=True` or `--emit-partial`. An incomplete final hop is not emitted.

Each window is recomputed; the implementation does not cache encoder state across windows. The CLI replays a file at the speed inference allows. Real-time throughput depends on your hardware and workload. The release was trained on complete clips, so evaluate windowed accuracy separately for your application. Questions requiring earlier context should use a longer window or the complete recording.

## Inference latency

Measured on a single **NVIDIA A100 40 GB**, using **BF16** and SDPA after warm-up.

| Workload | **p50** | **p90** | Timed samples |
|---|---:|---:|---:|
| Streaming update — 8-second window, 1-second step | **111.9 ms** | **112.9 ms** | 115 |
| Single 30-second audio recording | **123.5 ms** | **124.4 ms** | 20 |

These are CUDA-synchronized wall-clock timings with four CPU threads, including audio preprocessing, model execution, and result formatting for in-memory 16-kHz PCM. Audio collection, file decoding, network transport, and model loading are excluded. The released adapter is used without merging, quantization, or compilation.

Streaming values measure steady-state computation per window. The first full-window result also requires eight seconds of audio collection.

## Fine-tuning

Both training modes use the audio decision objective and save a model loadable through the same `ToddlerJev.from_pretrained()` interface.

| Mode | Updated components | Saved output |
|---|---|---|
| **LoRA continuation** | Released attention adapter and decision head | PEFT adapter + decision head |
| **Full fine-tuning** | Audio/language backbone and head after merging LoRA | Sharded Thinker weights + decision head |

Prepare labeled JSONL:

```json
{"audio":"audio/clip.wav","question":"What is the sentiment?","options":["negative","neutral","positive"],"label":"positive"}
```

`label` can be the exact candidate string or its zero-based index. Keep training and validation recordings separate.

**Continue LoRA training**

```bash
python train.py --train train.jsonl --validation validation.jsonl \
  --output ./my-toddler-jev --mode lora --epochs 1 \
  --batch-size 1 --gradient-accumulation 8 --gradient-checkpointing
```

**Full fine-tuning**

```bash
python train.py --train train.jsonl --validation validation.jsonl \
  --output ./my-full-toddler-jev --mode full --epochs 1 \
  --lr 1e-6 --head-lr 1e-5 --gradient-checkpointing
```

**Load your adapted model**

```python
adapted = ToddlerJev.from_pretrained("./my-toddler-jev")
result = adapted.predict("clip.wav", "What is the sentiment?",
                         ["negative", "neutral", "positive"])
```

Full training requires substantially more memory than LoRA. The reference trainer runs on one device; the unused vision encoder remains frozen. See [TRAINING.md](TRAINING.md) for optimization settings, dataset preparation, window labeling, and saved model formats.

To evaluate your own labeled data:

```bash
python evaluate.py --data test.jsonl --output test-predictions.jsonl --batch-size 1
```

## Evaluation results

Toddler-Jev benchmark campaign scores, measured before the final training continuation. Bold values identify Toddler-Jev's results.

### Native audio benchmarks

| Benchmark | Examples | **Toddler-Jev** |
|---|---:|---:|
| MMAU-mini | 1,000 | **68.70%** |
| MMAR | 1,000 | **56.90%** |
| MMSU | 5,000 | **61.30%** |

### Spoken-input decision benchmarks

| Benchmark | Examples | **Toddler-Jev** |
|---|---:|---:|
| Agentsky Easy | 48 | **100.00%** |
| Agentsky Hard | 111 | **41.44%** |
| Agentsky Original | 72 | **79.17%** |
| DecideBench | 400 | **74.50%** |
| Aegis2 prompt safety | 500 | **72.00%** |
| Aegis2 response safety | 500 | **65.20%** |
| Banking77 | 1,000 | **81.50%** |
| Jailbreak classification | 400 | **74.50%** |
| MedMCQA | 1,000 | **41.90%** |
| MedQA-USMLE | 1,000 | **44.70%** |
| MMLU-Pro | 1,000 | **36.50%** |
| Prompt injections | 116 | **51.72%** |
| PubMedQA | 500 | **58.40%** |
| ScienceQA (spoken text subset) | 1,000 | **84.00%** |
| SST-5 | 500 | **56.00%** |

The spoken-input suites use **TeenageWorks-synthesized audio** for the designated source inputs. Questions and candidate options remain text. The tables report full-recording accuracy; streaming-window accuracy is a separate measurement. Published text-model comparisons use a different input modality.

Read the [project and results blog](https://teenage.works/blog/toddler-jev) or inspect [evaluation_results.json](evaluation_results.json).

## How the model works

1. **Encode the recording and task.** The Qwen Thinker processes the audio together with the text question and marked candidate strings.
2. **Represent each candidate.** The runtime extracts contextual representations at candidate boundaries and at a decision marker.
3. **Score the candidates.** Two trained linear projections map candidate and decision representations into 256 dimensions. Scaled dot products and softmax produce the probability distribution.

This allows candidate counts and labels to change between requests. The original vocabulary output head is replaced by the decision path; answer-token generation is not used.

| Component | Release configuration |
|---|---|
| Base revision | [`f75b40e`](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/tree/f75b40e3da2003cdd6e1829b1f420ca70797c34e) |
| LoRA targets | Language attention query, key, value, and output projections |
| LoRA rank / alpha / dropout | `16` / `32` / `0.05` |
| Decision projection size | `256` |
| Backbone hidden size | `2,048` |
| Trained parameter count | `8,421,888` |
| Release training precision | BF16 |
| Release training objective | Hard-label cross-entropy |

### Training data

Release training used real audio from **NSynth, NonSpeech7k, and MELD**, plus **20,000 TeenageWorks-synthesized audio examples** from 13 text datasets. Only the designated source input was synthesized. Questions, candidate lists, output decisions, and probabilities were kept as text.

The text sources were Banking77, SST-5, ARC, CommonsenseQA, OpenBookQA, BoolQ, MNLI, TREC, AG News, DBpedia-14, IMDb, Amazon reviews, and Yelp reviews.

Across initial training, continuation, and focused replay, the release accumulated **110,279 training presentations over 13,785 optimizer steps**. Presentations include repeated examples and are not a count of unique recordings. Audio and vision encoders stayed frozen while the language attention adapter and decision head were trained.

## Input and hardware guidance

| Item | Guidance |
|---|---|
| Candidate set | At least two unique, nonempty strings |
| Audio files | SoundFile-readable formats, with mono conversion and resampling |
| Batch arrays | NumPy float PCM in `audio`, with `sampling_rate`; channels last |
| Stream chunks | Contiguous 16-kHz float PCM; mono or channels last |
| Recording length | Configured maximum 1,200 seconds, also subject to context and memory limits |
| CUDA execution | BF16-capable GPU |
| CPU execution | FP32; more host memory and slower inference |
| Memory pressure | Lower batch size and use task-appropriate audio durations |

The loader raises an error for oversized inputs instead of silently truncating them. Loading only the PEFT adapter does not restore the complete decision model; use the supplied runtime to load the decision head as well. Calls on one instance are serialized because the upstream model caches positional state.

The tested GPU environment used Linux, an A100 40 GB, PyTorch 2.9.1/CUDA 12.9, Transformers 4.57.1, and PEFT 0.17.1. A 1,200-second input limit does not imply that every 20-minute recording fits on every supported device.

## Repository guide

| File | Purpose |
|---|---|
| `adapter_model.safetensors` | Trained attention LoRA weights |
| `adapter_config.json` | PEFT configuration and pinned base |
| `decision_head.safetensors` | Trained candidate-scoring head |
| `decision_config.json` | Architecture and input settings |
| `processor/` | Tokenizer, chat template, and processor assets |
| `toddler_jev.py` | Python inference, batch, stream, and differentiable forward interface |
| `inference.py` | Single-recording CLI |
| `batch_inference.py` | JSONL batch CLI |
| `stream_inference.py` | Sliding-window PCM CLI |
| `train.py` | LoRA continuation and full fine-tuning |
| `evaluate.py` | Local labeled-data evaluation |
| `TRAINING.md` | Training guide |
| `evaluation_results.json` | Benchmark campaign metrics |
| `SHA256SUMS.json` | Release file checksums |

## Scope

Toddler-Jev is an English-focused research release for audio-conditioned candidate selection. Its output depends on the available candidates, their wording, audio quality, and retained context. The probabilities are relative to that candidate set and are not calibrated confidence estimates. Evaluate both accuracy and latency on your own application data.

## Researcher and organization

**Researcher:** Vyom Nishant Patel  
**Organization:** Teenage Works Private Limited  
**Contact:** [vyom@teenage.works](mailto:vyom@teenage.works)

**Registered office:** Awfis, 1st Floor, Parinee Cresenzo, C-38-39, G Block, Bandra Kurla Complex, Bandra East, Mumbai, Maharashtra 400051, India.  
**CIN:** U62011MH2026PTC469285. [Company and registered-office listing](https://www.zaubacorp.com/TEENAGE-WORKS-PRIVATE-LIMITED-U62011MH2026PTC469285).

[Project and results](https://teenage.works/blog/toddler-jev) · [Base model](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) · [Citation](CITATION.cff) · [Release history](CHANGELOG.md)

## License and attribution

**Improved using Qwen.**

The [Qwen Research License](LICENSE) governs this derivative release and permits research/evaluation use. Commercial use requires the permissions specified in that license. Attribution and modification notices are retained in [NOTICE](NOTICE). Dataset licenses are separate; datasets and audio are not redistributed in this repository.

## How to cite

If you use Toddler-Jev in your work, cite the model release:

> Patel, V. N. (2026). *Toddler-Jev* (Version 1.0.0) [Model]. Teenage Works Private Limited. https://huggingface.co/TeenageWorks/Toddler-Jev

```bibtex
@misc{patel2026toddlerjev,
  author       = {Patel, Vyom Nishant},
  title        = {Toddler-Jev},
  year         = {2026},
  version      = {1.0.0},
  publisher    = {Teenage Works Private Limited},
  organization = {Teenage Works Private Limited},
  address      = {Awfis, 1st Floor, Parinee Cresenzo, C-38-39, G Block,
                  Bandra Kurla Complex, Bandra East, Mumbai,
                  Maharashtra 400051, India},
  howpublished = {Teenage Works Private Limited, model release},
  url          = {https://huggingface.co/TeenageWorks/Toddler-Jev},
  note         = {Version 1.0.0}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).
