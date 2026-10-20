---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
language:
  - en
pipeline_tag: text-classification
library_name: transformers
tags:
  - jev
  - system-one
  - open-jev
  - decision-model
  - calibrated
  - routing
  - classification
  - guardrails
  - non-autoregressive
---

# Tiny-Jev

**A 0.6B "System One" decision model: structured state in, typed probabilistic decisions out.**

> **Not affiliated with TypeSafe AI.** Tiny-Jev is an independent, open-weights model *inspired by* the System One idea and the Jev decision API published by TypeSafe AI. It is not built, endorsed, or supported by them, shares no code or weights with their models, and "Jev" in the name refers only to the interface style it reproduces (Choice / Score / Noul with calibrated probabilities).

A larger, more accurate sibling is available as [Tiny-Jev-1.7B](https://huggingface.co/lostargon/Tiny-Jev-1.7B).

Tiny-Jev does not generate text. It reads a *state* (a message, a ticket, a JSON record, a transcript, a log, a diff), a *question* written the way a developer would write it in code, and a fixed set of *options* — and returns a probability distribution over those options in a single forward pass. Every answer comes with a calibrated confidence, so your code can act on the confident ones and route the rest.

Three primitives, one call:

| primitive | returns | example |
|---|---|---|
| **Choice** | the best option + a distribution over all of them | *Which team should handle this?* → billing / technical / sales |
| **Score** | a position on an ordered scale (expected value, can land between levels) | *How frustrated is the customer?* → calm … hostile |
| **Noul** | a probability that a statement about the state is true | *The customer explicitly requests a refund* → 0.97 |

It is meant to be the smart `if` statement in your pipeline: routing, triage, filtering before an expensive context window, scoring or guard-railing another model's output, moderation, tagging at volume, sub-100 ms decisions inside a request handler. On a consumer GPU a call takes a few milliseconds; on an Apple M-series laptop ~20–50 ms.

## Quick start

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("lostargon/Tiny-Jev")
model = AutoModel.from_pretrained("lostargon/Tiny-Jev", trust_remote_code=True).eval()   # .to("cuda") / .to("mps")

state = {"message": "My card was charged twice and nobody answers the phone.", "plan": "pro"}

model.choice(tok, state, "Which team should handle this",
             {"billing": "Payment or subscription issues", "technical": "Bugs or integration problems", "sales": "Plans and pricing"})
# {'choice': 'billing', 'probabilities': {'billing': 0.96, 'technical': 0.03, 'sales': 0.01}, 'confidence': 0.96}

model.noul(tok, state, "The customer explicitly requests a refund")        # 0.12
model.score(tok, state, "How frustrated is the customer", ["calm", "annoyed", "angry"])
# {'score': 1.7, 'probabilities': {...}, 'confidence': 0.71}

# Fan-out: several questions over one state in a single batched call
model.decide(tok, state, [
    {"kind": "choice", "instructions": "Which team should handle this", "criteria": ["billing", "technical", "sales"]},
    {"kind": "noul",   "instructions": "The customer is reporting a bug"},
    {"kind": "score",  "instructions": "Urgency", "criteria": ["low", "normal", "high", "urgent"]},
])
```

The whole model — decoder stack and decision head — is one `safetensors` file. `trust_remote_code=True` loads the 150-line `modeling_tiny_jev.py` shipped in this repo (no other dependencies beyond `torch` and `transformers`).

## How it works

Each option is rendered as its own line after the state and the question, followed by a marker token. The hidden state at every marker goes through a small linear head to a scalar; a softmax over the option scalars is the answer. Noul is a two-option Choice (`no` / `yes`), Score is a Choice over ordered levels whose expected value is reported. The model was trained with a soft cross-entropy objective against probability-vector targets and then temperature-calibrated on a held-out split, so the confidence it reports is meant to be honest rather than flattering.

Because the answer space is the option list you pass in, the output can never be malformed: no parsing, no retries, no schema errors.

## What it was trained on

Tiny-Jev was fine-tuned from Qwen3-0.6B (LoRA, merged into the weights) on **synthetic datasets** of typed decisions — roughly a hundred thousand states, each paired with several questions, with labels derived from explicit rules or computed by code so that every target is exact. The task families cover what such a model is used for in practice:

- **Support and operations** — ticket routing and categorisation, refund / cancellation / escalation intents, severity and churn scoring, email triage, invoice processing decisions, security-incident dispositions, workflow policies applied to facts.
- **Guard-railing other models** — is an assistant's answer grounded in the provided context, on topic, policy-compliant, what kind of failure it exhibits, should it be escalated; whether an autonomous agent's proposed action (shell command, file write, API call) is safe, reversible, in scope, or leaks a secret.
- **Content and text** — moderation categories and severity, review sentiment and aspects, fake-review signals, search intent and page relevance, code-review verdicts on real diffs.
- **Voice and dialogue** — turn-taking (has the caller finished?), addressee, voicemail detection, answer type, frustration, next bot action, from noisy ASR transcripts.
- **Structured reading** — log analysis, schedules and time zones, inventory levels, access-control rules, state-machine transitions, dependency graphs, rankings with tie rules, format validation, pricing with discounts and tax, retrieval passages, citation support.
- **Robustness sets** — contrastive negation and scoping (*asks for a refund* / *does not ask* / *asks only for a refund*), prompt-injection attempts embedded in the state that must not change the label, and Russian-language states with English questions.

Every domain includes a large share of deliberately hard cases (negations, near-miss categories, buried evidence, mixed signals), and label rules were audited by blind re-labelling before the data was used.

## Evaluation

All numbers are accuracy on held-out items; **ECE** is expected calibration error (lower is better; 0.01 means the stated confidence is off by about one point on average). *acc@0.9* is accuracy on the subset of answers where the model's confidence is at least 0.9, with *cov@0.9* the share of items that subset covers — the pair that matters when you gate actions on confidence.

![Calibration](assets/calibration.png)

### Held-out splits of the training distribution

| split | items | accuracy | ECE | acc@0.9 | cov@0.9 |
|---|---|---|---|---|---|
| test (same task families) | 22 110 | **95.8** | **0.004** | 99.8 | 90 % |
| OOD (held-out template and dialogue families) | 48 021 | **93.5** | 0.033 | 96.4 | 91 % |
| hand-written business domains, test | 152 | 71.1 | 0.191 | 82.5 | 68 % |
| two fully held-out domains (code review, search intent) | 830 | 53.1 | 0.299 | 67.6 | 54 % |

### Public benchmarks

Item lists follow the open **open-system-one** protocol (2 500 items per dataset, fixed seed), so the reference row is on exactly the same items. Tiny-Jev was additionally fitted on 2 000 items per dataset that are disjoint from the evaluation items — the same allowance that protocol gives its fitted open baselines; the reference model is zero-shot, so read the comparison as *"a 0.6B model plus a small fit set"* versus *"a frontier decision API with no fit set"*.

![Benchmarks](assets/benchmarks.png)

| dataset | options | Tiny-Jev acc | ECE | acc@0.9 | cov@0.9 | reference (zero-shot) |
|---|---|---|---|---|---|---|
| SST-2 | 2 | **90.4** | 0.023 | 96.4 | 77 % | 91.6 |
| AG News | 4 | **90.5** | 0.031 | 95.7 | 79 % | 88.6 |
| Emotion | 6 | **82.2** | 0.054 | 94.2 | 64 % | 59.0 |
| BANKING77 | 77 | **81.2** | 0.036 | 95.8 | 61 % | 77.8 |
| CLINC150 (+ out-of-scope)† | 151 | 61.1 | 0.074 | 92.2 | 32 % | — |

† not in the fit set: transfer from the other intent tasks only.

### Zero-shot multiple choice

No training on any of these tasks; 500 items each, options passed as a Choice question. They measure general knowledge and arithmetic, which a model this size only partly has.

![Zero-shot](assets/zero_shot.png)

### Reading the numbers

- On its own distribution the model is both accurate and honest: at ≥0.9 confidence it is right 99.8 % of the time and reaches that confidence on 90 % of items.
- On public benchmarks the calibration holds (ECE 0.02–0.07), which is the property that makes confidence-gated routing work.
- The two fully held-out domains show the limit of a 0.6B model: new rule systems it has never seen are handled at ~50–60 %, with confidence that correctly drops.

## Limitations

- **Not a text generator.** It cannot write, summarise, or explain its answer. Extract candidates with code or another model; use Tiny-Jev to pick.
- **Reads questions literally.** Negations and scoping words are taken at face value — that is by design, but phrase questions the way you would phrase a unit test.
- **No arithmetic, weak on dates.** Counting and date comparisons are handled only as far as the structured-reading training goes; compute exact things in code.
- **Confidence is calibrated on its training distribution.** On very short, out-of-context prompts it can be over-confident; keep states realistic and give it the fields it needs.
- **Unarmoured state.** Content inside the state can try to argue for its own label. The model was trained against this, but treat user-controlled text with the usual care.
- Context up to 4 096 tokens per question; the state is truncated (head and tail kept) when it does not fit alongside the options.

## Model details

| | |
|---|---|
| Base | Qwen3-0.6B (decoder stack; the language-model head is not used) |
| Added | one marker token, a 1-dimensional decision head, temperature |
| Parameters | 596 M |
| Precision | bfloat16 weights, float32 head math |
| Training | LoRA r=32 on all linear layers + head, soft cross-entropy, one epoch, post-hoc temperature calibration |
| License | Apache-2.0 |

If you use Tiny-Jev, a link back is appreciated. Issues and results on your own data are welcome in the community tab.
