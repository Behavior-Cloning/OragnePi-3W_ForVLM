# Context and RAG module

Implemented `context_rag.py` in Windows authoritative repository.

- `Context.add(text, at)` and `snapshot(now)` use one monotonic clock, preserve a 60 second rolling text deque; no raw audio or inferred acoustic semantics. Capacity 512 completed ASR rows, reports capacity drops.
- `compress(rows, encode, budget=64)` runs only on demand. Real supplied tokenizer counts include separators. Selects whole original clauses, newest request first then negation/correction, then recent context; exact repeats deduplicate. Output is chronological. Overlong whole clauses are omitted, never suffix-truncated; dropped critical clauses and latest-clause loss are explicit. This is lossy and not a learned semantic summary.
- `Retriever()` loads resident BGE-small Chinese transformer once, indexes 128 v2 atom retrieval texts once. FP32 CLS+L2, cosine scores, one torch thread. Real semantic retrieval; scores do not mean action compatibility or permission. All results are mock-only and hardware disabled.
- Board model default `/opt/opi-vlm/board-rag-atoms-20261021/model`. Runtime needs existing Jev venv torch/transformers.
- Transfer `research/home-companion-atoms128-v2-20261022/atoms.json` to same board directory as context_rag.py. Catalog IDs R001–R128 and namespace are checked. No skills or old action catalog accepted.
- `retriever.metadata` includes load/index seconds, catalog hash, parameter count and index bytes. `retriever.last_query` includes encoding/search/CPU seconds and query tokens.
- No board access from module author. Parent controller owns board locks.

Logic checks:

```json
{
  "syntax": "passed",
  "expire_window": "passed",
  "cancel_negation_preserved_when_fit": "passed",
  "overlong_clause_never_suffix_truncated": "passed",
  "exact_duplicate_keeps_latest": "passed",
  "note": "Character encoder tests logic only; actual Intern tokenizer count is runtime-measured"
}
```
