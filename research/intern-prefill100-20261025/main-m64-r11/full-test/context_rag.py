"""60-second text context, on-demand extractive compression and real BGE retrieval.
No audio waveform or invented acoustic semantics are retained here.
Retrieval produces MOCK candidates, never permission to actuate hardware.
"""
from collections import deque
from pathlib import Path
import hashlib
import json
import os
import re
import threading
import time

class Context:
    def __init__(self, seconds=60.0, max_rows=512):
        self.seconds = float(seconds)
        self.max_rows = int(max_rows)
        self.rows = deque()
        self.lock = threading.RLock()
        self.capacity_drops = 0

    def add(self, text, at):
        text = str(text).strip()
        if not text:
            return
        row = {'text': text, 'at': float(at)}
        with self.lock:
            if self.rows and row['at'] < self.rows[-1]['at']:
                self.rows = deque(sorted([*self.rows, row], key=lambda x: x['at']))
            else:
                self.rows.append(row)
            self._expire(max(x['at'] for x in self.rows))
            while len(self.rows) > self.max_rows:
                self.rows.popleft()
                self.capacity_drops += 1

    def _expire(self, now):
        while self.rows and self.rows[0]['at'] < now - self.seconds:
            self.rows.popleft()

    def snapshot(self, now):
        with self.lock:
            self._expire(float(now))
            return [dict(x) for x in self.rows if x['at'] <= now]

CRITICAL = re.compile(r'不|别|取消|停止|禁止|改成|改为|算了|等等|等一下|先等')
CLAUSES = re.compile(r'[^。！？!?；;\n]+[。！？!?；;]?')

def compress(rows, encode, budget=64):
    """Select complete verbatim clauses. Cost includes tokenization/selection.

    Priorities: newest clause, then negation/correction clauses newest-first,
    then remaining clauses newest-first. Output retains original time order.
    Whole clauses that cannot fit are omitted and explicitly accounted for;
    we NEVER cut a clause's prefix, which could strip a negation.
    This is a lossy extractive baseline, not a semantic entailment model.
    """
    t0 = time.perf_counter()
    budget = int(budget)
    if budget < 1:
        raise ValueError('budget must be positive')
    rows = sorted(rows, key=lambda x: x['at'])
    clauses = []
    for row in rows:
        for m in CLAUSES.finditer(row['text']):
            text = m.group(0).strip()
            if text:
                clauses.append({'text': text, 'at': row['at'], 'critical': bool(CRITICAL.search(text))})
    # Exact repeats keep their newest occurrence; no embedding or fuzzy deletion.
    seen = set(); keep = []
    for item in reversed(clauses):
        key = item['text'].rstrip('。！？!?；; ').strip()
        if key not in seen:
            seen.add(key); keep.append(item)
    clauses = list(reversed(keep))
    original = '\n'.join(r['text'] for r in rows)
    original_tokens = len(encode(original)) if original else 0
    selected = set()
    order = ([len(clauses)-1] if clauses else [])
    order += [i for i in range(len(clauses)-2, -1, -1) if clauses[i]['critical']]
    order += [i for i in range(len(clauses)-2, -1, -1) if not clauses[i]['critical']]
    def render(indices):
        return '；'.join(clauses[i]['text'].rstrip('；;') for i in sorted(indices))
    for i in order:
        candidate = selected | {i}
        if len(encode(render(candidate))) <= budget:
            selected = candidate
    text = render(selected)
    dropped = [i for i in range(len(clauses)) if i not in selected]
    return {'text': text, 'tokens': len(encode(text)) if text else 0,
            'budget': budget, 'input_tokens': original_tokens, 'input_rows': len(rows),
            'unique_clauses': len(clauses), 'selected_clauses': len(selected),
            'dropped_clauses': len(dropped),
            'dropped_critical_clauses': sum(clauses[i]['critical'] for i in dropped),
            'latest_clause_kept': bool(clauses and len(clauses)-1 in selected),
            'selected_at': [clauses[i]['at'] for i in sorted(selected)],
            'method': 'verbatim_clause_priority_v1', 'chronological': True,
            'lossy': bool(dropped), 'elapsed_s': time.perf_counter()-t0,
            'sound_semantics': 'not_inferred'}

class Retriever:
    """Resident real BGE-small-zh-v1.5 CLS embeddings of the 128 v2 atoms.

    Query-time scores are cosine similarities, not calibrated probabilities
    or a claim that top-k actions are compatible/physically executable.
    """
    def __init__(self, model_path=None, catalog_path=None, threads=1):
        t0 = time.perf_counter()
        import numpy as np
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.np, self.torch = np, torch
        self.lock = threading.Lock()
        torch.set_num_threads(int(threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # Another resident component may have initialized this pool.
        self.model_path = Path(model_path or os.environ.get('OPI_RAG_MODEL', '/opt/opi-vlm/board-rag-atoms-20261021/model'))
        self.catalog_path = Path(catalog_path or os.environ.get('OPI_ATOMS_CATALOG', str(Path(__file__).with_name('atoms.json'))))
        raw = self.catalog_path.read_bytes()
        self.cards = json.loads(raw)
        expected = {f'R{i:03d}' for i in range(1,129)}
        assert len(self.cards) == 128 and {x['id'] for x in self.cards} == expected, 'Requires latest 128 v2 atom catalog'
        assert all(x['type']=='atom' and x['namespace']=='home_companion_atoms128_v2' for x in self.cards)
        assert all(x.get('enabled_on_board') is False for x in self.cards), 'Harness is mock-only'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True, torch_dtype=torch.float32).eval()
        self.docs = [x['retrieval_text'] for x in self.cards]
        loaded = time.perf_counter()
        vectors = []
        for i in range(0,len(self.docs),4):
            v,_ = self._encode(self.docs[i:i+4]);vectors.append(v)
        self.vectors = np.ascontiguousarray(np.concatenate(vectors))
        self.metadata = {'model_path':str(self.model_path), 'catalog_sha256':hashlib.sha256(raw).hexdigest(),
            'atoms':128, 'skills':0, 'dtype':'float32', 'threads':int(threads),
            'load_s':loaded-t0, 'index_s':time.perf_counter()-loaded,
            'parameters':sum(p.numel() for p in self.model.parameters()),
            'index_bytes':self.vectors.nbytes, 'query_prefix':'为这个句子生成表示以用于检索相关文章：',
            'scoring':'CLS L2 normalized cosine; not action compatibility probability',
            'hardware_execution':False}
        self.last_query = None

    def _encode(self, texts):
        batch = self.tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
        length = int(batch['input_ids'].shape[1])
        if length > 512:
            raise ValueError(f'Embedding input {length} tokens exceeds 512; refusing silent truncation')
        with self.torch.inference_mode():
            states = self.model(**batch).last_hidden_state[:,0]
            vectors = self.torch.nn.functional.normalize(states,p=2,dim=1).float().numpy()
        return vectors,length

    def query(self, text, k=3):
        with self.lock:
            t0 = time.perf_counter();cpu0 = time.process_time()
            text = str(text).strip()
            if not text:
                self.last_query = {'empty':True,'total_s':time.perf_counter()-t0}
                return []
            vec,length = self._encode(['为这个句子生成表示以用于检索相关文章：'+text])
            t1 = time.perf_counter();scores = self.vectors @ vec[0]
            ids = self.np.argsort(-scores,kind='stable')[:max(0,min(int(k),128))]
            result = []
            for i in ids:
                card = self.cards[int(i)]
                result.append({'id':card['id'],'text':card['retrieval_text'],'name':card['name'],
                    'score':float(scores[i]),'commanded_channels':card['commanded_channels'],
                    'resource_locks':card['resource_locks'],'enabled_on_board':False,
                    'compatibility':'unadjudicated','mock_only':True})
            self.last_query = {'tokens':length, 'encode_s':t1-t0,
                'search_s':time.perf_counter()-t1, 'total_s':time.perf_counter()-t0,
                'cpu_s':time.process_time()-cpu0,'candidates':128}
            return result
