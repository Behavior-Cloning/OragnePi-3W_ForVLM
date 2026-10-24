"""Freeze aligned Intern text fixtures on Windows/WSL. Does not access the board.
Run: /opt/opi-lab/export-venv/bin/python freeze_aligned_inputs.py --tokenizer models/tokenizer-board.json
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
from importlib.metadata import version
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokenizer', type=Path, default=ROOT/'models/tokenizer-board.json',
                    help='Explicit tokenizer JSON; default is the exported board runtime tokenizer')
args = parser.parse_args()
TOKENIZER = args.tokenizer.resolve()
tokenizer = Tokenizer.from_file(str(TOKENIZER))
def enc(text):
    return tokenizer.encode(text, add_special_tokens=False).ids
def token_hash(ids):
    return hashlib.sha256(struct.pack('<' + str(len(ids)) + 'I', *ids)).hexdigest()
def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
HEADER = '<|im_start|>system\n'
FOOTER = '<|im_end|>\n'
USER = '<|im_start|>user\n'
ASSISTANT = '<|im_end|>\n<|im_start|>assistant\n'
CORE = '你是家庭陪伴机器人的动作规划器。根据情景只输出一个动作短语。可动头部、双臂、腰、底盘；无手指、屏幕。遵守取消和禁止要求，不拿物品，不回答问题。不确定则保持静止。请不要输出解释。'
SYSTEM_TEMPLATE_TOKENS = len(enc(HEADER + CORE + FOOTER)) - len(enc(CORE))
assert SYSTEM_TEMPLATE_TOKENS > 0
assert len(enc(CORE)) <= 64 - SYSTEM_TEMPLATE_TOKENS, ('core does not fit smallest system', len(enc(CORE)))

def pad_to(text, target):
    original = text
    initial = len(enc(text))
    assert initial <= target, (initial, target, text)
    for _ in range(512):
        n = len(enc(text))
        if n == target:
            return text, {'strategy': 'none' if text == original else 'trailing_full_stop_cost_padding',
                          'original_text': original, 'original_tokens': initial,
                          'added_text': text[len(original):], 'added_token_count': n - initial,
                          'cost_fixture': text != original}
        if n > target:
            raise ValueError('Punctuation tokenization overshot exact budget')
        text += '。'
    raise ValueError('Cannot fill requested token budget')

systems = []
for target in (64, 128):
    # Keep precisely the same core rules. The long system repeats those rules;
    # added punctuation is explicitly tagged as a cost fixture, not new policy.
    seed = CORE; repetitions = 1
    while len(enc(seed + CORE)) <= target - SYSTEM_TEMPLATE_TOKENS:
        seed += CORE; repetitions += 1
    body, padding = pad_to(seed, target - SYSTEM_TEMPLATE_TOKENS)
    prefix = HEADER + body + FOOTER
    ids = enc(prefix)
    assert len(ids) == target
    assert len(ids) - len(enc(body)) == SYSTEM_TEMPLATE_TOKENS
    systems.append({'id': f'system-{target}', 'static_tokens': target,
                    'core_text': CORE, 'core_text_tokens': len(enc(CORE)),
                    'system_body': body, 'system_body_tokens': len(enc(body)),
                    'system_template_tokens': SYSTEM_TEMPLATE_TOKENS, 'prefix': prefix, 'prefix_tokens': ids,
                    'prefix_sha256': token_hash(ids), 'prefix_text_sha256': text_hash(prefix),
                    'static_padding': {'core_repetitions': repetitions,
                                       'added_constraints': False, **padding},
                    'cross_static_quality_equivalence': 'not assumed; duplicated instructions can alter output'})

candidates = {
 32: [
  ['用户刚回到家，有些疲惫，希望机器人点头回应；桌边空间有限，底盘保持静止。用户还在轻声讲话，暂时不要打断他。',
   '用户刚回到家，有些疲惫，希望机器人点头回应；桌边空间有限，底盘保持静止。用户还在讲话，不要打断。'],
  ['用户正在阅读，希望保持安静；已取消向前移动的要求，请保持静止，不要再靠近。书本还放在桌上，请不要碰它。',
   '用户正在阅读，希望保持安静；已取消向前移动的要求，请保持静止，不要再靠近。书本还放在桌上，不要碰它。'],
  ['用户想和机器人打招呼，希望挥动手臂回应；桌上摆着水杯，底盘保持静止，不要碰杯子。请动作轻缓。']
 ],
 64: [
  ['用户刚回到家，有些疲惫，希望机器人点头回应；桌边空间有限，底盘保持静止。用户还在讲话，不要打断。刚才提过让机器人靠近，但随后取消了移动要求。现在不需要播放音乐，也不需要整理桌面，只希望得到一个安静的点头回应。'],
  ['用户正在阅读，希望保持安静；已取消向前移动的要求，请保持静止，不要再靠近。书本还放在桌上，不要碰它。刚才用户提到稍后可能聊天，目前还没有发出新的动作要求。机器人应保持当前位置，等待用户读完这一段，不要擅自转身。'],
  ['用户想和机器人打招呼，希望挥动手臂回应；桌上摆着水杯，底盘保持静止，不要碰杯子。请动作轻缓。用户没有要求机器人拿取物品，也没有要求移动到桌子另一侧。可以用手臂做一次招呼，不需要走动，更不要碰到旁边的书。']
 ]
}
scenario_ids = ['tired-greeting-no-base-motion', 'reading-cancel-motion', 'wave-no-base-motion']
prepared = {}
for budget, groups in candidates.items():
    prepared[budget] = []
    for scenario, choices in zip(scenario_ids, groups):
        legal = [text for text in choices if len(enc(text)) <= budget]
        assert legal, (budget, scenario)
        seed = max(legal, key=lambda text: len(enc(text)))
        body, padding = pad_to(seed, budget)
        prepared[budget].append({'scenario': scenario, 'body': body, 'body_tokens': budget,
                                 'body_sha256': token_hash(enc(body)), 'padding': padding})

performance = []
for system in systems:
    for budget in (32, 64):
        for i, content in enumerate(prepared[budget]):
            body = content['body']; text = system['prefix'] + USER + body + ASSISTANT
            ids = enc(text); overhead = len(ids) - system['static_tokens'] - len(enc(body))
            assert ids[:system['static_tokens']] == system['prefix_tokens']
            assert overhead == 8
            performance.append({'id': f's{system["static_tokens"]}-d{budget}-{i}',
                                'static_tokens': system['static_tokens'],
                                'system_body_tokens': system['system_body_tokens'],
                                'system_template_tokens': system['system_template_tokens'],
                                'prefix_tokens': system['prefix_tokens'],
                                'prefix_sha256': system['prefix_sha256'], **content,
                                'template_overhead': overhead, 'tokens': ids,
                                'prompt_tokens': len(ids), 'prompt_text': text,
                                'tokens_sha256': token_hash(ids), 'prompt_text_sha256': text_hash(text),
                                'expected_reused_tokens': system['static_tokens'],
                                'expected_new_tokens': budget + overhead,
                                'timing': 'one generated token; prefix preparation excluded and separately reported',
                                'quality_use': 'cost fixture; natural input quality is evaluated separately'})

natural = [
 ('tired-greeting', '用户刚回家有些疲惫，希望机器人点头回应，不要移动底盘。', '点头回应；底盘不得移动'),
 ('reading-cancel', '用户正在阅读，希望保持安静，取消刚才向前移动的要求。', '保持静止；不得继续向前移动'),
 ('wave', '用户想打个招呼，希望摆臂回应。桌边空间有限，底盘保持静止。', '摆臂招呼；底盘不得移动'),
 ('nod', '请点头回应。', '点头回应'),
 ('bend-not-look-up', '请弯腰，不要抬头。', '腰部弯曲；不得抬头'),
 ('cancel-motion', '取消刚才移动，保持静止。', '停止移动并保持静止')
]
quality = []
for system in systems:
    for case, text, constraint in natural:
        full = system['prefix'] + USER + text + ASSISTANT; ids = enc(full)
        assert ids[:system['static_tokens']] == system['prefix_tokens']
        assert len(ids)-system['static_tokens']-len(enc(text)) == 8
        quality.append({'id': f's{system["static_tokens"]}-{case}', 'case': case,
                        'static_tokens': system['static_tokens'], 'prefix_tokens': system['prefix_tokens'],
                        'prefix_sha256': system['prefix_sha256'], 'input': text,
                        'body_tokens': len(enc(text)), 'template_overhead': 8,
                        'tokens': ids, 'tokens_sha256': token_hash(ids),
                        'prompt_tokens': len(ids), 'prompt_text': full,
                        'expected_reused_tokens': system['static_tokens'],
                        'expected_new_tokens': len(ids)-system['static_tokens'],
                        'task_constraint_for_manual_audit': constraint,
                        'padding': 'none', 'independent_holdout': False})

result = {'schema_version': 1, 'date': '2026-10-25', 'tokenizer_path': str(TOKENIZER), 'encoding_baseline': 'board runtime exported tokenizer; file identity enforced by preflight',
          'tokenizer_sha256': hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
          'tokenizers_version': version('tokenizers'), 'add_special_tokens': False,
          'token_hash_encoding': 'SHA256 over concatenated uint32 little-endian token IDs',
          'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'definitions': {
              'static_tokens': 'Entire fixed system prefix including system role header and closing template',
              'system_body_tokens': 'Chinese system body only; exact measured counts are in systems',
              'system_template_tokens': f'Fixed system template contribution: {SYSTEM_TEMPLATE_TOKENS}',
              'body_tokens': 'Dynamic Chinese body only: 32 / 64; same body IDs across both static sizes',
              'template_overhead': 'Dynamic user/assistant role template contribution: 8',
              'new_tokens': 'Expected 40 / 72 with only fixed prefix reuse; not 32 / 64',
              'cache_policy': 'Prepare identical frozen prefix with reset per request; assert reuse equals64/128; no exact flag needed as both are16-aligned',
              'same_cell': 'All runtime modes must use exactly these same prefix and full prompt IDs, seed, feedback, and output length',
              'cross_cell': 'Long system repeats same core constraints; duplication can affect quality. Compare runtime algorithms within cells, not claim prompt semantics identical across cells',
              'scope': 'Intern NPU text decoder, synthetic audio in concurrent harness; mock RAG only; no image encoder or physical motors',
              'quality': 'Six previously used development prompts per system; not independent held-out quality validation'},
          'systems': systems, 'performance': performance, 'quality': quality}
assert len(performance) == 12 and len(quality) == 12
(ROOT/'aligned-inputs.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'systems': [(x['static_tokens'], x['system_body_tokens'], x['system_template_tokens'], x['static_padding']['added_token_count']) for x in systems],
                  'performance': [(x['id'], x['body_tokens'], x['prompt_tokens'], x['expected_new_tokens'], x['padding']['added_token_count']) for x in performance],
                  'quality_count': len(quality), 'output_sha256': hashlib.sha256((ROOT/'aligned-inputs.json').read_bytes()).hexdigest()}, ensure_ascii=False))
