"""Freeze complete chat turns with exact cached/new token counts for CPU/NPU."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import gguf
from jinja2 import Environment
from full_prompt import tokenizer_from_gguf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    models = [('smol500', 'smolvlm-500m', 'SmolVLM-500M-Instruct-Q8_0.gguf'),
              ('intern1b', 'internvl3-1b', 'InternVL3-1B-Instruct-Q8_0.gguf')]
    topics = ['下班路上的晚霞', '刚修好的程序', '窗边的雨声', '周末想读的书',
              '准备做的晚饭', '朋友分享的照片', '散步时的小猫', '今天听到的音乐']
    tail = ('我们可以慢慢聊，不必急着给今天打分。窗外的天色正在变暗，路灯亮起来了。'
            '你可以先坐下来喝一点水，让肩膀放松。今天的小事也值得被听见。'
            '也许等会儿听一首喜欢的歌，或者翻几页书，给自己留一点安静的时间。') * 12
    extra = ('今天回来的路上，我看见晚霞和路边的小猫，心情慢慢放松了。'
             '我想把工作先放一边，听一点温柔轻松的话，聊聊晚饭和明天的小安排。') * 8
    rows = []
    for short, model_id, name in models:
        model = project / 'models' / model_id / name
        reader = gguf.GGUFReader(model)
        tokenizer = tokenizer_from_gguf(reader)
        template = Environment().from_string(reader.fields['tokenizer.chat_template'].contents())

        def render(messages, generate):
            if short == 'smol500':
                messages = [{**m, 'content': [{'type': 'text', 'text': m['content']}]} for m in messages]
            return template.render(messages=messages, add_generation_prompt=generate)

        def ids(text):
            return tokenizer.encode(text, add_special_tokens=False).ids

        def fit(make, body, target):
            # Search Unicode character boundaries, then re-tokenize the complete
            # template. Never splice invalid UTF-8 or fabricate random token IDs.
            for end in range(len(body) + 1):
                for suffix in ('。', '！', '。嗯。', '。 好。', '。 OK.'):
                    content = body[:end] + suffix
                    text = make(content)
                    encoded = ids(text)
                    if len(encoded) == target:
                        return content, text, encoded
                if len(encoded) > target + 12:
                    break
            raise ValueError(f'cannot construct exact {target}-token text for {short}')

        for history in (512, 1024):
            messages = [{'role': 'system', 'content': '你是小橙，用户的桌边陪伴机器人。用自然中文回应，记住对话，不编造真实经历。'}]
            for i in range(32):
                topic = topics[i % len(topics)]
                pair = [{'role': 'user', 'content': f'我想聊聊{topic}，今天忙完之后终于有时间休息了。'},
                        {'role': 'assistant', 'content': f'那就慢慢聊{topic}吧。我在听，可以从最想说的一点开始。'}]
                if len(ids(render(messages + pair, False))) > history - 100:
                    break
                messages += pair
            messages += [{'role': 'user', 'content': '陪我慢慢聊一会儿吧。'}]
            last, prefix, prefix_ids = fit(lambda s: render(messages + [{'role': 'assistant', 'content': s}], False), tail, history)
            frozen_history = messages + [{'role': 'assistant', 'content': last}]
            for new in (32, 64, 128):
                content, text, tokens = fit(lambda s: render(frozen_history + [{'role': 'user', 'content': '陪我聊聊。' + s}], True), extra, history + new)
                assert text.startswith(prefix) and tokens[:history] == prefix_ids
                case = f'{short}-h{history}-n{new}'
                folder = a.output / case
                folder.mkdir()
                binary = struct.pack('<' + 'i' * len(tokens), *tokens)
                report = {'text': 'exact-length companion performance fixture; not held-out quality data',
                          'rendered_prompt': text, 'token_ids': tokens, 'cached_prefix': history,
                          'new_tokens': new, 'capacity': history + 256, 'steps': 128,
                          'messages': frozen_history + [{'role': 'user', 'content': '陪我聊聊。' + content}],
                          'history_rendered_prompt': prefix,
                          'new_token_scope': 'all newly evaluated tokens, including role/template and generation marker',
                          'prompt_sha256': hashlib.sha256(binary).hexdigest(),
                          'history_sha256': hashlib.sha256(binary[:history * 4]).hexdigest(),
                          'model_id': model_id, 'model_path': str(model.relative_to(project)),
                          'scope': 'identical CPU/NPU tokens within each model; model tokenizers differ across models',
                          'tokenizer_verification': 'host exact template; independently checked by board llama.cpp before measurement'}
                (folder / 'prompt.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
                (folder / 'prompt.i32').write_bytes(binary)
                rows.append({'case': case, 'model': short, 'model_id': model_id, 'history': history,
                             'new_tokens': new, 'capacity': history + 256, 'folder': str(folder.resolve()),
                             'prompt_sha256': report['prompt_sha256'], 'history_sha256': report['history_sha256']})
                print(case, len(tokens), flush=True)
    (a.output / 'matrix.json').write_text(json.dumps({'version': 1, 'generated_steps': 128, 'repeats': 5,
        'scope': 'full language decoder, final norm, complete head and sampling; no vision encoder',
        'cases': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
