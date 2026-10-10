#!/usr/bin/env python3
"""Sequential real-model prompt comparison; run on the board, stdlib only."""
import argparse
import json
import time
import urllib.request
from pathlib import Path

PROMPTS = {
    'baseline': '',
    'english': 'You are Orange, an AI assistant running on an Orange Pi. Answer the user directly in one or two short sentences. Follow the conversation. Do not invent personal experiences. If an image is provided, answer based on what is visible. If you do not know, say so. Reply in the same language as the user.',
    'chinese': '你是橙语，一个运行在香橙派上的AI助手。请用简短、自然的中文回答用户最后一条消息。记住前面的聊天内容。不要编造自己的姓名、经历或用户的信息。用户分享生活时自然回应，心情不好时先表达理解。看图时只说能看到的内容。不知道就说不知道。',
    'short': 'You are a helpful AI assistant named Orange. Give a short, relevant answer to the last user message. Use the conversation history. 中文问题用中文回答。',
}
QUESTIONS = ['你好，你是谁？', '我今天吃了黄焖鸡', '心情有点不好', '我刚才说吃了什么？']


def request(messages, temperature, repeat_penalty):
    body = json.dumps({'model': 'smolvlm-256m', 'messages': messages, 'stream': False,
                       'max_tokens': 96, 'temperature': temperature, 'repeat_penalty': repeat_penalty,
                       'seed': 42}).encode()
    request = urllib.request.Request('http://127.0.0.1:8081/v1/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=['initial', 'followup', 'validation'], default='initial')
    parser.add_argument('--prompt-file', type=Path)
    args = parser.parse_args()
    results = []
    variants = [(name, prompt, .2, 1.1) for name, prompt in PROMPTS.items()]
    variants += [('short_greedy', PROMPTS['short'], 0, 1.0), ('chinese_greedy', PROMPTS['chinese'], 0, 1.0)]
    if args.suite == 'followup':
        concise = 'You are Orange, an AI assistant. Answer the latest question briefly and accurately. Use the conversation history. Do not invent facts. If an image is provided, describe only what you can see.'
        variants = [('english_baseline', '', .2, 1.1), ('english_concise', concise, .2, 1.1),
                    ('english_greedy', concise, 0, 1.1),
                    ('zh_minimal', '你是AI助手橙语。简短回答，不要编造。', .2, 1.1),
                    ('zh_examples', '你是AI助手橙语，用简短中文聊天。以下只是回复风格示例，不是用户的经历：\n问：你叫什么？\n答：我是橙语，一个AI助手。\n问：我去公园散步了。\n答：散步感觉怎么样？\n问：我有些烦恼。\n答：愿意说说发生了什么吗？\n现在回答真实用户的最新消息。', .2, 1.1)]
    if args.suite == 'validation':
        if not args.prompt_file:
            parser.error('--prompt-file is required for validation')
        selected = args.prompt_file.read_text().strip()
        variants = [('english_holdout', selected, .2, 1.1), ('chinese_selected', selected, .2, 1.1)]
    for name, prompt, temperature, penalty in variants:
        messages = [{'role': 'system', 'content': prompt}] if prompt else []
        run = {'variant': name, 'prompt': prompt, 'temperature': temperature, 'repeat_penalty': penalty, 'turns': []}
        results.append(run)
        questions = ['Hi, who are you?', 'I ate braised chicken today.', 'I feel a bit sad.', 'What did I say I ate?'] if name.startswith('english_') else QUESTIONS
        if name == 'english_holdout':
            questions = ['Are you a human or an AI?', 'My pet rabbit is called Luna.', 'What is my pet called?', 'What color is my pet?']
        for question in questions:
            messages.append({'role': 'user', 'content': question})
            start = time.monotonic()
            response = request(messages, temperature, penalty)
            answer = response['choices'][0]['message']['content']
            messages.append({'role': 'assistant', 'content': answer})
            turn = {'question': question, 'answer': answer, 'seconds': time.monotonic()-start}
            run['turns'].append(turn)
            print(json.dumps({'variant': name, **turn}, ensure_ascii=False), flush=True)
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
