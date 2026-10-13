"""Report HTTP development checks without claiming product or holdout acceptance."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'results'


def main():
    records = []
    lines = ['# NPU HTTP多轮接口验证', '',
             '模型和KV在开发板上常驻，完整32/24层及词表头由NPU执行；分词、HTTP网关和测试客户端在工作电脑，通过SSH传递token和事件。该布置用于验证接口，尚非板端独立部署，也没有替换8080的CPU网页。', '',
             '实验页为`resident_chat.html`，接口为`resident_http.py`。支持流式文字、停止、新对话和整轮历史裁剪；只有仍保留在当前页面的聊天记录，刷新不持久保存。独立留出、人工质量评阅、30分钟稳定性与真实音频并发另行验收。', '',
             '| 模型 | 实板HTTP状态 | 完整层数 | 上下文容量 | 已完成请求 |',
             '| --- | --- | ---: | ---: | ---: |']
    for model, folder in [('500M', 'resident-http-smol500-20261013'), ('1B', 'resident-http-intern1b-20261013')]:
        path = ROOT / folder / 'result.json'
        if not path.exists():
            lines.append(f'| {model} | 排队，尚无结果 | — | — | 0 |')
            continue
        data = json.loads(path.read_text())
        record = {**data, 'source_path': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        records.append(record)
        lines.append(f'| {model} | {data["status"]} | {data["layers"]} | {data["capacity"]} | {len(data["runs"])} |')
    lines += ['', '三个开发对话轮次使用模型自己的实际输出续聊，检查下一轮KV复用；再测试超窗删除完整旧轮、前缀变化重置，释放驱动至0，并重新加载后核对同一问题的完整输出token一致。这里没有用独立留出指导调参。', '',
              '本地单测验证UTF-8跨token切分、半途关闭流后丢弃worker、跨域拒绝及忙碌请求返回409。实板HTTP成功不自动证明浏览器停止按钮已上板验收；读表时按实际checks字段判断。']
    for record in records:
        lines += ['', '## ' + Path(record['model']).stem, '']
        if record.get('error'):
            lines += ['本次失败：`' + record['error'] + '`；原始记录保留，不算通过。', '']
        lines += ['| 请求 | 复用/新增token | 生成token | 引擎首token s | HTTP首个正文 s | 移除旧消息 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for i, run in enumerate(record['runs'], 1):
            m = run['metrics']
            first = run.get('http_first_text_seconds')
            first_text = '无正文' if first is None else f'{first:.3f}'
            lines.append(f'| {i} | {m["reused_tokens"]}/{m["new_tokens"]} | {m["generated_tokens"]} | {m["ttft_seconds"]:.3f} | {first_text} | {run["dropped_messages"]} |')
        lines += ['', 'HTTP首个正文时间包括必要的模型加载、SSH和UTF-8凑齐，不与五轮引擎首token直接相除。此组短开发对话没有控温配对，不作正式速度结论。', '', '实际自由生成内容：', '']
        for i, run in enumerate(record['runs'][:3], 1):
            lines += [f'第{i}轮用户：' + run['messages'][-1]['content'], '',
                      '> ' + (run['answer'] or '（空回答）').replace('\n', '\n> '), '']
    lines += ['', '复现：`run_resident_http_validation.py`，报告生成：`report_resident_http.py`。',
              '[紧凑证据](evidence-attack-20261013/resident-http.json)。', '']
    (HERE / 'HTTP多轮接口验证.md').write_text('\n'.join(lines))
    (HERE / 'evidence-attack-20261013/resident-http.json').write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')
    print('HTTP board result files', len(records))


if __name__ == '__main__':
    main()
