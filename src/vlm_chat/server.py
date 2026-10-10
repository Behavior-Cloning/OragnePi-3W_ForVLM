#!/usr/bin/env python3
"""Small LAN web gateway for a local llama.cpp multimodal server (stdlib only)."""
import argparse
import http.client
import json
import socket
import select
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

STATIC = Path(__file__).resolve().parent / 'web'
SYSTEM_PROMPT = Path(__file__).with_name('system_prompt.txt').read_text(encoding='utf-8').strip()
MAX_BODY = 16 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def reply(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def backend(self):
        return http.client.HTTPConnection(self.server.backend_host,
                                          self.server.backend_port, timeout=360)

    def control(self, method='GET', data=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.control_port, timeout=360)
        try:
            connection.request(method, '/state' if method == 'GET' else '/settings',
                               body=json.dumps(data) if data else None,
                               headers={'Content-Type': 'application/json', 'X-OPI-Control': 'local-gateway'})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/health':
            if self.server.control_port:
                try:
                    _, state = self.control()
                    state['busy'] = self.server.operation.locked()
                    return self.reply(200, state)
                except (OSError, ValueError, http.client.HTTPException):
                    return self.reply(503, {'ready': False, 'error': '模型控制服务暂不可用'})
            conn = self.backend()
            try:
                conn.request('GET', '/health')
                response = conn.getresponse()
                response.read()
                self.reply(200 if response.status == 200 else 503, {
                    'ready': response.status == 200,
                    'model': self.server.model_name, 'backend': 'CPU', 'context': 4096,
                })
            except (OSError, http.client.HTTPException):
                self.reply(503, {'ready': False, 'error': '模型正在启动或暂时不可用'})
            finally:
                conn.close()
            return
        files = {'/': ('index.html', 'text/html; charset=utf-8'),
                 '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                 '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if path not in files:
            self.reply(404, {'error': 'Not found'})
            return
        filename, mime = files[path]
        body = (STATIC / filename).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data: blob:; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in ('/api/chat', '/api/settings'):
            self.reply(404, {'error': 'Not found'})
            return
        # Browser requests must originate from this site; no arbitrary backend URLs.
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            self.close_connection = True
            self.reply(403, {'error': 'Origin not allowed'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_BODY:
                self.close_connection = True
                self.reply(413, {'error': '图片或历史过大，请减少图片或新建对话'})
                return
            self.connection.settimeout(30)
            data = json.loads(self.rfile.read(size))
            if self.path == '/api/settings':
                if not self.server.control_port:
                    return self.reply(503, {'error': '此部署未启用模型切换'})
                if not self.server.operation.acquire(blocking=False):
                    return self.reply(409, {'error': '模型正在生成或加载，请稍后切换'})
                try:
                    code, state = self.control('POST', data)
                    self.reply(code, state)
                except (OSError, ValueError, http.client.HTTPException):
                    self.reply(503, {'error': '切换连接中断，请查看模型状态后重试'})
                finally:
                    self.server.operation.release()
                return
            messages = data.get('messages')
            if not isinstance(messages, list) or not messages or len(messages) > 1024:
                raise ValueError('需要 1–1024 条消息')
            for message in messages:
                if not isinstance(message, dict) or message.get('role') not in ('user', 'assistant'):
                    raise ValueError('不支持的消息格式')
                content = message.get('content')
                if isinstance(content, str):
                    if len(content) > 12000:
                        raise ValueError('消息过长')
                elif isinstance(content, list) and 0 < len(content) <= 5:
                    for part in content:
                        if not isinstance(part, dict):
                            raise ValueError('不支持的消息内容')
                        if part.get('type') == 'text' and isinstance(part.get('text'), str):
                            if len(part['text']) > 12000:
                                raise ValueError('消息过长')
                        elif part.get('type') == 'image_url' and message['role'] == 'user':
                            url = part.get('image_url', {}).get('url', '')
                            if not isinstance(url, str) or not url.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,', 'data:image/webp;base64,')):
                                raise ValueError('仅接受上传的 JPEG/PNG/WebP 图片')
                        else:
                            raise ValueError('不支持的消息内容')
                else:
                    raise ValueError('消息内容不能为空')
        except (ValueError, TypeError, AttributeError, socket.timeout):
            self.close_connection = True
            self.reply(400, {'error': '消息格式错误、过长或上传超时'})
            return
        conn = self.backend()
        started = False
        if not self.server.operation.acquire(timeout=.5):
            return self.reply(409, {'error': '模型正在生成或加载，请稍后重试'})
        cancelled = threading.Event()
        upstream = [None]
        def watch_disconnect():
            while not cancelled.wait(.05):
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    if readable and not self.connection.recv(1, socket.MSG_PEEK):
                        cancelled.set()
                        if upstream[0]:
                            upstream[0].shutdown(socket.SHUT_RDWR)
                        return
                except OSError:
                    return
        watcher = threading.Thread(target=watch_disconnect, daemon=True)
        watcher.start()
        try:
            alias = self.server.model_alias
            if self.server.control_port:
                _, state = self.control()
                if state.get('maintenance'):
                    return self.reply(503, {'error': '性能评测进行中，请等待完成'})
                if not state.get('ready'):
                    return self.reply(503, {'error': '模型正在加载，请等待'})
                alias = state['model_id']
                if state.get('vision') is False and any(isinstance(m['content'],list) and any(p['type']=='image_url' for p in m['content']) for m in messages):
                    return self.reply(400, {'error':'当前模型仅支持文字，请新建纯文字对话或切回视觉模型。'})
                if (data.get('model_id', alias) != alias or
                    data.get('context', state['context']) != state['context']):
                    return self.reply(409, {'error': '其他用户已更改模型或窗口，请确认当前设置后重发'})
            dropped = 0
            while True:
                model_messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, *messages]
                payload = json.dumps({'model': alias, 'messages': model_messages,
                                  'stream': True, 'max_tokens': 256,
                                  'temperature': 0.2, 'repeat_penalty': 1.1}).encode()
                conn.request('POST', '/v1/chat/completions', body=payload,
                         headers={'Content-Type': 'application/json'})
                upstream[0] = conn.sock
                if cancelled.is_set():
                    return
                response = conn.getresponse()
                if response.status == 200:
                    break
                detail = response.read(16384).decode(errors='replace')
                try:
                    overflow = json.loads(detail).get('error', {}).get('type') == 'exceed_context_size_error'
                except (ValueError, AttributeError):
                    overflow = False
                next_turn = next((i for i,m in enumerate(messages) if i > 0 and m['role'] == 'user'), None)
                if overflow and next_turn is not None:
                    dropped += next_turn
                    messages = messages[next_turn:]
                    conn.close()
                    conn = self.backend()
                    continue
                self.reply(response.status, {'error': '最新问题和图片已超过上下文，请增大窗口或减少输入。' if overflow else '模型无法处理当前请求。若上下文已满，请调整窗口。',
                                             'detail': detail})
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.send_header('X-Accel-Buffering', 'no')
            self.send_header('X-Context-Dropped', str(dropped))
            self.end_headers()
            self.close_connection = True
            started = True
            # read1 does not wait for a full buffer, so tokens stream immediately.
            while True:
                chunk = response.read1(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # Closing upstream also cancels generation on llama-server.
        except (OSError, http.client.HTTPException):
            if not started:
                self.reply(503, {'error': '模型服务暂不可用，请稍后重试'})
            else:
                try:
                    self.wfile.write(b'data: {"error":"Inference connection interrupted"}\n\n')
                    self.wfile.flush()
                except OSError:
                    pass
        finally:
            cancelled.set()
            watcher.join(timeout=.2)
            conn.close()
            self.server.operation.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--backend-host', default='127.0.0.1')
    parser.add_argument('--backend-port', type=int, default=8081)
    parser.add_argument('--model-alias', default='smolvlm-256m')
    parser.add_argument('--model-name', default='SmolVLM-256M')
    parser.add_argument('--control-port', type=int, default=0)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.backend_host, server.backend_port = args.backend_host, args.backend_port
    server.model_alias, server.model_name = args.model_alias, args.model_name
    server.control_port = args.control_port
    server.operation = threading.Lock()
    print(f'SmolVLM chat listening on {args.host}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
