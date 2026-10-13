"""Isolated text-only HTTP acceptance service for a resident full-depth NPU.

The worker can be local or an SSH event transport. CPU production is untouched.
The deployment/validation owner holds the host and board benchmark locks for
the lifetime of this service. This module never chooses or promotes a model.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit


class ContextOverflow(ValueError):
    pass


class ChatCodec:
    def __init__(self, model, system):
        import gguf
        from full_prompt import tokenizer_from_gguf
        from jinja2 import Environment
        reader = gguf.GGUFReader(model)
        self.tokenizer = tokenizer_from_gguf(reader)
        self.template = Environment().from_string(reader.fields['tokenizer.chat_template'].contents())
        self.typed_content = reader.fields['tokenizer.ggml.pre'].contents() == 'smollm'
        self.system = system

    def encode(self, messages):
        messages = [{'role': 'system', 'content': self.system}, *messages]
        if self.typed_content:
            messages = [{**m, 'content': [{'type': 'text', 'text': m['content']}]} for m in messages]
        prompt = self.template.render(messages=messages, add_generation_prompt=True)
        return self.tokenizer.encode(prompt, add_special_tokens=False).ids

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def prepare_prompt(codec, messages, capacity, max_new):
    if not isinstance(messages, list) or not 0 < len(messages) <= 1023 or len(messages) % 2 != 1:
        raise ValueError('需要交替的用户/助手消息，并以用户问题结束')
    for i, message in enumerate(messages):
        if (not isinstance(message, dict) or
                message.get('role') != ('user' if i % 2 == 0 else 'assistant') or
                not isinstance(message.get('content'), str) or
                not message['content'] or len(message['content']) > 12000):
            raise ValueError('目前只接受非空文字消息')
    if type(max_new) is not int or not 1 <= max_new <= min(256, capacity):
        raise ValueError('输出长度必须为1–256 token，且不超过窗口')
    dropped = 0
    while True:
        tokens = codec.encode(messages[dropped:])
        if tokens and len(tokens) + max_new - 1 <= capacity:
            return tokens, dropped
        if len(messages) - dropped <= 1:
            raise ContextOverflow('最新问题超过上下文容量，请缩短输入')
        dropped += 2  # Never keep a dangling assistant or discard the system role.


class ChatEngine:
    def __init__(self, codec, worker_factory, model, capacity):
        self.codec, self.worker_factory = codec, worker_factory
        self.model, self.capacity = model, capacity
        self.worker = None
        self.operation = threading.Lock()

    def close(self):
        if self.worker is not None:
            worker, self.worker = self.worker, None
            return worker.close()

    def stream(self, tokens, max_new):
        # A disconnected/incomplete request retires its worker, so unread token
        # events and half-written KV cannot contaminate the next request.
        complete = False
        received = []
        emitted = ''
        http_start = time.monotonic()
        load_seconds = 0
        if self.worker is None:
            start = time.monotonic()
            self.worker = self.worker_factory()
            load_seconds = time.monotonic() - start
            if self.worker.capacity != self.capacity:
                self.worker.abort()
                self.worker = None
                raise RuntimeError('worker context does not match service configuration')
        source = self.worker.generate(tokens, max_new=max_new)
        try:
            for event in source:
                if event['event'] == 'token':
                    received.append(event['token'])
                    # Byte-level tokens can end in an incomplete UTF-8 sequence.
                    # Wait for continuation bytes instead of emitting replacement
                    # characters which cannot be retracted from an SSE stream.
                    text = self.codec.decode(received).rstrip('\ufffd')
                    if not text.startswith(emitted):
                        raise RuntimeError('tokenizer revised already streamed text')
                    delta, emitted = text[len(emitted):], text
                    yield {'choices': [{'delta': {'content': delta}, 'finish_reason': None}]}
                else:
                    text = self.codec.decode(received)
                    if not text.startswith(emitted):
                        raise RuntimeError('final text disagrees with stream')
                    yield {'choices': [{'delta': {'content': text[len(emitted):]},
                                        'finish_reason': 'stop' if event['finish_reason'] == 'eos' else 'length'}],
                           'metrics': {**event, 'model': self.model, 'model_load_seconds': load_seconds,
                                       'http_total_seconds': time.monotonic() - http_start,
                                       'scope': 'HTTP/transport included separately; not the controlled five-run benchmark'}}
                    complete = True
        finally:
            source.close()
            if not complete and self.worker is not None:
                self.worker.abort()
                self.worker = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_args):
        pass

    def reply(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        engine = self.server.engine
        if self.path == '/api/health':
            return self.reply(200, {'model': engine.model, 'backend': 'NPU experimental',
                                   'context': engine.capacity, 'busy': engine.operation.locked(),
                                   'resident': engine.worker is not None,
                                   'independent_quality_accepted': False})
        if self.path != '/':
            return self.reply(404, {'error': 'Not found'})
        body = Path(__file__).with_name('resident_chat.html').read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.close_connection = True
        if self.path != '/api/chat':
            return self.reply(404, {'error': 'Not found'})
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            return self.reply(403, {'error': 'Origin not allowed'})
        engine = self.server.engine
        try:
            self.connection.settimeout(30)
            size = int(self.headers.get('Content-Length', 0))
            if self.headers.get('Transfer-Encoding') or not 0 < size <= 256 * 1024:
                raise ValueError('请求过大或缺少长度')
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('请求格式错误')
            max_new = data.get('max_tokens', 128)
            tokens, dropped = prepare_prompt(engine.codec, data.get('messages'), engine.capacity, max_new)
        except (ValueError, OSError) as exc:
            return self.reply(400, {'error': str(exc)})
        if not engine.operation.acquire(blocking=False):
            return self.reply(409, {'error': '模型正在生成，请稍后再试'})
        stream = None
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Connection', 'close')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Context-Dropped', str(dropped))
            self.end_headers()
            stream = engine.stream(tokens, max_new)
            for event in stream:
                self.wfile.write(('data: ' + json.dumps(event, ensure_ascii=False) + '\n\n').encode())
                self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                self.wfile.write(('data: ' + json.dumps({'error': str(exc)}) + '\n\n').encode())
                self.wfile.flush()
            except OSError:
                pass
        finally:
            if stream is not None:
                stream.close()
            engine.operation.release()


def make_server(engine, address=('127.0.0.1', 0)):
    server = ThreadingHTTPServer(address, Handler)
    server.engine = engine
    return server
