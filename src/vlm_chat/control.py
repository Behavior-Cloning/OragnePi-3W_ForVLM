#!/usr/bin/env python3
"""Retired CPU controller: reports status, never mutates services or loads models."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from runtime_policy import retired_state
class Handler(BaseHTTPRequestHandler):
    def reply(self, code):
        data = json.dumps(retired_state(), ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        self.reply(200 if self.path == '/state' else 404)
    def do_POST(self):
        self.close_connection = True
        self.reply(410)
if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 8083), Handler).serve_forever()
