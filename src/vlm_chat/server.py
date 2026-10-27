#!/usr/bin/env python3
"""Retirement notice for the old CPU web gateway; no implicit model fallback."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from runtime_policy import retired_state
class Handler(BaseHTTPRequestHandler):
    def reply(self, code):
        data = json.dumps(retired_state(), ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        self.reply(200 if self.path == '/api/health' else 410)
    def do_POST(self):
        self.close_connection = True
        self.reply(410)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    # Accept old unit arguments only so upgrades can display the retirement notice.
    parser.add_argument('--control-port', type=int, default=0)
    parser.add_argument('--backend-host')
    parser.add_argument('--backend-port', type=int)
    parser.add_argument('--model-alias')
    parser.add_argument('--model-name')
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
if __name__ == '__main__':
    main()
