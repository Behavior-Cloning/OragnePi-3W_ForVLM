"""Drive an isolated headless Chrome through the experimental chat page."""
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request


def validate_browser(url, output, engine):
    import websocket
    output = Path(output)
    with tempfile.TemporaryDirectory(prefix='opi-chat-chrome-') as profile:
        with (output / 'chrome.log').open('x') as log:
            browser = subprocess.Popen(['/usr/bin/google-chrome', '--headless=new', '--disable-gpu',
                                        '--no-first-run', '--no-default-browser-check',
                                        '--remote-debugging-port=0', '--remote-allow-origins=http://127.0.0.1',
                                        '--user-data-dir=' + profile, 'about:blank'], stdout=log, stderr=log)
            ws = None
            try:
                portfile = Path(profile) / 'DevToolsActivePort'
                deadline = time.monotonic() + 30
                while not portfile.exists():
                    if browser.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('isolated Chrome startup failed')
                    time.sleep(.1)
                port = int(portfile.read_text().splitlines()[0])
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/json', timeout=5) as response:
                    target = next(t for t in json.load(response) if t['type'] == 'page')
                ws = websocket.create_connection(target['webSocketDebuggerUrl'], timeout=10,
                                                  origin='http://127.0.0.1')
                sequence = 0
                errors = []
                def call(method, params=None):
                    nonlocal sequence
                    sequence += 1
                    ws.send(json.dumps({'id': sequence, 'method': method, 'params': params or {}}))
                    while True:
                        response = json.loads(ws.recv())
                        if response.get('method') == 'Runtime.exceptionThrown':
                            errors.append(response)
                        if response.get('id') == sequence:
                            if 'error' in response:
                                raise RuntimeError(response['error'])
                            return response.get('result', {})
                def evaluate(expression):
                    value = call('Runtime.evaluate', {'expression': expression, 'returnByValue': True})
                    if value.get('exceptionDetails'):
                        raise RuntimeError(value['exceptionDetails'])
                    return value['result'].get('value')
                def wait(expression, timeout=180):
                    deadline = time.monotonic() + timeout
                    while True:
                        value = evaluate(expression)
                        if value:
                            return value
                        if time.monotonic() > deadline:
                            raise TimeoutError('browser condition: ' + expression)
                        time.sleep(.1)
                call('Runtime.enable')
                call('Page.enable')
                call('Page.navigate', {'url': url})
                wait("document.getElementById('status')?.textContent.includes('token')", 30)
                turns = []
                for text in ('I am arranging flowers on my desk. Say something brief.',
                             'What did I put on my desk?'):
                    evaluate("document.getElementById('prompt').value=" + json.dumps(text) +
                             "; document.querySelector('form').requestSubmit(); true")
                    wait("!document.getElementById('send').disabled")
                    notice = evaluate("document.getElementById('notice').textContent")
                    assert not notice, notice
                    turns.append(evaluate("({answer:document.querySelector('#messages article:last-child').textContent,metrics:document.getElementById('metrics').textContent})"))
                time.sleep(.3)  # Allow the compositor to paint the last streamed turn.
                screenshot = call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': True})
                (output / 'browser-chat.png').write_bytes(base64.b64decode(screenshot['data']))
                evaluate("document.getElementById('prompt').value='Tell me a long story about the flowers.'; document.querySelector('form').requestSubmit(); true")
                wait("document.querySelector('#messages article:last-child').textContent.length>20 && !document.getElementById('stop').disabled")
                evaluate("document.getElementById('stop').click(); true")
                wait("!document.getElementById('send').disabled")
                assert evaluate("document.getElementById('notice').textContent") == '已停止生成。'
                deadline = time.monotonic() + 30
                while engine.operation.locked():
                    if time.monotonic() > deadline:
                        raise TimeoutError('browser cancellation did not release engine')
                    time.sleep(.1)
                assert engine.worker is None, 'cancelled worker survived'
                evaluate("document.getElementById('clear').click(); true")
                assert evaluate("document.getElementById('messages').children.length") == 0
                evaluate("document.getElementById('prompt').value='Say hello.'; document.querySelector('form').requestSubmit(); true")
                wait("!document.getElementById('send').disabled")
                assert not evaluate("document.getElementById('notice').textContent")
                assert not errors, errors
                return {'status': 'passed', 'turns': turns,
                        'checks': ['two browser turns', 'stop aborts worker', 'new conversation after stop'],
                        'screenshot': str(output / 'browser-chat.png')}
            finally:
                if ws is not None:
                    ws.close()
                browser.terminate()
                try:
                    browser.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    browser.kill()
                    browser.wait(timeout=5)
