"""Private subprocess protocol for the experimental resident NPU runner.

One caller owns a worker at a time. No HTTP endpoint or model promotion is
implicit. Deadlines include event waiting; failures terminate the worker.
"""
import json
import os
from pathlib import Path
import selectors
import struct
import subprocess
import time


class ResidentWorker:
    def __init__(self, command, cwd, log, dump_dir=None, timeout=120,
                 events_on_stdout=False):
        self.capacity = None
        self.last_request = 0
        self.events = bytearray()
        read_fd, write_fd = os.pipe()
        self.read_fd = read_fd
        self.selector = selectors.DefaultSelector()
        self.selector.register(read_fd, selectors.EVENT_READ)
        env = {k: v for k, v in os.environ.items() if not k.startswith('OPI_')}
        env['OPI_SESSION_FD'] = str(write_fd)
        if dump_dir is not None:
            env['OPI_SESSION_DUMP_DIR'] = str(Path(dump_dir).resolve())
        self.log = open(log, 'xb')
        try:
            self.process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                            stdout=write_fd if events_on_stdout else self.log,
                                            stderr=self.log,
                                            pass_fds=(write_fd,))
        except BaseException:
            self.selector.close()
            os.close(read_fd)
            self.log.close()
            raise
        finally:
            os.close(write_fd)
        try:
            self.ready = self._event(time.monotonic() + timeout)
            if self.ready.get('event') != 'ready':
                raise RuntimeError('worker did not become ready')
            self.capacity = self.ready['capacity']
        except BaseException:
            self.abort()
            raise

    def _event(self, deadline):
        while b'\n' not in self.events:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError('resident worker event timed out')
            block = os.read(self.read_fd, 65536)
            if not block:
                raise RuntimeError('resident worker event pipe closed')
            self.events.extend(block)
            if len(self.events) > 1024 * 1024:
                raise RuntimeError('invalid oversized worker event')
        line, _, rest = self.events.partition(b'\n')
        self.events = bytearray(rest)
        return json.loads(line)

    def generate(self, tokens, max_new=128, feedback=None, reset=False, diagnostic=False,
                 timeout=300):
        """Yield token events then one done event, with exact reused/new counts."""
        feedback = [] if feedback is None else feedback
        if (not tokens or max_new < 1 or len(tokens) + max_new - 1 > self.capacity or
                (feedback and len(feedback) != max_new)):
            raise ValueError('invalid token counts or context overflow')
        if any(type(t) is not int or not 0 <= t < 2**31 for t in [*tokens, *feedback]):
            raise ValueError('tokens must be nonnegative signed int32 values')
        self.last_request += 1
        request = self.last_request
        payload = struct.pack('<8I', 0x4F504931, request, len(tokens), max_new, len(feedback),
                              bool(reset), bool(diagnostic), 0)
        payload += struct.pack('<' + 'i' * (len(tokens) + len(feedback)), *tokens, *feedback)
        deadline = time.monotonic() + timeout
        generated = []
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
            while True:
                event = self._event(deadline)
                if event.get('id') != request:
                    raise RuntimeError('response request ID mismatch')
                if event.get('event') == 'token':
                    generated.append(event['token'])
                    if len(generated) > max_new:
                        raise RuntimeError('worker generated too many tokens')
                    yield event
                elif event.get('event') == 'done':
                    if event['generated_tokens'] != len(generated) or not generated:
                        raise RuntimeError('worker token count mismatch')
                    if event['prompt_tokens'] != len(tokens):
                        raise RuntimeError('worker prompt count mismatch')
                    yield {**event, 'tokens': generated}
                    return
                else:
                    raise RuntimeError('unexpected worker event')
        except BaseException:
            self.abort()
            raise

    def close(self):
        self.process.stdin.close()
        try:
            closed = self._event(time.monotonic() + 30)
            if closed.get('event') != 'closed' or self.process.wait(timeout=30):
                raise RuntimeError('worker failed during release')
            return closed
        finally:
            self.abort()

    def abort(self):
        if hasattr(self, 'process') and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=10)
        if hasattr(self, 'process') and self.process.stdin:
            self.process.stdin.close()
        self.selector.close()
        if self.read_fd is not None:
            os.close(self.read_fd)
            self.read_fd = None
        self.log.close()
