"""Exercise context boundaries, interrupted streams and HTTP using a fake engine."""
import http.client
import json
import threading
import unittest

from resident_http import ChatCodec, ChatEngine, ContextOverflow, make_server, prepare_prompt


class Codec:
    def encode(self, messages):
        return [0] + [ord(c) for m in messages for c in m['content']]

    def decode(self, tokens):
        # A token pair splits a multibyte character in the same way as byte BPE.
        return {(): '', (1,): '\ufffd', (1, 2): '橙', (1, 2, 3): '橙子'}[tuple(tokens)]


class Worker:
    capacity = 32

    def __init__(self):
        self.aborted = False

    def generate(self, tokens, max_new):
        for t in (1, 2, 3):
            yield {'event': 'token', 'token': t}
        yield {'event': 'done', 'finish_reason': 'eos', 'generated_tokens': 3}

    def abort(self):
        self.aborted = True

    def close(self):
        return {'driver_bytes': 0}


class ResidentHTTPTest(unittest.TestCase):
    def setUp(self):
        self.workers = []

        def factory():
            worker = Worker()
            self.workers.append(worker)
            return worker

        self.engine = ChatEngine(Codec(), factory, 'test', 32)

    def test_context_truncates_complete_turn_and_keeps_current_question(self):
        messages = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'reply'},
                    {'role': 'user', 'content': 'new'}]
        tokens, dropped = prepare_prompt(Codec(), messages, 8, 3)
        self.assertEqual(dropped, 2)
        self.assertEqual(tokens, [0, *map(ord, 'new')])
        with self.assertRaises(ContextOverflow):
            prepare_prompt(Codec(), messages[-1:], 5, 3)

    def test_rejects_dangling_assistant_and_boolean_length(self):
        with self.assertRaises(ValueError):
            prepare_prompt(Codec(), [{'role': 'assistant', 'content': 'oops'}], 32, 1)
        with self.assertRaises(ValueError):
            prepare_prompt(Codec(), [{'role': 'user', 'content': 'ok'}], 32, True)

    def test_overflow_headroom_is_opt_in_and_keeps_latest_question(self):
        messages = [{'role': 'user', 'content': 'aaa'}, {'role': 'assistant', 'content': 'bbb'},
                    {'role': 'user', 'content': 'ccc'}, {'role': 'assistant', 'content': 'ddd'},
                    {'role': 'user', 'content': 'last'}]
        # No overflow: a low retention target must not evict valid history.
        self.assertEqual(prepare_prompt(Codec(), messages, 20, 3, 6)[1], 0)
        # Overflow: ordinary policy removes one turn; headroom policy two.
        self.assertEqual(prepare_prompt(Codec(), messages, 15, 3)[1], 2)
        self.assertEqual(prepare_prompt(Codec(), messages, 15, 3, 6)[1], 4)
        # The current question may exceed the soft target, never the hard cap.
        tokens, dropped = prepare_prompt(Codec(), messages, 15, 3, 2)
        self.assertEqual(tokens, [0, *map(ord, 'last')])
        self.assertEqual(dropped, 4)
        with self.assertRaises(ValueError):
            prepare_prompt(Codec(), messages, 20, 3, True)

    def test_utf8_stream_and_worker_reuse(self):
        first = list(self.engine.stream([5], 3))
        self.assertEqual(''.join(e['choices'][0]['delta']['content'] for e in first), '橙子')
        self.assertNotIn('\ufffd', json.dumps(first, ensure_ascii=False))
        list(self.engine.stream([5, 6], 3))
        self.assertEqual(len(self.workers), 1)
        self.assertFalse(self.workers[0].aborted)

    def test_smol_generated_separator_is_not_duplicated(self):
        from jinja2 import Environment
        from types import SimpleNamespace
        codec = ChatCodec.__new__(ChatCodec)
        codec.system = 'System'
        codec.typed_content = True
        codec.template = Environment().from_string(
            '{% for m in messages %}{{m.role | capitalize}}: {{m.content[0].text}}\n{% endfor %}Assistant:')
        codec.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: SimpleNamespace(ids=list(text.encode())))
        first = [{'role': 'user', 'content': 'Hi'}]
        next_turn = first + [{'role': 'assistant', 'content': ' Hello!'}, {'role': 'user', 'content': 'Continue'}]
        before = codec.encode(first) + list(b' Hello!')
        after = codec.encode(next_turn)
        self.assertEqual(after[:len(before)], before)

    def test_interrupt_discards_worker_before_next_request(self):
        stream = self.engine.stream([5], 3)
        next(stream)
        stream.close()
        self.assertTrue(self.workers[0].aborted)
        self.assertIsNone(self.engine.worker)
        list(self.engine.stream([6], 3))
        self.assertEqual(len(self.workers), 2)

    def test_http_sse_origin_busy_and_no_internal_prompt_mutation(self):
        server = make_server(self.engine)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def request(data, origin=None):
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            try:
                headers = {'Content-Type': 'application/json'}
                if origin:
                    headers['Origin'] = origin
                connection.request('POST', '/api/chat', json.dumps(data), headers)
                response = connection.getresponse()
                return response.status, response.read().decode()
            finally:
                connection.close()
        data = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 3}
        try:
            self.assertEqual(request(data, 'http://foreign.test')[0], 403)
            with self.engine.operation:
                self.assertEqual(request(data)[0], 409)
            status, body = request(data)
            self.assertEqual(status, 200)
            self.assertIn('data: [DONE]', body)
            self.assertIn('橙', body)
            self.assertEqual(request([])[0], 400)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


if __name__ == '__main__':
    unittest.main()
