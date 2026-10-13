"""Exercise context boundaries, interrupted streams and HTTP using a fake engine."""
import http.client
import json
import threading
import unittest

from resident_http import ChatEngine, ContextOverflow, make_server, prepare_prompt


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

    def test_utf8_stream_and_worker_reuse(self):
        first = list(self.engine.stream([5], 3))
        self.assertEqual(''.join(e['choices'][0]['delta']['content'] for e in first), '橙子')
        self.assertNotIn('\ufffd', json.dumps(first, ensure_ascii=False))
        list(self.engine.stream([5, 6], 3))
        self.assertEqual(len(self.workers), 1)
        self.assertFalse(self.workers[0].aborted)

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
