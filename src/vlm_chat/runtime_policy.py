"""Legacy CPU gateway retirement; never substitute a CPU fallback for Intern NPU."""
def retired_state():
    return {
        'ready': False, 'retired': True, 'loading': False, 'models': [],
        'model_id': None, 'backend': None,
        'replacement': 'InternVL3-1B NPU resident worker',
        'error': '旧 CPU 网页推理链路已退役。当前使用 Intern NPU worker；此 HTTP 入口尚未接入该 worker。',
    }
