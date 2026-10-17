"""Serialize this host's deployments and board experiments, including SCP.

The board flock still protects inference against remote experiment clients.
This outer lock also excludes uploads/hashing that would perturb a benchmark.
"""
import fcntl
import os
from pathlib import Path

_held = None


def lock_board_from_host():
    global _held
    if _held is not None:
        raise RuntimeError('board host lock already held')
    path = Path(os.environ['OPI_BOARD_HOST_LOCK']) if os.environ.get('OPI_BOARD_HOST_LOCK') else Path(__file__).resolve().parents[1] / 'results' / 'board-host.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    _held = path.open('a')
    print('Waiting for exclusive board deployment/benchmark access...', flush=True)
    fcntl.flock(_held, fcntl.LOCK_EX)
