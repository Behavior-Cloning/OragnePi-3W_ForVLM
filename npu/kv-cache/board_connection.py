"""Environment-configurable SSH transport for board tools (strict known hosts)."""
import os
from pathlib import Path
import shutil


def board_transport():
    host = os.environ.get('OPI_BOARD_HOST', 'root@192.168.88.29')
    if host.startswith('-') or any(c.isspace() for c in host):
        raise ValueError('invalid OPI_BOARD_HOST')
    prefix = []
    if 'SSHPASS' in os.environ:
        binary = shutil.which('sshpass')
        if not binary:
            candidate = Path.home() / '.local/bin/sshpass'
            if candidate.is_file():
                binary = str(candidate)
        if not binary:
            raise RuntimeError('SSHPASS is set but sshpass is unavailable')
        prefix = [binary, '-e']
    options = ['-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8', '-o', 'StrictHostKeyChecking=yes']
    if os.environ.get('OPI_KNOWN_HOSTS'):
        options += ['-o', 'UserKnownHostsFile=' + os.environ['OPI_KNOWN_HOSTS']]
    return host, prefix + ['ssh', *options, host], prefix + ['scp', *options]
