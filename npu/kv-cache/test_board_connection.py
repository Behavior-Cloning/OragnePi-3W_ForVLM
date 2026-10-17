"""Check Windows/WSL board address and strict-host-key configuration."""
import os
import unittest
from unittest.mock import patch
from board_connection import board_transport


class BoardConnectionTests(unittest.TestCase):
    def test_remote_host_and_known_hosts_apply_to_ssh_and_scp(self):
        with patch.dict(os.environ, {'OPI_BOARD_HOST': 'root@192.0.2.5',
                                    'OPI_KNOWN_HOSTS': '/tmp/known hosts',
                                    'SSHPASS': 'unit-test-placeholder'}, clear=True), \
             patch('board_connection.shutil.which', return_value='/usr/bin/sshpass'):
            host, ssh, scp = board_transport()
        self.assertEqual(host, 'root@192.0.2.5')
        self.assertEqual(ssh[-1], host)
        for command in (ssh, scp):
            self.assertEqual(command[:2], ['/usr/bin/sshpass', '-e'])
            self.assertIn('StrictHostKeyChecking=yes', command)
            self.assertIn('UserKnownHostsFile=/tmp/known hosts', command)
            self.assertNotIn('unit-test-placeholder', command)

    def test_key_auth_has_no_sshpass_dependency(self):
        with patch.dict(os.environ, {'OPI_BOARD_HOST': 'board'}, clear=True):
            _, ssh, scp = board_transport()
        self.assertEqual(ssh[0], 'ssh')
        self.assertEqual(scp[0], 'scp')

    def test_option_or_whitespace_host_is_rejected(self):
        for value in ('-Fmalicious', 'root@board extra'):
            with patch.dict(os.environ, {'OPI_BOARD_HOST': value}, clear=True):
                with self.assertRaises(ValueError):
                    board_transport()


if __name__ == '__main__':
    unittest.main()
