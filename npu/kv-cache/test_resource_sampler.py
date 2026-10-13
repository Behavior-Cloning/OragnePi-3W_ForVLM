"""Check accounting boundaries that previously let heater load enter responses."""
import unittest
import os
from pathlib import Path
import tempfile
from resource_sampler import counter_delta, cpu_usage, response_summary, process_snapshot, driver_bytes


class ResourceAccountingTests(unittest.TestCase):
    def test_driver_bytes_do_not_read_allocation_count(self):
        profile='video memory profile\n| current(byte) | peak(byte) | alloc count | free count\n| 0x2c11ada8 | 0x3b90d828 | 701 | 11\n'
        self.assertEqual(driver_bytes(profile),0x2c11ada8)
        self.assertEqual(driver_bytes('| 0x00000000 | 0x0121b510 | 27 | 27\n'),0)
        with self.assertRaises(ValueError):driver_bytes('| 701 | 11\n')

    def test_missing_process_io_keeps_memory_and_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            # Kernel fields 3..22, including a command name containing a ')'.
            fields=['S']+['0']*19
            fields[7]='7';fields[9]='2';fields[11]='120';fields[12]='30';fields[19]='999'
            (root/'stat').write_text('42 (model ) worker) '+' '.join(fields)+'\n')
            (root/'status').write_text('VmRSS: 1234 kB\nVmHWM: 1400 kB\n')
            (root/'smaps').write_text('Pss: 700 kB\nSwapPss: 12 kB\nPss: 200 kB\n')
            result,errors=process_snapshot(root,42,True)
            self.assertEqual(result['rss_kib'],1234)
            self.assertEqual(result['pss_kib'],900)
            self.assertEqual(result['cpu_seconds'],150/os.sysconf('SC_CLK_TCK'))
            self.assertEqual(result['starttime_ticks'],999)
            self.assertNotIn('io',result)
            self.assertIn('process_io',errors)

    def test_one_of_eight_busy_cores_and_iowait(self):
        old = {'cpu': [0]*8, 'cpu6': [0]*8}
        new = {'cpu': [100, 0, 0, 680, 20, 0, 0, 0], 'cpu6': [100, 0, 0, 0, 0, 0, 0, 0]}
        usage = cpu_usage(old, new)
        self.assertEqual(usage['cpu']['busy_percent'], 12.5)
        self.assertEqual(usage['cpu']['iowait_percent'], 2.5)
        self.assertEqual(usage['cpu6']['busy_percent'], 100)

    def test_reset_is_not_negative_io(self):
        self.assertEqual(counter_delta({'read_bytes': 500, 'write_bytes': 40},
                                       {'read_bytes': 10, 'write_bytes': 50}), {'write_bytes': 10})

    def test_heater_and_teardown_intervals_excluded(self):
        def sample(start, end, busy, pages):
            return {'interval_start_monotonic_seconds': start, 'monotonic_seconds': end,
                    'sampling_seconds': .001, 'interval_seconds': end-start,
                    'cpu_usage': {'cpu': {'busy_percent': busy, 'iowait_percent': 0, 'total_ticks': 800}},
                    'vm_delta': {'pswpin': pages, 'pswpout': 0}, 'page_bytes': 4096,
                    'disk_delta': {'mmcblk1': {'read_bytes': pages*4096, 'write_bytes': 0}}}
        result = response_summary([sample(9, 10.5, 100, 100), sample(10.5, 11.5, 1, 2),
                                   sample(11.5, 12.5, 3, 3), sample(12.5, 13.5, 80, 100)], 10, 13)
        self.assertEqual(result['intervals'], 2)
        self.assertEqual(result['cpu']['cpu']['busy_percent'], 2)
        self.assertEqual(result['swap_in_bytes'], 5*4096)
        self.assertEqual(result['disk_io_bytes']['mmcblk1']['read_bytes'], 5*4096)


if __name__ == '__main__': unittest.main()
