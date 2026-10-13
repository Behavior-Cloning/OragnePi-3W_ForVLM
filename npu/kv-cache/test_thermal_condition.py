"""Exercise heater cleanup and band admission without warming the host."""
import unittest
from unittest.mock import patch
import thermal_condition as module


class Heater:
    def __init__(self,*args,**kwargs):self.stopped=False;self.reaped=False
    def poll(self):return 0 if self.stopped else None
    def terminate(self):self.stopped=True
    def wait(self,timeout):self.reaped=True;return 0


class ThermalTests(unittest.TestCase):
    def execute(self,temperatures,clock=None):
        heater=Heater()
        with patch.object(module,'temperature',side_effect=temperatures), \
             patch.object(module.os,'sched_getaffinity',return_value={6,7}), \
             patch.object(module.os,'sched_setaffinity') as affinity, \
             patch.object(module.subprocess,'Popen',return_value=heater), \
             patch.object(module.time,'sleep'), \
             patch.object(module.time,'monotonic',side_effect=clock or [0,1,2,3,4,5,6]):
            try:return module.condition(55000,65000),heater,affinity
            except TimeoutError:
                self.assertTrue(heater.stopped and heater.reaped)
                self.assertEqual(affinity.call_args.args,(0,{6,7}))
                raise

    def test_already_in_band_does_not_heat(self):
        result,heater,_=self.execute([60000,60000])
        self.assertFalse(result['heated']);self.assertFalse(heater.stopped)

    def test_heating_stops_before_acceptance(self):
        result,heater,_=self.execute([40000,40000,70500,62000])
        self.assertTrue(heater.stopped and heater.reaped)
        self.assertEqual(result['after_mC'],62000)
        self.assertEqual(result['heater_workers_alive'],0)

    def test_timeout_reaps_heater(self):
        with self.assertRaises(TimeoutError):self.execute([40000,40000,45000],[0,1,301])


if __name__=='__main__':unittest.main()
