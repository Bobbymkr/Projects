import unittest
import numpy as np
from src.control.fuzzy_control import FuzzySet, FuzzyRule, FuzzyController

class TestFuzzySet(unittest.TestCase):
    def test_membership(self):
        # Test cases for FuzzySet membership function
        fs = FuzzySet(low=0, mid=5, high=10)
        self.assertEqual(fs.membership(0), 0.0)
        self.assertEqual(fs.membership(2.5), 0.5)
        self.assertEqual(fs.membership(5), 1.0)
        self.assertEqual(fs.membership(7.5), 0.5)
        self.assertEqual(fs.membership(10), 0.0)
        self.assertEqual(fs.membership(-1), 0.0)
        self.assertEqual(fs.membership(11), 0.0)

class TestFuzzyRule(unittest.TestCase):
    def test_apply(self):
        # Test cases for FuzzyRule apply method
        fs = FuzzySet(low=0, mid=5, high=10)
        fr = FuzzyRule(queue_level=fs, output_timing=100)
        self.assertEqual(fr.apply(5), 100.0)
        self.assertEqual(fr.apply(2.5), 50.0)
        self.assertEqual(fr.apply(0), 0.0)

class TestFuzzyController(unittest.TestCase):
    def setUp(self):
        self.controller = FuzzyController()

    def test_fuzzify(self):
        memberships = self.controller.fuzzify(7)
        # Expected memberships based on the defined fuzzy sets and rules
        # short_queue (0,0,5), medium_queue (3,7,12), long_queue (10,15,20)
        # For queue_length = 7:
        # short_queue: 0 (7 > 5)
        # medium_queue: 1.0 (7 == 7)
        # long_queue: 0 (7 < 10)
        self.assertAlmostEqual(memberships[0], 0.0)
        self.assertAlmostEqual(memberships[1], 1.0)
        self.assertAlmostEqual(memberships[2], 0.0)

    def test_inference(self):
        memberships = [0.5, 0.8, 0.2] # Example memberships
        fuzzy_outputs = self.controller.inference(memberships)
        # Expected fuzzy outputs: membership * output_timing
        # rules: (short_queue, 10), (medium_queue, 20), (long_queue, 30)
        self.assertAlmostEqual(fuzzy_outputs[0], 0.5 * 10)
        self.assertAlmostEqual(fuzzy_outputs[1], 0.8 * 20)
        self.assertAlmostEqual(fuzzy_outputs[2], 0.2 * 30)

    def test_defuzzify(self):
        fuzzy_outputs = [5, 16, 6] # Example fuzzy outputs from test_inference
        # weighted_sum = (5*10) + (16*20) + (6*30) = 50 + 320 + 180 = 550
        # sum_of_outputs = 5 + 16 + 6 = 27
        # defuzzified_value = 550 / 27 = 20.370...
        defuzzified_value = self.controller.defuzzify(fuzzy_outputs)
        self.assertAlmostEqual(defuzzified_value, 550/27)

        # Test with zero fuzzy outputs (should return default timing)
        self.assertEqual(self.controller.defuzzify([0, 0, 0]), 15)

    def test_compute_timing(self):
        # Test with a single queue length
        timing = self.controller.compute_timing([7])
        # For queue_length = 7, medium_queue membership is 1.0, others 0.0
        # fuzzy_outputs = [0, 1.0*20, 0] = [0, 20, 0]
        # defuzzify: (0*10 + 20*20 + 0*30) / (0+20+0) = 400 / 20 = 20
        self.assertAlmostEqual(timing, 20.0)

        # Test with multiple queue lengths
        timing = self.controller.compute_timing([3, 8, 15, 5])
        # avg_queue = (3+8+15+5)/4 = 31/4 = 7.75
        # memberships for 7.75:
        # short_queue: 0
        # medium_queue: (12-7.75)/(12-7) = 4.25/5 = 0.85
        # long_queue: (7.75-10)/(15-10) = -2.25/5 = -0.45 (but should be 0 if outside range)
        # Let's re-evaluate memberships for 7.75 more carefully:
        # short_queue (0,0,5): 0
        # medium_queue (3,7,12): (12-7.75)/(12-7) = 4.25/5 = 0.85
        # long_queue (10,15,20): 0 (since 7.75 < 10)
        # memberships = [0, 0.85, 0]
        # fuzzy_outputs = [0*10, 0.85*20, 0*30] = [0, 17, 0]
        # defuzzify: (0*10 + 17*20 + 0*30) / (0+17+0) = 340 / 17 = 20
        self.assertAlmostEqual(timing, 20.0)

if __name__ == '__main__':
    unittest.main()