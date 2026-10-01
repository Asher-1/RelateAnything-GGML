import unittest
import numpy as np
from validate_full_graph import stage_statistics

class StageStatisticsTests(unittest.TestCase):
    def test_signed_zero_is_a_bit_difference_without_rmse(self):
        ref=np.array([[0.,1.]],np.float32);candidate=np.array([[-0.,1.]],np.float32)
        result=stage_statistics(candidate,ref)
        self.assertEqual(result['rmse'],0)
        self.assertEqual(result['bit_equal_fraction'],.5)
        self.assertEqual(result['first_nonidentical_value']['index'],[0,0])
        self.assertEqual(result['first_nonidentical_value']['candidate_bits'],0x80000000)

    def test_first_difference_respects_tensor_order(self):
        ref=np.ones((2,3),np.float32);candidate=ref.copy();candidate[0,2]+=1;candidate[1,0]+=4
        result=stage_statistics(candidate,ref)
        self.assertEqual(result['first_nonidentical_value']['index'],[0,2])
        self.assertEqual(result['max_abs'],4)
        self.assertTrue(result['finite'])
