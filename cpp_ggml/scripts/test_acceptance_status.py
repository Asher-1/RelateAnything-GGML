#!/usr/bin/env python3
"""Missing measurements and changed pair sets must never produce green status."""
import json
from pathlib import Path
import tempfile
import unittest

from write_acceptance_status import acceptance, MODELS, DTYPES


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        (self.run/'artifacts/full_graph/quantization_audit').mkdir(parents=True)
        (self.run/'artifacts/gallery').mkdir()
        (self.run/'run.json').write_text('{"status": "complete"}')

    def tearDown(self):
        self.tmp.cleanup()

    def status(self, name):
        return next(row['status'] for row in acceptance(self.run)['statuses'] if row['id'] == name)

    def original_rows(self):
        configs={b+'-'+d for b in ('cpu','cuda','vulkan') for d in DTYPES}|{'cuda-q8_0-f32-input'}
        return [{'model':m,'config':k,'bit_equal_fraction':1.,
                 'pair_logit_bit_equal_fraction':1.,'pair_set_jaccard':1.}
                for m in MODELS for k in sorted(configs)]

    def write_originals(self,rows):
        (self.run/'artifacts/gallery/numerical_first_divergence.json').write_text(json.dumps(rows))

    def test_absent_oracle_and_timing_are_pending(self):
        self.assertEqual(self.status('strict_bit_equality'), 'pending')
        self.assertEqual(self.status('strict_faster_than_python'), 'pending')

    def test_equal_matched_logits_do_not_hide_missing_pairs(self):
        rows = [{'model': model, 'dtype': dtype,
                 **{backend+'_same_weights': {'logit_bit_equal_fraction': 1.0,
                                              'pair_logit_bit_equal_fraction': 1.0,
                                              'pair_set_jaccard': 1.0}
                    for backend in ('cuda', 'vulkan')}}
                for model in MODELS for dtype in DTYPES]
        rows[-1]['vulkan_same_weights']['pair_set_jaccard'] = 0.99
        path = self.run/'artifacts/full_graph/quantization_audit/summary.json'
        path.write_text(json.dumps(rows))
        originals=self.original_rows();originals[-1]['pair_set_jaccard']=.99
        self.write_originals(originals)
        self.assertEqual(self.status('strict_bit_equality'), 'open')

    def test_equal_predicates_do_not_hide_pair_score_bits(self):
        rows = [{'model': model, 'dtype': dtype,
                 **{backend+'_same_weights': {'logit_bit_equal_fraction': 1.0,
                     'pair_logit_bit_equal_fraction': 1.0,'pair_set_jaccard': 1.0}
                    for backend in ('cuda', 'vulkan')}}
                for model in MODELS for dtype in DTYPES]
        path=self.run/'artifacts/full_graph/quantization_audit/summary.json'
        path.write_text(json.dumps(rows))
        originals=self.original_rows();self.write_originals(originals)
        self.assertEqual(self.status('strict_bit_equality'), 'complete')
        rows[0]['cuda_same_weights']['pair_logit_bit_equal_fraction']=0.99
        path.write_text(json.dumps(rows))
        # CPU oracle differences cannot substitute for the requested CUDA reference.
        self.assertEqual(self.status('strict_bit_equality'), 'complete')
        originals[0]['pair_logit_bit_equal_fraction']=.99;self.write_originals(originals)
        self.assertEqual(self.status('strict_bit_equality'), 'open')

    def test_same_weight_equality_cannot_certify_original_cuda(self):
        originals=self.original_rows();originals[0]['bit_equal_fraction']=.5
        self.write_originals(originals)
        self.assertEqual(self.status('strict_bit_equality'), 'open')
        self.write_originals(originals[:-1])
        self.assertEqual(self.status('strict_bit_equality'), 'pending')

    def test_faster_partial_matrix_cannot_certify_all_configs(self):
        path = self.run/'artifacts/full_graph/summary.json'
        path.write_text(json.dumps({'models': {
            MODELS[0]: {'python': {'latency_median_ms': 20.0},
                        'cuda-f32': {'latency_median_ms': 10.0}}}}))
        self.assertEqual(self.status('strict_faster_than_python'), 'pending')

    def test_gpu_speed_advantage_cannot_hide_a_slow_cpu_configuration(self):
        configs = {b+'-'+d for b in ('cpu', 'cuda', 'vulkan') for d in DTYPES}
        configs.add('cuda-q8_0-f32-input')
        models = {m: {'python': {'latency_median_ms': 20.},
                      **{k: {'latency_median_ms': 10.} for k in configs}} for m in MODELS}
        path = self.run/'artifacts/full_graph/summary.json'
        path.write_text(json.dumps({'models': models}))
        self.assertEqual(self.status('strict_faster_than_python'), 'complete')
        models[MODELS[0]]['cpu-q8_0']['latency_median_ms'] = 21.
        path.write_text(json.dumps({'models': models}))
        self.assertEqual(self.status('strict_faster_than_python'), 'open')


if __name__ == '__main__':
    unittest.main()
