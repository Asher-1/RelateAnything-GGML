#!/usr/bin/env python3
"""Negative tests for stale evidence and interrupted release audits."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import provenance as p
import verify_benchmark_release as release


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.input=self.root/'weights.gguf';self.input.write_bytes(b'original weights')
        self.folder=self.root/'run'
        self.build={'source':'original source','binary_sha256':'original binary'}
        self.mock=patch.object(p,'build_fingerprint',side_effect=lambda *a,**kw:dict(self.build))
        self.mock.start()

    def tearDown(self):
        self.mock.stop();self.tmp.cleanup()

    def run_valid(self):
        with p.recorded_run(self.folder,inputs=[self.input],config={},binary='fixture'):
            (self.folder/'output.rafo').write_bytes(b'original tensors')
        p.validate_run(self.folder,binary='fixture')

    def test_missing_input_replaces_old_complete(self):
        self.run_valid();self.input.unlink()
        with self.assertRaises(FileNotFoundError):
            with p.recorded_run(self.folder,inputs=[self.input],config={}):pass
        self.assertEqual(json.loads((self.folder/'run_provenance.json').read_text())['status'],'failed')

    def test_changed_weights(self):
        self.run_valid();self.input.write_bytes(b'changed weights')
        with self.assertRaisesRegex(RuntimeError,'changed provenance'):p.validate_run(self.folder,binary='fixture')

    def test_changed_binary_and_source(self):
        for field in ('source','binary_sha256'):
            with self.subTest(field=field):
                self.run_valid();old=self.build[field];self.build[field]='changed'
                with self.assertRaisesRegex(RuntimeError,'obsolete binary'):p.validate_run(self.folder,binary='fixture')
                self.build[field]=old

    def test_changed_artifact(self):
        self.run_valid();(self.folder/'output.rafo').write_bytes(b'changed tensors')
        with self.assertRaisesRegex(RuntimeError,'changed provenance'):p.validate_run(self.folder)

    def test_interrupt(self):
        self.run_valid()
        with self.assertRaises(KeyboardInterrupt):
            with p.recorded_run(self.folder,inputs=[self.input],config={}):raise KeyboardInterrupt()
        with self.assertRaisesRegex(RuntimeError,'incomplete run'):p.validate_run(self.folder)

    def test_missing_record(self):
        self.run_valid();(self.folder/'run_provenance.json').unlink()
        with self.assertRaises(FileNotFoundError):p.validate_run(self.folder)

    def test_audit_failure_replaces_old_green(self):
        p.save(self.root/'release_audit.json',{'all_complete':True})
        with patch('sys.argv',['audit','--folder',str(self.root)]),patch.object(release,'audit',side_effect=RuntimeError('fixture mismatch')):
            with self.assertRaisesRegex(RuntimeError,'fixture mismatch'):release.main()
        record=json.loads((self.root/'release_audit.json').read_text())
        self.assertEqual(record['status'],'failed');self.assertFalse(record['all_complete'])


if __name__=='__main__':unittest.main()
