#!/usr/bin/env python3
"""Relocation must preserve original evidence and fail closed on collisions."""
import json
from pathlib import Path
import tempfile
import unittest

import benchmark_layout as layout
from provenance import fingerprint, validate_files


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'legacy'
        self.source.mkdir()
        (self.source/'raw.rafo').write_bytes(b'original output')
        (self.source/'summary.json').write_text('{"measured": true}\n')
        (self.source/'README.md').write_text('[Summary](summary.json) [Raw](raw.rafo)\n')
        self.identity = fingerprint(self.source.iterdir())

    def tearDown(self):
        self.tmp.cleanup()

    def move(self):
        return layout.migrate(self.source, 'fixture', root=self.root/'outputs', apply=True)

    def test_preserve_absolute_provenance_and_report_links(self):
        self.move()
        validate_files(self.identity)
        run = self.root/'outputs/runs/fixture'
        self.assertTrue(self.source.is_symlink())
        self.assertEqual((run/'artifacts/raw.rafo').read_bytes(), b'original output')
        self.assertFalse((run/'reports/raw.rafo').exists())
        report = (run/'reports/README.md').read_text()
        self.assertIn('[Raw](../artifacts/raw.rafo)', report)
        self.assertIn('[Summary](summary.json)', report)
        layout.verify_inventory(run/'artifacts', json.loads((run/'relocation.json').read_text())['files'])

    def test_external_relative_links_keep_original_target(self):
        target = self.root/'outside.md'
        target.write_text('external evidence')
        (self.source/'README.md').write_text('[External](../outside.md)\n')
        self.move()
        report = self.root/'outputs/runs/fixture/reports/README.md'
        link = layout.LINK.search(report.read_text())[2]
        self.assertEqual((report.parent/link).resolve(), target)

    def test_dry_run_has_no_mutations(self):
        layout.migrate(self.source, 'fixture', root=self.root/'outputs')
        self.assertFalse((self.root/'outputs').exists())
        self.assertFalse(self.source.is_symlink())

    def test_relocated_run_without_readme_has_verified_metadata(self):
        (self.source/'README.md').unlink()
        self.move()
        run = self.root/'outputs/runs/fixture'
        self.assertIn('**relocated**', (run/'reports/README.md').read_text())
        self.assertEqual(layout.verify(run)['status'], 'complete')

        record = json.loads((run/'relocation.json').read_text())
        record['status'] = 'failed'
        (run/'relocation.json').write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, 'relocation did not complete'):
            layout.verify(run)

    def test_completed_run_cannot_be_overwritten(self):
        self.move()
        with self.assertRaises(RuntimeError):
            self.move()
        with self.assertRaises(FileExistsError):
            layout.allocate('fixture', root=self.root/'outputs')

    def test_reject_parent_destination_and_unsafe_id(self):
        with self.assertRaises(RuntimeError):
            layout.migrate(self.source, 'fixture', root=self.source/'outputs', apply=True)
        with self.assertRaises(ValueError):
            layout.allocate('../escape', root=self.root/'outputs')

    def test_changed_evidence_cannot_verify(self):
        self.move()
        run = self.root/'outputs/runs/fixture'
        (run/'artifacts/raw.rafo').write_bytes(b'tampered')
        with self.assertRaises(RuntimeError):
            layout.verify_inventory(run/'artifacts', json.loads((run/'relocation.json').read_text())['files'])

    def test_report_is_independent_of_execution_artifact(self):
        self.move()
        run = self.root/'outputs/runs/fixture'
        (run/'reports/summary.json').write_text('{"changed": true}')
        validate_files(self.identity)

    def test_historical_cannot_be_current(self):
        run = layout.allocate('history', root=self.root/'outputs', archive=True)
        record = json.loads((run/'run.json').read_text())
        layout.save(run/'run.json', {**record, 'status': 'relocated'})
        with self.assertRaises(RuntimeError):
            layout.set_current(run, link=self.root/'current')

    def test_fresh_run_has_verified_report_index(self):
        run = layout.allocate('fresh', root=self.root/'outputs')
        (run/'artifacts').mkdir()
        (run/'artifacts/summary.json').write_text('{"measured": true}')
        layout.save(run/'run.json', {'run_id': 'fresh', 'status': 'complete'})
        layout.publish_reports(run)
        self.assertTrue((run/'reports/README.md').is_file())
        self.assertEqual(layout.verify(run)['report_files'], 2)
        (run/'artifacts/summary.json').write_text('{"measured": false}')
        with self.assertRaises(RuntimeError):
            layout.verify(run)

    def test_verify_rejects_changed_report(self):
        self.move()
        run = self.root/'outputs/runs/fixture'
        (run/'reports/summary.json').write_text('{"tampered": true}')
        with self.assertRaises(RuntimeError):
            layout.verify(run)

    def test_partial_execution_cannot_be_current(self):
        run = layout.allocate('partial', root=self.root/'outputs')
        layout.save(run/'run.json', {'status': 'complete', 'config': {
            'suites': ['interfaces'], 'backends': ['cpu', 'cuda', 'vulkan']}})
        with self.assertRaises(RuntimeError):
            layout.set_current(run, link=self.root/'current')

    def test_current_does_not_replace_real_directory(self):
        self.move()
        current = self.root/'current'
        current.mkdir()
        with self.assertRaises(RuntimeError):
            layout.set_current(self.root/'outputs/runs/fixture', link=current)


if __name__ == '__main__':
    unittest.main()
