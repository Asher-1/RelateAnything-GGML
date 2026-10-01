"""Verify rejection of corrupt, incomplete and unsafe cached archives."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from package_test_data import PACKAGE, digest_file, encoded, verify, zip_info


class PackageIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'test.zip'

    def archive(self, *, payload=b'image', image_hash=None, extra=None):
        original_hash = hashlib.sha256(b'image').hexdigest()
        members = {'image.jpg': payload, 'annotations.json': b'{}',
                   'catalog.json': encoded({'samples': [{
                       'image': 'image.jpg', 'image_sha256': image_hash or original_hash,
                       'annotation_manifest': 'annotations.json'}]})}
        rows = [{'file': name, 'bytes': len(b'image') if name == 'image.jpg' else len(content),
                 'sha256': original_hash if name == 'image.jpg' else hashlib.sha256(content).hexdigest()}
                for name, content in members.items()]
        with zipfile.ZipFile(self.path, 'w') as packed:
            for name, content in members.items():
                packed.writestr(zip_info(name), content)
            packed.writestr(zip_info('manifest.json'), encoded({'files': rows}))
            if extra:
                packed.writestr(zip_info(extra), b'bad')
        return {'zip_bytes': self.path.stat().st_size,
                'zip_sha256': digest_file(self.path), 'samples': 1}

    def test_valid_cache_and_extraction_layout(self):
        verify(self.path, self.archive())
        target = Path(self.temporary.name) / 'extract'
        with zipfile.ZipFile(self.path) as packed:
            packed.extractall(target)
        self.assertEqual((target / PACKAGE / 'image.jpg').read_bytes(), b'image')

    def test_download_identity_mismatch(self):
        descriptor = self.archive()
        descriptor['zip_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            verify(self.path, descriptor)

    def test_payload_mismatch_even_with_new_outer_identity(self):
        descriptor = self.archive(payload=b'edited image')
        with self.assertRaisesRegex(ValueError, 'payload hash mismatch'):
            verify(self.path, descriptor)

    def test_catalog_image_identity_mismatch(self):
        descriptor = self.archive(image_hash='0' * 64)
        with self.assertRaisesRegex(ValueError, 'catalog image mismatch'):
            verify(self.path, descriptor)

    def test_path_traversal_is_rejected(self):
        descriptor = self.archive(extra='../outside.txt')
        with self.assertRaisesRegex(ValueError, 'unsafe archive member'):
            verify(self.path, descriptor)

    def test_unlisted_member_is_rejected(self):
        descriptor = self.archive(extra='unlisted.txt')
        with self.assertRaisesRegex(ValueError, 'member list differs'):
            verify(self.path, descriptor)


if __name__ == '__main__':
    unittest.main()
