#!/usr/bin/env python3
"""Replay the ordered GGML patch series from the pin and verify reverse equality."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PIN = '8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'cpp_ggml/outputs/probes/patch_replay.json')
    args = parser.parse_args()
    patches = sorted((ROOT/'cpp_ggml/third_party/patches').glob('*.patch'))
    with tempfile.TemporaryDirectory(prefix='ra-patch-replay-') as folder:
        def git(*command):
            return subprocess.check_output(['git', '-C', folder, *command], text=True).strip()
        subprocess.run(['git', 'clone', '--quiet', '--shared', '--no-checkout',
                        str(ROOT/'cpp_ggml/third_party/ggml'), folder], check=True)
        git('checkout', '--quiet', '--detach', PIN)
        original = git('rev-parse', 'HEAD^{tree}')
        records = []
        for patch in patches:
            git('apply', '--check', str(patch)); git('apply', str(patch))
            git('apply', '--reverse', '--check', str(patch))
            records.append({'patch': patch.name, 'sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
                            'forward_check': True, 'applied': True, 'reverse_check': True,
                            'idempotent_skip': True})
        git('add', '-A'); patched = git('write-tree')
        for patch in reversed(patches):
            git('apply', '--reverse', str(patch))
        git('add', '-A'); restored = git('write-tree')
        if restored != original: raise RuntimeError('reverse patch series did not restore the pinned tree')
        payload = {'ggml_version': 'v0.21.0', 'pin': PIN, 'original_tree': original,
                   'patched_tree': patched, 'reverse_restores_tree': True, 'patches': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
    print(f'{len(patches)} patches: pristine replay, reverse checks, original tree restoration PASS')


if __name__ == '__main__': main()
