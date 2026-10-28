#!/usr/bin/env python3
"""Audit environment-free compiled GGML sources, not a guessed backend list."""
import argparse,json,re
from pathlib import Path
from provenance import ROOT,sha256,save

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();files=set();builds={}
    for backend in ('cpu','cuda','vulkan'):
        path=ROOT/f'cpp_ggml/build-{backend}/compile_commands.json'
        builds[backend]=sha256(path)
        for row in json.loads(path.read_text()):
            source=Path(row['file'])
            if source.is_file() and source.is_relative_to(ROOT/'cpp_ggml'):
                files.add(source)
    files.update((ROOT/'cpp_ggml/include').rglob('*.h'))
    files.update((ROOT/'cpp_ggml/include').rglob('*.hpp'))
    files.update((ROOT/'cpp_ggml/third_party/ggml/src/ggml-cuda').glob('*.cuh'))
    pattern=re.compile(r'\b(?:getenv|setenv|unsetenv|putenv|_putenv|_dupenv_s)\s*\(')
    findings=[]
    for path in sorted(files):
        for line,text in enumerate(path.read_text(errors='replace').splitlines(),1):
            if pattern.search(text) and not text.lstrip().startswith('//'):
                findings.append({'path':str(path.relative_to(ROOT)),'line':line,'source':text.strip()})
    save(args.output,{'status':'pass' if not findings else 'fail',
        'scope':'all compiled project/GGML translation units, CUDA headers and public API headers',
        'build_compile_commands_sha256':builds,
        'audited_sources':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(files)},
        'environment_switches':findings})
    if findings:raise RuntimeError(f'environment switches remain: {findings}')
    print(f'{len(files)} compiled sources/headers: explicit options PASS')
if __name__=='__main__':main()
