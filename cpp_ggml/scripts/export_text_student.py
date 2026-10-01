#!/usr/bin/env python3
"""Losslessly export the released predicate text student to F32 GGUF."""
import argparse
from pathlib import Path
import torch
from gguf import GGUFWriter
from provenance import save, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(args.output, 'relateanything')
    writer.add_string('ra.graph_kind', 'relateanything_text_student_v1')
    writer.add_string('ra.checkpoint_sha256', sha256(args.checkpoint))
    for key, value in payload['cfg'].items():
        if key != 'full' and value is not None:
            writer.add_uint32('ra.'+key, value)
    for name, tensor in payload['state_dict'].items():
        writer.add_tensor('ts.'+name, tensor.detach().float().cpu().numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    save(args.output.with_suffix('.json'), {'checkpoint_sha256':sha256(args.checkpoint),
         'gguf_sha256':sha256(args.output),'tensors':len(payload['state_dict']),
         'storage':'f32','cfg':payload['cfg']})


if __name__ == '__main__':
    main()
