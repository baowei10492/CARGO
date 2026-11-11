"""Compare frozen-run and interpolation harnesses on the same small sample.

Run in the existing DiffRegCD environment. Does not overwrite paper results.
"""
import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.evaluate_diffregcd_external import ExternalDiffRegCD
from tools.p27_diffregcd_rerun import PRIMARY, load, run_part
from tools.p33_diffregcd_interp import run_dataset
from tools.run_p27_diffregcd_f1 import audit_weights, weight_paths


def digest_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crops', type=int, default=8)
    parser.add_argument('--dataset', default='DSIFN-CD')
    parser.add_argument('--output', required=True)
    parser.add_argument('--original-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.30)
    rerun, config, evaluation = load(Path('configs/p27_diffregcd_rerun.yaml'))
    spec = next(s for s in evaluation['datasets'] if s['name'] == args.dataset)
    spec = {**spec, 'split': config['split']}
    root = Path(config['external']['root'])
    weights = audit_weights(config, args.dataset)
    property = torch.cuda.get_device_properties(0)
    audit = {'dataset': args.dataset, 'crops': args.crops, 'weights': weights,
             'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                             'cuda': torch.version.cuda, 'opencv': cv2.__version__,
                             'numpy': np.__version__, 'gpu': property.name,
                             'multiprocessors': property.multi_processor_count,
                             'cudnn_benchmark': torch.backends.cudnn.benchmark,
                             'cudnn_deterministic': torch.backends.cudnn.deterministic,
                             'matmul_tf32': torch.backends.cuda.matmul.allow_tf32}, 'runs': {}}
    adapter = ExternalDiffRegCD(root, config_path=root / config['datasets'][args.dataset]['config_json'],
                                checkpoint_paths=weight_paths(config, args.dataset))
    captures = {}
    common = dict(margin=int(rerun['crop_margin']), seed=int(config['seed']),
                  threshold=float(config['probability_threshold']), device=torch.device('cuda'),
                  max_crops=args.crops)
    if args.original_only:
        current = run_part('border_free', spec, evaluation, adapter.infer, cap=None, **common)
        pointer = Path('reports/p27_localizer/diffregcd/rerun/LATEST').read_text().strip()
        historical = json.loads((Path(pointer) / 'border_free' / f'{args.dataset}.json').read_text())
        control = json.loads((Path('reports/p33/diffregcd_interp') / f'{args.dataset}.json').read_text())
        audit['runs']['original_full'] = current
        audit['current_control_f1_differences'] = {
            k: abs(current['f1'][f'{k}/{PRIMARY}/scoring'] - control['f1'][f'{k}/{PRIMARY}']) for k in ['0', '3']}
        audit['historical_f1_differences'] = {
            k: current['f1'][f'{k}/{PRIMARY}/scoring'] - historical['f1'][f'{k}/{PRIMARY}/scoring'] for k in ['0', '3']}
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(audit, indent=2), encoding='utf-8')
        print('FULL_AUDIT_CHECKS', json.dumps({k: audit[k] for k in ['current_control_f1_differences',
                                                                  'historical_f1_differences']}), flush=True)
        return
    for label, original in [('original_1', True), ('control_1', False), ('control_2', False)]:
        captures[label] = []

        def infer(a, b):
            output = adapter.infer(a, b)
            captures[label].append({'a': digest_tensor(a), 'b': digest_tensor(b),
                                    'output': digest_tensor(output[PRIMARY]),
                                    'predicted_change_pixels': int((output[PRIMARY][:, 1] > output[PRIMARY][:, 0]).sum())})
            return output

        if original:
            result = run_part('border_free', spec, evaluation, infer, cap=None, **common)
        else:
            result = run_dataset(spec, evaluation, infer, **common)
        audit['runs'][label] = result
        print('AUDIT_RUN', label, json.dumps(result['f1']), flush=True)
    original = audit['runs']['original_1']['f1']
    control = audit['runs']['control_1']['f1']
    audit['harness_f1_differences'] = {k: abs(original[f'{k}/{PRIMARY}/scoring'] - control[f'{k}/{PRIMARY}'])
                                      for k in ['0', '3']}
    audit['control_repeat_f1_differences'] = {k: abs(control[f'{k}/{PRIMARY}'] - audit['runs']['control_2']['f1'][f'{k}/{PRIMARY}'])
                                             for k in ['0', '0_interp', '3']}
    # The original runner emits five views per crop: 0, 1, 2, 3, bordered_3.
    # The interpolation runner emits three: 0, 0_interp, 3.
    comparisons = []
    for i in range(args.crops):
        for level, first_offset, second_offset in [('0', 0, 0), ('3', 3, 2)]:
            first = captures['original_1'][i * 5 + first_offset]
            second = captures['control_1'][i * 3 + second_offset]
            comparisons.append({'crop': i, 'level': level, 'inputs_equal': first['a'] == second['a'] and first['b'] == second['b'],
                                'outputs_equal': first['output'] == second['output']})
    audit['per_crop_harness_checks'] = comparisons
    audit['repeat_output_hashes_equal'] = captures['control_1'] == captures['control_2']
    audit['captures'] = captures
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('AUDIT_CHECKS', json.dumps({k: audit[k] for k in ['harness_f1_differences', 'control_repeat_f1_differences',
                                                         'repeat_output_hashes_equal', 'per_crop_harness_checks']}), flush=True)
    del adapter
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
