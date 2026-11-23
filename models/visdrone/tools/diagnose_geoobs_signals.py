"""Bounded existing-weight observations: no backward, optimizer or EMA updates."""
import argparse
import copy
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from tools import run_geoobs_g2 as r
from tools import evaluate_geoobs_g2 as e


OUTPUT = Path('/root/autodl-tmp/exps/geoobs_signals_20260927')
RECOVERY = e.OUTPUT
CASES = [682, 10281, 1388, 8843, 11163, 4431, 7056, 1536]
ADDED = ('tools/geoobs_signal_trace.py', 'tests/test_geoobs_signal_trace.py',
         'tools/diagnose_geoobs_signals.py', 'tests/test_geoobs_signals_runner.py')
MODES = [('baseline', 'off'), ('static_ema', 'static'),
         ('distribution_ema', 'distribution')]
RECOVERED_SOURCE_DIGEST = 'e4c495a5926497a1a20473b689daac4da9e1d329b518ce9043433b617c8a7de8'


def select_images(rows):
    by_id = {int(x['image_id']): x for x in rows}
    if len(rows) != 256 or len(by_id) != 256 or not set(CASES) <= set(by_id):
        raise ValueError('requires authenticated 256 probe IDs including historical cases')
    remaining = sorted(set(by_id) - set(CASES))
    uniform = [remaining[int(i)] for i in np.linspace(0, len(remaining)-1, 8)]
    remaining = set(remaining) - set(uniform)
    dense = sorted(remaining, key=lambda i: (
        -int((by_id[i]['gt_area'] <= 1024).sum()), i))[:8]
    strata = dict(historical=list(CASES), uniform=uniform, dense_small=dense)
    ids = sorted(i for group in strata.values() for i in group)
    if len(set(ids)) != 24:
        raise ValueError('selection must have exactly24 unique images')
    return dict(ids=ids, strata=strata)


class ForwardBudget:
    def __init__(self):
        self.attempts = 0

    def consume(self):
        if self.attempts >= 75:
            raise ValueError('75 attempted-forward budget exhausted')
        self.attempts += 1
        return self.attempts


def verify_source_map(pinned, current, allowed_added):
    if set(current) - set(pinned) != set(allowed_added) or any(
        current.get(k) != v for k, v in pinned.items()
    ):
        raise ValueError('unexpected source modification/addition/removal')


def reserve_output(output, protected):
    output = Path(output).resolve()
    for source in protected:
        source = Path(source).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('output/source overlap forbidden')
    output.mkdir(parents=True, exist_ok=False)


def verify_record(actual, expected):
    if r.digest(actual) != r.digest(expected):
        raise ValueError('traced capture differs from authenticated old capture')


def cache_directory(name):
    return (r.OUTPUT if name == 'baseline' else RECOVERY) / 'run' / name


def authenticated_inputs():
    provenance = r.read_json(RECOVERY/'provenance.json')
    if r.digest(provenance['source']) != RECOVERED_SOURCE_DIGEST:
        raise ValueError('recovered source provenance mismatch')
    current = r.source_manifest()
    verify_source_map(provenance['source'], current,
                      {str(r.ROOT/p) for p in ADDED})
    for root in (r.OUTPUT, RECOVERY):
        if (root/'active.lock').exists():
            raise ValueError('existing experiment is active')
    if e.tree_hashes(r.OUTPUT) != provenance['old_artifacts']:
        raise ValueError('original G2 artifacts changed')
    manifest = r.read_json(r.OUTPUT/'manifest.json')
    r.verify_manifest(r.OUTPUT, dict(manifest, source=current))
    split = r.read_json(r.OUTPUT/'split.json')
    baseline, inputs = e.load_baseline(
        r.OUTPUT/'run/baseline', split['probe'], manifest['source_checkpoint_sha256'])
    selection = select_images(baseline)
    input_by_id = {x['image_id']: x['digest'] for x in inputs}
    arms, cached = {}, {}
    for name, mode in MODES:
        directory = cache_directory(name)
        capture_manifest = r.read_json(directory/'manifest.json')
        if name != 'baseline':
            report = r.read_json(r.OUTPUT/'run'/mode/'report.json')
            e.validate_final(r.OUTPUT/'run'/mode/'final.pth', report, e.FINAL_SHA[mode])
            arms[mode] = report
        expected_checkpoint = manifest['source_checkpoint_sha256'] if name == 'baseline' else e.FINAL_SHA[mode]
        if (capture_manifest['checkpoint_sha256'] != expected_checkpoint or
                capture_manifest['source_sha256'] != manifest['source_checkpoint_sha256']):
            raise ValueError('capture checkpoint lineage mismatch')
        if capture_manifest['inputs'] != inputs:
            raise ValueError('capture inputs differ across modes')
        cached[name] = {}
        for image_id in selection['ids']:
            path = directory/f'{image_id}.pt'
            expected = capture_manifest['records'][path.name]
            if r.sha256(path) != expected:
                raise ValueError('old capture SHA mismatch')
            cached[name][image_id] = expected
    return dict(source=current, source_digest=r.digest(current), manifest=manifest,
                selection=selection, inputs={str(i): input_by_id[i] for i in selection['ids']},
                arms=arms, cached=cached, split=split,
                old_trees={str(p): e.tree_hashes(p) for p in (r.OUTPUT, RECOVERY)},
                max_model_forwards=75, backward_calls=0, optimizer_updates=0)


def prepare(output):
    reserve_output(output, [r.OUTPUT, RECOVERY, r.ROOT])
    data = authenticated_inputs()
    if torch.cuda.is_initialized():
        raise ValueError('CPU prepare initialized CUDA')
    r.write_json(output/'prepare.json', data)
    r.status(output, dict(state='prepared', selected=data['selection'], attempts=0,
                          max_attempts=75, optimizer_updates=0, backward_calls=0))
    print('PREPARED ' + json.dumps(data['selection']), flush=True)


def verify_prepared(output, expected_source_digest):
    prepared = r.read_json(output/'prepare.json')
    current = authenticated_inputs()
    if r.digest(current) != r.digest(prepared):
        raise ValueError('prepared sources/inputs/selection/artifacts changed')
    if current['source_digest'] != expected_source_digest:
        raise ValueError('external source pin mismatch')
    return prepared


def record_call(output, budget, name, image_id, kind):
    count = budget.consume()
    entry = dict(attempt=count, mode=name, image_id=image_id, kind=kind)
    with (output/'attempts.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(entry, sort_keys=True)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    r.status(output, dict(state='observing', pid=os.getpid(), attempts=count,
                          max_attempts=75, mode=name, image_id=image_id,
                          optimizer_updates=0, backward_calls=0))


def run(output, expected_source_digest):
    # Reserve the run before loading a model. A failed run cannot be resumed.
    (output/'run').mkdir(exist_ok=False)
    with (output/'active.lock').open('x') as stream:
        stream.write(str(os.getpid())+' inference-only\n')
    budget = ForwardBudget()
    try:
        prepared = verify_prepared(output, expected_source_digest)
        v = r.verifier()
        cfg, instantiate, _, helpers, _ = v.bootstrap(r.ROOT)
        e.verify_config(cfg, r.OUTPUT)
        records, mapping = e.training_records(cfg, prepared['manifest'], prepared['split'])
        from tools.geoobs_signal_trace import GeoObsSignalTrace
        coco = r.read_json(r.ANNOTATION)
        vehicle = [mapping[c['id']] for c in coco['categories'] if c['name'].lower() == 'vehicle']
        if len(vehicle) != 1:
            raise ValueError('exactly one vehicle category required')
        classes = r.contiguous_class_names(coco['categories'], mapping)
        ids = prepared['selection']['ids']
        annotations = {i: [] for i in ids}
        for ann in coco['annotations']:
            if ann['image_id'] in annotations:
                annotations[ann['image_id']].append(ann)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        reports = {}
        for name, mode in MODES:
            work = output/'run'/name
            work.mkdir(exist_ok=False)
            helpers['verify_proposal_flow'].seed_all(20260927)
            model, _, _ = v.construct(copy.deepcopy(cfg), mode, helpers)
            if mode != 'off':
                arm = prepared['arms'][mode]
                if r.digest(r.snapshot_frozen(model)) != arm['frozen_digest']:
                    raise ValueError('frozen source mismatch before loading')
                checkpoint = torch.load(r.OUTPUT/'run'/mode/'final.pth',
                                        map_location='cpu', weights_only=False)
                r.load_audit(model, checkpoint, 'ema', v.CHECKPOINT_SHA)
                del checkpoint
                if r.digest(r.snapshot_frozen(model)) != arm['frozen_digest']:
                    raise ValueError('frozen source mismatch after loading')
            model.eval().float()
            for param in model.parameters():
                param.requires_grad_(False)
            initial_state = r.digest(r.named_tensors(model))
            mapper = instantiate(cfg.dataloader.test.mapper)
            torch.cuda.reset_peak_memory_stats()
            capture_hashes = {}
            with torch.inference_mode():
                for index, image_id in enumerate(ids):
                    raw = records[image_id]
                    sample = r.isolated_map(mapper, raw, 1000+image_id)
                    sample_sha = r.digest(sample)
                    if sample_sha != prepared['inputs'][str(image_id)]:
                        raise ValueError('mapped input mismatch')
                    plain_output = plain_rng = None
                    if index == 0:
                        helpers['verify_proposal_flow'].seed_all(1000+image_id)
                        record_call(output, budget, name, image_id, 'plain_sentinel')
                        plain_sample = copy.deepcopy(sample)
                        plain_output = r.digest(model([plain_sample]))
                        plain_rng = r.digest(v.rng_state())
                        if r.digest(plain_sample) != sample_sha:
                            raise ValueError('plain forward mutated input')
                    helpers['verify_proposal_flow'].seed_all(1000+image_id)
                    record_call(output, budget, name, image_id, 'traced')
                    with GeoObsSignalTrace(model) as trace:
                        outputs = model([sample])
                    end_rng = r.digest(v.rng_state())
                    if index == 0 and (r.digest(outputs) != plain_output or end_rng != plain_rng):
                        raise ValueError('sentinel actual output/RNG passivity failed')
                    trace.finish(outputs, (raw['height'], raw['width']))
                    signals = trace.finish_signals()
                    row = r.make_record(trace, annotations[image_id], raw, mapping,
                                        vehicle[0], model.criterion.matcher, classes)
                    old_path = cache_directory(name)/f'{image_id}.pt'
                    if r.sha256(old_path) != prepared['cached'][name][str(image_id)]:
                        raise ValueError('cached capture changed during run')
                    old = torch.load(old_path, map_location='cpu', weights_only=False)
                    verify_record(row, old)
                    if r.digest(sample) != sample_sha:
                        raise ValueError('trace mutated input')
                    path = work/f'{image_id}.pt'
                    torch.save(dict(image_id=image_id, input_digest=sample_sha,
                                    record=row, signals=signals,
                                    mapped_hw=tuple(sample['image'].shape[-2:]),
                                    output_digest=r.digest(outputs), final_rng_digest=end_rng), path)
                    capture_hashes[path.name] = r.sha256(path)
                    print(f'CAPTURE {name} {index+1}/24 image={image_id} attempts={budget.attempts}', flush=True)
                    del trace, outputs, signals, row, old, sample
            if r.digest(r.named_tensors(model)) != initial_state or any(
                p.grad is not None for p in model.parameters()
            ):
                raise ValueError('model state or gradients changed')
            reports[name] = dict(captures=24, sentinel_output_rng_equal=True,
                                 all_cached_records_exact=True, state_unchanged=True,
                                 state_digest=initial_state, records=capture_hashes,
                                 cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            r.write_json(work/'manifest.json', reports[name])
            del model, mapper
            torch.cuda.empty_cache()
        if budget.attempts != 75:
            raise ValueError('unexpected final attempt count')
        if r.source_manifest() != prepared['source']:
            raise ValueError('source changed during observation')
        for root, before in prepared['old_trees'].items():
            if e.tree_hashes(Path(root)) != before:
                raise ValueError('old experiment artifacts changed')
        r.verify_manifest(r.OUTPUT, dict(prepared['manifest'], source=prepared['source']))
        r.write_json(output/'run/report.json', dict(complete=True, modes=reports,
            model_forward_attempts=75, traced_images=72, untraced_sentinels=3,
            backward_calls=0, optimizer_updates=0, original_artifacts_unchanged=True,
            scientific_gate_reclassified=False, AP_computed=False))
        r.status(output, dict(state='complete', exit_code=0, attempts=75,
                              optimizer_updates=0, backward_calls=0))
        print('SIGNALS_COMPLETE attempts=75 updates=0 backward=0', flush=True)
    except Exception as exc:
        failure = dict(state='failed', exit_code=1, attempts=budget.attempts,
                       reason=f'{type(exc).__name__}: {exc}', optimizer_updates=0,
                       backward_calls=0, traceback=traceback.format_exc())
        r.write_json(output/'failure.json', failure)
        r.status(output, failure)
        raise
    finally:
        (output/'active.lock').unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--expected-source-digest')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not __debug__:
        raise ValueError('python -O forbidden')
    if args.phase == 'prepare':
        prepare(output)
    else:
        if not args.expected_source_digest:
            raise ValueError('run requires external source digest')
        run(output, args.expected_source_digest)


if __name__ == '__main__':
    main()
