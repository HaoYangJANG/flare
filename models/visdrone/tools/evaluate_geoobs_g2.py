"""Finish G2 evaluation only; never creates a trainer or resumes optimization."""

import argparse
import copy
import os
from pathlib import Path
import traceback

import torch

from tools import run_geoobs_g2 as r


SOURCE = r.OUTPUT
OUTPUT = Path("/root/autodl-tmp/exps/geoobs_g2_eval_repair_20260927")
CHANGED = ("tools/run_geoobs_g2.py", "tests/test_geoobs_g2_runner.py")
ADDED = ("tools/evaluate_geoobs_g2.py", "tests/test_geoobs_g2_evaluation.py")
FINAL_SHA = {
    "static": "ee66a553c7de4165cb35b50ab07101904e033a5244a52e615228818d9408c11f",
    "distribution": "b82855830a93ac301f39496fba35453033746017b571f7f0682710c761cffc74",
}


def source_delta(old, current, root):
    changed = {str(root / p) for p in CHANGED}
    added = {str(root / p) for p in ADDED}
    if set(current) - set(old) != added or set(old) - set(current):
        raise ValueError("unexpected source additions/removals")
    actual_changed = {p for p in old if old[p] != current[p]}
    if actual_changed != changed:
        raise ValueError("unexpected source modifications")
    return {p: dict(before=old.get(p), after=current[p])
            for p in sorted(changed | added)}


def reserve_output(source, output):
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source/output overlap forbidden")
    if (source / "active.lock").exists():
        raise ValueError("source experiment is active")
    output.mkdir(parents=True, exist_ok=False)
    (output / "run").mkdir()


def tree_hashes(root):
    return {str(p.relative_to(root)): r.sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def load_baseline(directory, probe_ids, source_sha):
    manifest = r.read_json(directory / "manifest.json")
    ids = sorted(probe_ids)
    if len(set(ids)) != len(ids) or [x["image_id"] for x in manifest["inputs"]] != ids:
        raise ValueError("baseline input IDs mismatch")
    expected = {f"{i}.pt" for i in ids}
    if set(manifest["records"]) != expected or {p.name for p in directory.glob("*.pt")} != expected:
        raise ValueError("baseline capture files mismatch")
    if any(manifest[k] != source_sha for k in ("source_sha256", "checkpoint_sha256")):
        raise ValueError("baseline source lineage mismatch")
    rows = []
    for image_id, entry in zip(ids, manifest["inputs"]):
        path = directory / f"{image_id}.pt"
        if r.sha256(path) != manifest["records"][path.name]:
            raise ValueError("baseline capture hash mismatch")
        row = torch.load(path, map_location="cpu", weights_only=False)
        if row["image_id"] != image_id or len(entry["digest"]) != 64:
            raise ValueError("baseline row/input IDs or digest mismatch")
        rows.append(row)
    return rows, manifest["inputs"]


def validate_final(path, report, expected_sha):
    if (report["attempts"], report["applied"], report["skipped"]) != (1000, 1000, 0):
        raise ValueError("completed G2 budget required")
    if report["final_sha256"] != expected_sha or r.sha256(path) != expected_sha:
        raise ValueError("final checkpoint hash mismatch")


def preflight(source, expected_source_digest):
    manifest = r.read_json(source / "manifest.json")
    current = r.source_manifest()
    if r.digest(current) != expected_source_digest:
        raise ValueError("reviewed source digest mismatch")
    delta = source_delta(manifest["source"], current, r.ROOT)
    # Validate the amended in-memory view; NEVER rewrite the old manifest.
    amended = dict(manifest, source=current)
    r.verify_manifest(source, amended)
    if not r.read_json(source / "prepare/report.json").get("pass"):
        raise ValueError("original prepare did not pass")
    if (source / "active.lock").exists():
        raise ValueError("source experiment is active")
    split = r.read_json(source / "split.json")
    schedule = r.read_json(source / "schedule.json")
    if len(split["probe"]) != 256 or len(schedule) != 4000:
        raise ValueError("original G2 split/schedule size mismatch")
    arms = {}
    for mode in ("static", "distribution"):
        report = r.read_json(source / "run" / mode / "report.json")
        validate_final(source / "run" / mode / "final.pth", report, FINAL_SHA[mode])
        if len(report["inputs"]) != 4000 or [
            {k: row[k] for k in ("image_id", "mapper_seed")} for row in report["inputs"]
        ] != schedule:
            raise ValueError("completed arm input schedule mismatch")
        arms[mode] = report
    for key in ("initial_digest", "frozen_digest", "inputs"):
        if arms["static"][key] != arms["distribution"][key]:
            raise ValueError("completed arms were not paired: " + key)
    cache = load_baseline(source / "run/baseline", split["probe"], manifest["source_checkpoint_sha256"])
    return manifest, amended, delta, split, arms, cache


def verify_config(cfg, source):
    for mode in ("static", "distribution"):
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg.model.flow_spatial_observation_mode = mode
        old = r.read_json(source / f"effective_config_{mode}.json")
        old.pop("g2_runner_effective_training")
        if r.plain_config(arm_cfg) != old:
            raise ValueError("effective config changed: " + mode)


def training_records(cfg, manifest, split):
    from detectron2.data import DatasetCatalog, MetadataCatalog

    name = cfg.dataloader.train.dataset.names
    if isinstance(name, (list, tuple)):
        if len(name) != 1:
            raise ValueError("one TRAIN dataset required")
        name = name[0]
    metadata = MetadataCatalog.get(name)
    if Path(metadata.json_file).resolve() != r.ANNOTATION.resolve():
        raise ValueError("registered dataset is not authorized TRAIN")
    records = {int(x["image_id"]): x for x in DatasetCatalog.get(name)}
    for image_id in split["probe"]:
        if Path(records[image_id]["file_name"]).resolve() != Path(manifest["images"][str(image_id)]["path"]).resolve():
            raise ValueError("TRAIN image path mismatch")
    return records, metadata.thing_dataset_id_to_contiguous_id


def evaluate(source, output, expected_source_digest):
    manifest, amended, delta, split, arms, cache = preflight(source, expected_source_digest)
    before = tree_hashes(source)
    r.write_json(output / "provenance.json", dict(
        source_root=str(source), source_delta=delta, source=amended["source"],
        old_artifacts=before, baseline_reused=True, optimizer_updates=0,
        checkpoint_sha256=FINAL_SHA, source_digest=expected_source_digest,
    ))
    v = r.verifier()
    cfg, instantiate, _trainer_class, helpers, _ = v.bootstrap(r.ROOT)
    verify_config(cfg, source)
    if v.CHECKPOINT_SHA != manifest["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint lineage changed")
    records, mapping = training_records(cfg, manifest, split)
    coco = r.read_json(r.ANNOTATION)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def progress(name, count, total):
        r.status(output, dict(state="evaluating", mode=name, completed=count,
                              total=total, optimizer_updates=0, pid=os.getpid()))

    gate = r.probe(output, cfg, instantiate, helpers, v, manifest, records,
                   split, coco, mapping, arms, checkpoint_root=source,
                   cached_baseline=cache, on_progress=progress)
    r.verify_manifest(source, amended)
    if tree_hashes(source) != before:
        raise ValueError("original experiment artifacts changed")
    r.write_json(output / "run/report.json", dict(
        complete=True, scientific_gate=gate, optimizer_updates=0,
        baseline_reused_images=256, candidate_images_per_model=256,
        original_artifacts_unchanged=True,
    ))
    r.status(output, dict(state="complete", exit_code=0, scientific_gate=gate,
                          optimizer_updates=0))
    print("EVAL_COMPLETE " + str(gate), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--expected-source-digest", required=True)
    args = parser.parse_args(argv)
    owned = False
    try:
        if not __debug__:
            raise ValueError("python -O forbidden")
        reserve_output(args.source, args.output)
        owned = True
        with (args.output / "active.lock").open("x") as stream:
            stream.write(str(os.getpid()) + " eval-only\n")
        r.status(args.output, dict(state="preflight", pid=os.getpid(), optimizer_updates=0))
        evaluate(args.source, args.output, args.expected_source_digest)
        return 0
    except Exception as exc:
        failure = dict(state="failed", reason=f"{type(exc).__name__}: {exc}",
                       exit_code=1, optimizer_updates=0)
        if owned:
            r.write_json(args.output / "failure.json", dict(failure, traceback=traceback.format_exc()))
            r.status(args.output, failure)
        traceback.print_exc()
        return 1
    finally:
        if owned and (args.output / "active.lock").exists():
            (args.output / "active.lock").unlink()


if __name__ == "__main__":
    raise SystemExit(main())
