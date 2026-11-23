"""Recreate a shared 10-class initialization from the verified zero-update archive."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

SOURCE_SHA256 = "fbb441c67ad7ae3a0a51033e64f6ef49e4e73b13e2ef4a43acaad77c6440abbe"


def adapted_initialization(source):
    source = Path(source)
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Expected the verified zero-training-update common initialization")
    cp = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    if cp["initialization_provenance"]["DIOR_training_updates"] != 0:
        raise ValueError("Initialization is not the common zero-update archive")
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.random.default_generator.manual_seed(42)
        classifier = nn.Linear(256, 10)
        classifier.bias.fill_(math.log(.01 / .99))
        embedding = nn.Embedding(11, 256)
        calibrator_input = nn.Linear(201 + 10, 64)
    state = dict(cp["model"])
    for prefix in ("class_embed", "transformer.decoder.class_embed"):
        for level in range(7):
            for kind, value in classifier.state_dict().items():
                key = f"{prefix}.{level}.{kind}"
                if state[key].shape[0] != 20:
                    raise ValueError(f"Unexpected source classifier: {key}")
                state[key] = value
    state["label_enc.weight"] = embedding.weight.detach()
    for kind, value in calibrator_input.state_dict().items():
        state["refined_score_calibrator.mlp.0." + kind] = value
    return {
        "model": state, "initialization_buffers": cp["initialization_buffers"],
        "initialization_provenance": {
            "source_sha256": digest, "seed": 42, "dataset": "VisDrone2019-DET",
            "num_classes": 10, "training_updates": 0,
        },
    }


def load_into_model(model, cp):
    """Load every model tensor from the paired archive, including nonpersistent buffers."""
    from projects.diffu_dino.quality_checkpoint import restore_initialization_buffers

    if cp["initialization_provenance"].get("dataset") != "VisDrone2019-DET":
        raise ValueError("Wrong initialization")
    state = model.state_dict()
    substate = {key: cp["model"][key] for key in state}
    for key, tensor in state.items():
        supplied = substate[key]
        if supplied.dtype != tensor.dtype or supplied.shape != tensor.shape:
            raise ValueError(f"Incompatible initialization for {key}")
    model.load_state_dict(substate, strict=True)
    restore_initialization_buffers(model, cp)
    return len(substate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cp = adapted_initialization(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(cp, tmp_path)
        os.link(tmp_path, args.output)  # Fail atomically if the output already exists.
    finally:
        tmp_path.unlink()
    print(args.output)


if __name__ == "__main__":
    main()
