#!/usr/bin/env python3

"""This module is to save the full pytorch model loaded from a checkpoint
as a Torchscript model that can be used in inference.
It can be used as:
```bash
python scripts/save_full_model.py examples/Example_1/gnn_train.yaml -o saved_onnx_files --tag v1
```
As of writing, 2024/02/27, it supports the following models:
* MetricLearning
* InteractionGNN2
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from pytorch_lightning import LightningModule

from acorn import stages
from acorn.core.core_utils import find_latest_checkpoint

torch.use_deterministic_algorithms(True)


def model_save(
    stage_name: str,
    model_name: str,
    checkpoint_path: str | Path,
    output_path: str,
    tag_name: str | None = None,
):
    lightning_model = getattr(getattr(stages, stage_name), "Jitable" + model_name)
    if not issubclass(lightning_model, LightningModule):
        raise ValueError(f"Model {model_name} is not a LightningModule")

    # find the best checkpoint in the checkpoint path
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} not found")

    if checkpoint_path.is_dir():
        checkpoint_path = find_latest_checkpoint(
            checkpoint_path, templates=["best*.ckpt", "*.ckpt"]
        )
        if not checkpoint_path:
            raise ValueError(f"No checkpoint found in {checkpoint_path}")

    # load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    model = lightning_model.load_from_checkpoint(checkpoint_path).to("cpu")
    # print(model.eval())
    # print(vars(model))
    print("edge features:", model.hparams["edge_features"])
    print("node features:", model.hparams["node_features"])


    # save for use in production environment
    out_path = Path(output_path)
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    # perform some dummy inference
    num_spacepoints = 100
    num_edges = 2000
    spacepoint_features = len(model.hparams["node_features"])

    node_features = torch.rand(num_spacepoints, spacepoint_features).to(torch.float32)
    try:
        n_edge_features = len(model.hparams["edge_features"])
        edge_features = torch.rand(num_edges, n_edge_features)
    except:
        edge_features = None

    edge_index = torch.randint(0, 100, (2, num_edges)).to(torch.int64)

    if "MetricLearning" in model_name:
        input_data = (node_features,)
        input_names = ["node_features"]
        dynamic_axes = {"node_features": {0: "num_spacepoints"}}
    else:
        input_data = (node_features, edge_index, edge_features)
        input_names = ["node_features", "edge_index", "edge_features"]
        dynamic_axes = {
            "node_features": {0: "num_spacepoints"},
            "edge_index": {1: "num_edges"},
            "edge_features": {0: "num_edges"},
        }

    output = model(*input_data)
    print("sucessfully run model!")

    torch_script_path = (
        out_path / f"{stage_name}-{model_name}-{tag_name}.pt"
        if tag_name
        else out_path / f"{stage_name}-{model_name}.pt"
    )

    # exporting to torchscript
    with torch.jit.optimized_execution(True):
        script = model.to_torchscript(example_inputs=input_data, method='trace')

    new_output = script(*input_data)
    torch.jit.freeze(script)
    assert new_output.equal(output)

    # save the model
    print(f"Saving model to {torch_script_path}")
    torch.jit.save(script, torch_script_path)
    print(f"Done saving model to {torch_script_path}")

    # try to save the model to ONNX
    try:
        print("Trying to save the model to ONNX")
        onnx_path = (
            out_path / f"{stage_name}-{model_name}-{tag_name}.onnx"
            if tag_name
            else out_path / f"{stage_name}-{model_name}.onnx"
        )
        torch.onnx.export(
            model,
            input_data,
            onnx_path,
            verbose=False,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        print(f"Done saving model to {onnx_path}")
    except Exception as e:
        print(f"Failed to save the model to ONNX: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save a model from a checkpoint")
    parser.add_argument("--config", type=str, help="configuration file")
    parser.add_argument("-o", "--output", type=str, help="Output path", default=".")
    parser.add_argument("--tag", type=str, default=None, help="version name")
    parser.add_argument(
        "-c", "--checkpoint", type=str, help="checkpoint path", default=None
    )
    parser.add_argument("--stage", type=str, help="configuration file")
    parser.add_argument("--model", type=str, help="configuration file")
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        stage = args.stage
        model = args.model
    else:
        config_file = Path(args.config)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file {config_file} not found")

        with open(config_file) as f:
            config = yaml.safe_load(f)

        stage = config["stage"]
        model = config["model"]
        checkpoint = Path(config["stage_dir"]) / "artifacts"

    model_save(stage, model, checkpoint, args.output, args.tag)
