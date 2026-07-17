#!/usr/bin/env python3
"""Export the exact Ka epoch-15000 forward graph to fixed-batch ONNX.

The source weights are the parity-locked float32 tensors extracted from
``ace.html``.  A fixed batch dimension lets DirectML fully compile and tune the
graph for the laptop GPU; generate another batch size if benchmarking finds a
better throughput point.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "reference" / "ka_weights_export" / "ka-weights.json"
DEFAULT_OUT = Path(__file__).with_name("native_runtime") / "ka_epoch15000_b128.onnx"


class Graph:
    def __init__(self, batch: int, tensors: dict[str, np.ndarray]) -> None:
        self.batch = batch
        self.tensors = tensors
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.serial = 0

    def name(self, stem: str) -> str:
        self.serial += 1
        return f"{stem}_{self.serial}"

    def init(self, name: str, value: np.ndarray | list[int] | float) -> str:
        arr = np.asarray(value)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype.kind in "iu" and arr.dtype != np.int64:
            arr = arr.astype(np.int64)
        self.initializers.append(numpy_helper.from_array(arr, name=name))
        return name

    def op(self, op_type: str, inputs: list[str], *, stem: str | None = None, **attrs: object) -> str:
        output = self.name(stem or op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attrs))
        return output

    def conv_weight(self, key: str) -> str:
        # TensorFlow/Ka HWIO -> ONNX OIHW.
        value = np.transpose(self.tensors[key], (3, 2, 0, 1)).astype(np.float32)
        return self.init(key.replace(".", "_"), value)

    def channel(self, key: str) -> str:
        return self.init(key.replace(".", "_"), self.tensors[key].reshape(1, -1, 1, 1))

    def softsign_norm(self, x: str, layer: int) -> str:
        mean = self.op("ReduceMean", [x], stem=f"mean{layer}", axes=[1, 2, 3], keepdims=1)
        delta = self.op("Sub", [x, mean], stem=f"delta{layer}")
        square = self.op("Mul", [delta, delta], stem=f"square{layer}")
        eps = self.init(f"eps{layer}", np.asarray([1e-3], dtype=np.float32))
        denom = self.op("Sqrt", [self.op("Add", [square, eps], stem=f"vareps{layer}")], stem=f"denom{layer}")
        unit = self.op("Div", [delta, denom], stem=f"unit{layer}")
        scaled = self.op("Mul", [unit, self.channel(f"gammas.{layer}")], stem=f"gamma{layer}")
        return self.op("Add", [scaled, self.channel(f"betas.{layer}")], stem=f"norm{layer}")

    def conv(self, x: str, key: str, kernel: int, stem: str) -> str:
        pad = kernel // 2
        return self.op(
            "Conv",
            [x, self.conv_weight(key)],
            stem=stem,
            pads=[pad, pad, pad, pad],
            strides=[1, 1],
        )


def read_tensors(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    tensors: dict[str, np.ndarray] = {}
    for name, item in doc["tensors"].items():
        raw = base64.b64decode(item["b64"])
        tensors[name] = np.frombuffer(raw, dtype="<f4").reshape(item["shape"]).copy()
    return doc.get("meta", {}), tensors


def export(
    batch: int,
    output: Path,
    *,
    attention_layers: frozenset[int] = frozenset(),
    trunk_layers: frozenset[int] = frozenset(),
) -> None:
    meta, tensors = read_tensors(WEIGHTS)
    graph = Graph(batch, tensors)
    x = graph.op("Transpose", ["features"], stem="input_nchw", perm=[0, 3, 1, 2])

    h = graph.conv(x, "W_convs.0", 3, "stem_conv")
    h = graph.op("Relu", [graph.softsign_norm(h, 0)], stem="stem_relu")
    position = tensors.get("position_const")
    if position is None:
        position = np.zeros((9, 9, 128), dtype=np.float32)
        for ix in range(9):
            for iy in range(9):
                position[ix, iy, 0] = (ix - 4) / 4
                position[ix, iy, 1] = (iy - 4) / 4
                position[ix, iy, 2] = ix % 3 - 1
                position[ix, iy, 3] = iy % 3 - 1
                position[ix, iy, 4] = ix // 3 - 1
                position[ix, iy, 5] = iy // 3 - 1
    pos_nchw = np.transpose(position, (2, 0, 1))[None, ...].astype(np.float32)
    h = graph.op("Add", [h, graph.init("position_const_nchw", pos_nchw)], stem="stem_position")

    diagnostic_outputs: list[tuple[str, list[int]]] = []
    for layer in range(1, 18):
        if layer % 3:
            kernel = 3 if layer <= 5 else 1
            branch = graph.conv(h, f"W_convs.{layer}", kernel, f"conv{layer}")
            branch = graph.op("Relu", [graph.softsign_norm(branch, layer)], stem=f"relu{layer}")
            h = graph.op("Add", [h, branch], stem=f"residual{layer}")
            if layer in trunk_layers:
                output_name = f"trunk_l{layer}"
                graph.nodes.append(helper.make_node("Identity", [h], [output_name]))
                diagnostic_outputs.append((output_name, [batch, 128, 9, 9]))
            continue

        normalized = graph.softsign_norm(h, layer)
        tokens = graph.op("Transpose", [normalized], stem=f"nhwc{layer}", perm=[0, 2, 3, 1])
        tokens = graph.op(
            "Reshape",
            [tokens, graph.init(f"tokens_shape{layer}", [batch, 81, 128])],
            stem=f"tokens{layer}",
        )
        projections = []
        for kind in ("Q", "K", "V"):
            weight = graph.init(f"W{kind}_{layer}", tensors[f"W{kind}s.{layer}"].astype(np.float32))
            projected = graph.op("MatMul", [tokens, weight], stem=f"{kind.lower()}{layer}")
            projected = graph.op(
                "Reshape",
                [projected, graph.init(f"{kind.lower()}shape{layer}", [batch, 81, 4, 32])],
                stem=f"{kind.lower()}heads{layer}",
            )
            projected = graph.op("Transpose", [projected], stem=f"{kind.lower()}transpose{layer}", perm=[0, 2, 1, 3])
            projections.append(projected)
        q, k, v = projections
        kt = graph.op("Transpose", [k], stem=f"kt{layer}", perm=[0, 1, 3, 2])
        scores = graph.op("MatMul", [q, kt], stem=f"scores{layer}")
        scale = graph.init(f"attention_scale{layer}", np.asarray([1 / math.sqrt(32)], dtype=np.float32))
        scores = graph.op("Mul", [scores, scale], stem=f"scaled_scores{layer}")
        attention = graph.op("Softmax", [scores], stem=f"attention{layer}", axis=-1)
        if layer in attention_layers:
            output_name = f"attention_l{layer}"
            graph.nodes.append(helper.make_node("Identity", [attention], [output_name]))
            diagnostic_outputs.append((output_name, [batch, 4, 81, 81]))
        attended = graph.op("MatMul", [attention, v], stem=f"attended{layer}")
        attended = graph.op("Transpose", [attended], stem=f"unheads{layer}", perm=[0, 2, 1, 3])
        attended = graph.op(
            "Reshape",
            [attended, graph.init(f"attended_shape{layer}", [batch, 9, 9, 128])],
            stem=f"attended_nhwc{layer}",
        )
        attended = graph.op("Transpose", [attended], stem=f"attended_nchw{layer}", perm=[0, 3, 1, 2])
        h = graph.op("Add", [h, attended], stem=f"attention_residual{layer}")
        if layer in trunk_layers:
            output_name = f"trunk_l{layer}"
            graph.nodes.append(helper.make_node("Identity", [h], [output_name]))
            diagnostic_outputs.append((output_name, [batch, 128, 9, 9]))

    value_pre = graph.conv(h, "W_y_head_conv", 3, "value_conv")
    value_pre = graph.op("Add", [value_pre, graph.channel("b_y_head_conv")], stem="value_bias")
    value_mean = graph.op("ReduceMean", [value_pre], stem="value_mean", axes=[1, 2, 3], keepdims=0)
    value = graph.op("Tanh", [value_mean], stem="value_black")

    policy = graph.conv(h, "W_p_head_conv", 3, "policy_conv")
    policy = graph.op("Add", [policy, graph.channel("b_p_head_conv")], stem="policy_bias")
    policy = graph.op("Relu", [policy], stem="policy_relu")
    policy = graph.op("Transpose", [policy], stem="policy_nhwc", perm=[0, 2, 3, 1])
    policy = graph.op(
        "Reshape",
        [policy, graph.init("policy_flat_shape", [batch, 2592])],
        stem="policy_flat",
    )
    w2 = graph.init("W2", tensors["W2"].astype(np.float32))
    b2 = graph.init("b2", tensors["b2"].astype(np.float32))
    logits = graph.op("Add", [graph.op("MatMul", [policy, w2], stem="policy_matmul"), b2], stem="policy_logits")
    probabilities = graph.op("Softmax", [logits], stem="policy", axis=1)
    graph.nodes.append(helper.make_node("Identity", [probabilities], ["policy"]))
    graph.nodes.append(helper.make_node("Identity", [value], ["value_black"]))

    model_graph = helper.make_graph(
        graph.nodes,
        "ka_epoch15000",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [batch, 9, 9, 15])],
        [
            helper.make_tensor_value_info("policy", TensorProto.FLOAT, [batch, 137]),
            helper.make_tensor_value_info("value_black", TensorProto.FLOAT, [batch]),
            *[
                helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
                for name, shape in diagnostic_outputs
            ],
        ],
        initializer=graph.initializers,
    )
    model = helper.make_model(
        model_graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="quoridor-ka-teacher",
        producer_version="1",
    )
    model.ir_version = min(model.ir_version, 10)
    checkpoint_meta = model.metadata_props.add()
    checkpoint_meta.key = "checkpoint"
    checkpoint_meta.value = str(meta.get("checkpoint", "epoch15000.ckpt"))
    batch_meta = model.metadata_props.add()
    batch_meta.key = "batch_size"
    batch_meta.value = str(batch)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output)
    print(f"wrote {output} ({output.stat().st_size:,} bytes), batch={batch}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--attention-layer",
        type=int,
        action="append",
        default=[],
        choices=(3, 6, 9, 12, 15),
        help="also export the selected 4x81x81 self-attention matrix",
    )
    parser.add_argument(
        "--trunk-layer",
        type=int,
        action="append",
        default=[],
        choices=tuple(range(1, 18)),
        help="also export the selected 128x9x9 residual activation",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    export(
        args.batch_size,
        args.out,
        attention_layers=frozenset(args.attention_layer),
        trunk_layers=frozenset(args.trunk_layer),
    )


if __name__ == "__main__":
    main()
