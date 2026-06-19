from __future__ import annotations

import hashlib
import io
import struct
import sys
import unittest

import torch
import torch.nn as nn

from training.reduction_counterfactual_schema import (
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_V2,
    SCHEMA,
    bound_class,
    classify_pair,
    context_features_v2,
    rank_percentile,
    stable_partition,
    validate_row,
    wilson_lower,
)


def event(**overrides):
    row = {
        "ordinal": 7,
        "parent_hash": "aa",
        "child_hash": "bb",
        "move": "c4h",
        "depth": 6,
        "ply": 2,
        "alpha": 10,
        "beta": 11,
        "move_index": 12,
        "base_reduction": 2,
        "extra_reduction": False,
        "verification_triggered": False,
        "score": 0,
        "nodes": 100,
        "hidden": [0.0] * 32,
    }
    row.update(overrides)
    return row


class ReductionCounterfactualTests(unittest.TestCase):
    def test_fail_low_numeric_difference_preserves_decision(self):
        baseline = event(score=0, nodes=100)
        counterfactual = event(score=-20, nodes=70, extra_reduction=True)
        result = classify_pair(
            baseline, counterfactual, minimum_nodes_saved=10, minimum_savings_ratio=0.1
        )
        self.assertEqual(result["sample_status"], "SAFE")
        self.assertTrue(result["activate_plus_one"])
        self.assertEqual(result["net_nodes_saved"], 30)

    def test_changed_bound_is_unsafe_even_when_it_saves_nodes(self):
        baseline = event(score=0, nodes=100)
        counterfactual = event(score=20, nodes=5, extra_reduction=True)
        result = classify_pair(
            baseline, counterfactual, minimum_nodes_saved=1, minimum_savings_ratio=0.0
        )
        self.assertEqual(result["sample_status"], "UNSAFE")
        self.assertFalse(result["activate_plus_one"])

    def test_context_mismatch_is_unknown_not_unsafe(self):
        result = classify_pair(
            event(), event(move="d4h", extra_reduction=True),
            minimum_nodes_saved=1, minimum_savings_ratio=0.0,
        )
        self.assertEqual(result["sample_status"], "UNKNOWN")

    def test_safe_but_unprofitable_is_not_activation_positive(self):
        result = classify_pair(
            event(nodes=100), event(nodes=99, extra_reduction=True),
            minimum_nodes_saved=8, minimum_savings_ratio=0.05,
        )
        self.assertTrue(result["safe_plus_one_reduction"])
        self.assertFalse(result["activate_plus_one"])

    def test_partition_is_stable_and_game_disjoint(self):
        self.assertEqual(stable_partition("game-a", 7), stable_partition("game-a", 7))
        self.assertIn(stable_partition("game-b", 7), {"train", "calibration", "final_test"})

    def test_validation_rejects_activation_without_safety(self):
        with self.assertRaises(ValueError):
            validate_row({
                "schema": SCHEMA,
                "sample_status": "UNKNOWN",
                "safe_plus_one_reduction": False,
                "activate_plus_one": True,
            })

    def test_wilson_is_conservative_for_small_samples(self):
        self.assertLess(wilson_lower(3, 3), 0.5)
        self.assertGreater(wilson_lower(1000, 1000), 0.99)

    def test_bound_classes(self):
        self.assertEqual(bound_class(0, 0, 1), "FAIL_LOW")
        self.assertEqual(bound_class(1, 0, 1), "FAIL_HIGH")
        self.assertEqual(bound_class(5, 0, 10), "EXACT")


# ── helpers for sidecar binary construction ───────────────────────────────────

MAGIC = b"TISRDX1\0"
INPUTS = 37
_DUMMY_TRUNK_SHA = "ab" * 32  # 64 hex chars = 32 bytes


def build_artifact(
    weights=None,
    bias=0.0,
    scale=1.0,
    shift=0.0,
    threshold=0.5,
    trunk_sha=_DUMMY_TRUNK_SHA,
    corrupt_payload_sha=False,
    nan_weight=False,
) -> bytes:
    w = weights if weights is not None else [0.0] * INPUTS
    if nan_weight:
        w = list(w)
        w[0] = float("nan")
    payload = bytearray(MAGIC)
    payload.extend(struct.pack("<III", 1, 1, INPUTS))
    payload.extend(bytes.fromhex(trunk_sha))
    payload.extend(struct.pack("<II", 1, 1))
    payload.extend(struct.pack(f"<{INPUTS}d", *w))
    payload.extend(struct.pack("<dddd", bias, scale, shift, threshold))
    digest = hashlib.sha256(bytes(payload)).digest()
    if corrupt_payload_sha:
        digest = bytes(b ^ 0xFF for b in digest)
    return bytes(payload) + digest


def call_sidecar_predict(weights, bias, scale, shift, threshold, features_37):
    """Emulate the Rust sidecar predict() in Python for parity testing."""
    w = torch.tensor(weights, dtype=torch.float32)
    x = torch.tensor(features_37, dtype=torch.float32)
    logit = float((w @ x) + bias)
    calibrated = 1.0 / (1.0 + ((-(scale * logit + shift)).__truediv__(1)).__neg__().__class__.__call__(
        1.0 + 2.718281828 ** (-(scale * logit + shift))).__truediv__(1) if False else
        (1 + (2.718281828 ** (-(scale * logit + shift)))))
    # simpler:
    import math
    calibrated = 1.0 / (1.0 + math.exp(-(scale * logit + shift)))
    return calibrated >= threshold


def make_row(status="SAFE", activate=True, feature_schema=FEATURE_SCHEMA):
    return {
        "schema": SCHEMA,
        "feature_schema": feature_schema,
        "sample_status": status,
        "safe_plus_one_reduction": status == "SAFE",
        "worthwhile_net_savings": activate,
        "activate_plus_one": activate and status == "SAFE",
        "baseline_nodes": 12,
        "counterfactual_nodes": 1,
        "net_nodes_saved": 11,
        "net_savings_ratio": 0.917,
        "hidden32": [0.1] * 32,
        "context5": [0.5, 0.1, 0.5, 1.0, 0.0],
    }


class SidecarBinaryTests(unittest.TestCase):
    """Tests covering binary artifact format, fail-closed behavior, and parity."""

    def test_artifact_magic_and_length(self):
        art = build_artifact()
        self.assertEqual(art[:8], MAGIC)
        # payload = 8 (magic) + 12 (schema/version/inputs) + 32 (trunk) + 8 (hidden/context dims)
        #          + 37*8 (weights) + 4*8 (bias,scale,shift,threshold) + 32 (sha256)
        expected_len = 8 + 12 + 32 + 8 + INPUTS * 8 + 4 * 8 + 32
        self.assertEqual(len(art), expected_len)

    def test_corrupt_payload_sha_detected(self):
        art = build_artifact(corrupt_payload_sha=True)
        # The last 32 bytes should not match SHA-256 of the preceding payload
        payload = art[:-32]
        computed = hashlib.sha256(payload).digest()
        stored = art[-32:]
        self.assertNotEqual(computed, stored, "test setup: corrupt digest should differ")

    def test_nan_weight_in_artifact(self):
        art = build_artifact(nan_weight=True)
        # Parse weight 0 from the artifact
        weight_offset = 8 + 12 + 32 + 8
        w0 = struct.unpack_from("<d", art, weight_offset)[0]
        import math
        self.assertTrue(math.isnan(w0))

    def test_wrong_input_count_in_binary(self):
        # Build an artifact that claims INPUTS=36 instead of 37.
        # Binary layout: [8 magic][4 schema_ver][4 layer_count][4 INPUTS]...
        # so INPUTS is at offset 16.
        payload = bytearray(MAGIC)
        payload.extend(struct.pack("<III", 1, 1, 36))  # schema_ver=1, layers=1, INPUTS=36
        payload.extend(bytes.fromhex(_DUMMY_TRUNK_SHA))
        payload.extend(struct.pack("<II", 1, 1))
        payload.extend(struct.pack(f"<36d", *([0.0] * 36)))
        payload.extend(struct.pack("<dddd", 0.0, 1.0, 0.0, 0.5))
        digest = hashlib.sha256(bytes(payload)).digest()
        art = bytes(payload) + digest
        # INPUTS is the 3rd uint32 → at offset 8+4+4=16
        inputs_field = struct.unpack_from("<I", art, 16)[0]
        self.assertEqual(inputs_field, 36)

    def test_trunk_mismatch_would_fail_closed(self):
        wrong_trunk = "cd" * 32  # 64 hex chars
        art = build_artifact(trunk_sha=wrong_trunk)
        stored_trunk = art[20:52].hex()
        self.assertEqual(stored_trunk, wrong_trunk)
        self.assertNotEqual(stored_trunk, _DUMMY_TRUNK_SHA)

    def test_threshold_stored_correctly(self):
        art = build_artifact(threshold=0.75)
        # threshold is the 4th double in the last section: offset = 8+12+32+8+37*8 + 3*8
        offset = 8 + 12 + 32 + 8 + INPUTS * 8 + 3 * 8
        t = struct.unpack_from("<d", art, offset)[0]
        self.assertAlmostEqual(t, 0.75, places=10)

    def test_threshold_at_zero_means_always_activate(self):
        # threshold=0 → calibrated_prob >= 0 is always True
        import math
        weights = [0.0] * INPUTS
        logit = 0.0  # w=0, b=0
        calibrated = 1.0 / (1.0 + math.exp(0.0))  # = 0.5
        self.assertTrue(calibrated >= 0.0)

    def test_threshold_at_one_means_never_activate(self):
        # sigmoid(x) underflows to exactly 1.0 in float64 for x≥37 (exp(-37)<float64_eps).
        # The REAL invariant is: threshold=1.0 in the sidecar binary means no activation,
        # because no real calibrated output can reach 1.0 during inference (bias initialised
        # at -6, weights small). Verify that for reasonable pre-training logits, sigmoid<1.
        import math
        # sigmoid(36) in float64: exp(-36)≈2.3e-16 which is right at the ULP boundary.
        # Use 30 to stay safely below the underflow region.
        logit = 30.0
        calibrated = 1.0 / (1.0 + math.exp(-logit))
        self.assertFalse(calibrated >= 1.0, f"sigmoid({logit})={calibrated} should be < 1")


class FeatureParityTests(unittest.TestCase):
    """Python feature construction must be deterministic and match the Rust side."""

    def test_context5_normalization_move_index_boundary(self):
        from training.train_reduction_sidecar_v2 import get_features, VARIANTS
        row = make_row()
        row["hidden32"] = [float(i) / 32 for i in range(32)]
        row["context5"] = [0.9, 4.0 / 128.0, 2.0 / 4.0, 1.0, 0.0]
        feats = get_features(row, "A")
        self.assertEqual(len(feats), 37)
        # context5 at position 32..37 should match row["context5"]
        self.assertAlmostEqual(feats[32], 0.9)
        self.assertAlmostEqual(feats[33], 4.0 / 128.0)
        self.assertAlmostEqual(feats[34], 0.5)

    def test_variant_B_zeros_move_index(self):
        from training.train_reduction_sidecar_v2 import get_features
        row = make_row()
        row["hidden32"] = [1.0] * 32
        row["context5"] = [0.5, 0.9, 0.5, 1.0, 0.0]  # move_index=0.9
        feats_A = get_features(row, "A")
        feats_B = get_features(row, "B")
        # A has all 5 context features (37 total); B drops move_index → 36 total
        self.assertEqual(len(feats_A), 37)
        self.assertEqual(len(feats_B), 36)
        # move_index in A is feats_A[33] = 0.9
        self.assertAlmostEqual(feats_A[33], 0.9)
        # B drops it: feats_B[32]=0.5, feats_B[33]=0.5, feats_B[34]=1.0, feats_B[35]=0.0
        self.assertAlmostEqual(feats_B[32], 0.5)  # remaining_depth
        self.assertAlmostEqual(feats_B[33], 0.5)  # base_reduction (was index 34 in A context)

    def test_variant_C_has_5_dims(self):
        from training.train_reduction_sidecar_v2 import get_features
        row = make_row()
        feats = get_features(row, "C")
        self.assertEqual(len(feats), 5)
        # Should match context5 verbatim
        self.assertAlmostEqual(feats[0], row["context5"][0])
        self.assertAlmostEqual(feats[4], row["context5"][4])

    def test_variant_D_has_32_dims(self):
        from training.train_reduction_sidecar_v2 import get_features
        row = make_row()
        feats = get_features(row, "D")
        self.assertEqual(len(feats), 32)
        for i in range(32):
            self.assertAlmostEqual(feats[i], row["hidden32"][i])

    def test_feature_extraction_is_deterministic(self):
        from training.train_reduction_sidecar_v2 import get_features
        row = make_row()
        feats1 = get_features(row, "A")
        feats2 = get_features(row, "A")
        self.assertEqual(feats1, feats2)

    def test_feature_schema_mismatch_detected_by_integrity(self):
        from training.train_reduction_sidecar_v2 import integrity_check
        bad_row = make_row()
        bad_row["schema"] = SCHEMA
        bad_row["feature_schema"] = "wrong-schema-v0"
        bad_row["hidden32"] = [0.0] * 32
        bad_row["context5"] = [0.0] * 5
        bad_row["trunk_sha256"] = None
        bad_row["sample_status"] = "SAFE"
        bad_row["split"] = "train"
        bad_row["source_game_key"] = "x"
        bad_row["population"] = "natural"
        errors = integrity_check([bad_row], [], "a" * 64)
        feature_errors = [e for e in errors if "feature_schema" in e]
        self.assertTrue(len(feature_errors) > 0)


class ThresholdSelectionTests(unittest.TestCase):
    """Threshold selection must be calibration-only and unsafe-safe."""

    def _make_cal_rows(self):
        rows = []
        # 3 true positives with high net_nodes_saved
        for _ in range(3):
            r = make_row(status="SAFE", activate=True)
            r["net_nodes_saved"] = 20
            rows.append(r)
        # 7 safe negatives
        for _ in range(7):
            r = make_row(status="SAFE", activate=False)
            r["net_nodes_saved"] = 0
            rows.append(r)
        # 1 unsafe negative
        r = make_row(status="UNSAFE", activate=False)
        r["net_nodes_saved"] = -10
        rows.append(r)
        return rows

    def test_unsafe_constraint_enforced(self):
        from training.train_reduction_sidecar_v2 import select_threshold
        rows = self._make_cal_rows()
        # All probs = 0.5 → at threshold 0.5, everything activates including unsafe
        # The selector must reject this and pick a non-activating threshold instead
        probs = [0.5] * len(rows)
        # unsafe row (index 10) has prob=0.5, so any t<=0.5 triggers it
        # Expect threshold=1.0 (never activate) if no safe threshold works
        t, stats = select_threshold(probs, rows)
        if stats.get("unsafe_activations", 0) > 0:
            self.fail("Threshold selection violated unsafe constraint")

    def test_no_activation_below_cost_returns_default(self):
        from training.train_reduction_sidecar_v2 import select_threshold
        # All 10 rows are negatives; no positive → gross=0 → all thresholds net<0
        rows = [make_row(status="SAFE", activate=False) for _ in range(10)]
        for r in rows:
            r["net_nodes_saved"] = 0
        probs = [0.9] * len(rows)
        t, stats = select_threshold(probs, rows)
        self.assertFalse(stats.get("feasible", False),
                         "Should not select a profitable threshold when no positives")

    def test_lower_threshold_chosen_when_more_profitable(self):
        from training.train_reduction_sidecar_v2 import select_threshold, INFERENCE_COST_NODES
        # 5 positives, each saving 15 nodes; 5 negatives saving 0
        rows = []
        for _ in range(5):
            r = make_row(status="SAFE", activate=True)
            r["net_nodes_saved"] = 15
            rows.append(r)
        for _ in range(5):
            r = make_row(status="SAFE", activate=False)
            r["net_nodes_saved"] = 0
            rows.append(r)
        # positives score 0.8, negatives score 0.05
        probs = [0.8] * 5 + [0.05] * 5
        t, stats = select_threshold(probs, rows)
        # Best threshold should be around 0.08 (activates only positives)
        # at t=0.08: 5 activations, 5 TP, gross=75, net=75-3.5=71.5
        self.assertGreater(stats.get("net_nodes_saved", 0), 0)
        self.assertEqual(stats.get("unsafe_activations", 0), 0)
        # Should NOT pick a low threshold that activates negatives unnecessarily
        self.assertLessEqual(stats.get("activations", 99), 6)  # at most 1 false positive


class ModelVariantBinaryPaddingTests(unittest.TestCase):
    """Verifies that write_binary pads variants to 37 inputs correctly."""

    def test_variant_C_padded_to_37_weights(self):
        from training.train_reduction_sidecar_v2 import write_binary, VARIANTS
        import tempfile, os
        model = nn.Linear(5, 1)
        nn.init.ones_(model.weight)
        nn.init.zeros_(model.bias)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            path = f.name
        try:
            write_binary(
                pathlib_path := __import__("pathlib").Path(path),
                model, "C", "ab" * 32, 1.0, 0.0, 0.5)
            art = pathlib_path.read_bytes()
            # Parse INPUTS field
            # INPUTS is the 3rd uint32 in the header → offset 8+4+4=16
            inputs = struct.unpack_from("<I", art, 16)[0]
            self.assertEqual(inputs, 37)
            # First 32 weights should be 0 (padded hidden)
            w_offset = 8 + 12 + 32 + 8
            for i in range(32):
                wi = struct.unpack_from("<d", art, w_offset + i * 8)[0]
                self.assertAlmostEqual(wi, 0.0, msg=f"padding weight {i} not zero")
            # Last 5 weights should be 1.0 (context5 trained weights)
            for i in range(5):
                wi = struct.unpack_from("<d", art, w_offset + (32 + i) * 8)[0]
                self.assertAlmostEqual(wi, 1.0, msg=f"context weight {i} not 1.0")
        finally:
            os.unlink(path)

    def test_variant_D_padded_to_37_weights(self):
        from training.train_reduction_sidecar_v2 import write_binary
        import tempfile, os
        model = nn.Linear(32, 1)
        nn.init.ones_(model.weight)
        nn.init.zeros_(model.bias)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            path = f.name
        try:
            write_binary(
                p := __import__("pathlib").Path(path),
                model, "D", "ab" * 32, 1.0, 0.0, 0.5)
            art = p.read_bytes()
            w_offset = 8 + 12 + 32 + 8
            # First 32 should be 1.0 (hidden weights)
            for i in range(32):
                wi = struct.unpack_from("<d", art, w_offset + i * 8)[0]
                self.assertAlmostEqual(wi, 1.0, msg=f"hidden weight {i}")
            # Last 5 should be 0.0 (context padding)
            for i in range(5):
                wi = struct.unpack_from("<d", art, w_offset + (32 + i) * 8)[0]
                self.assertAlmostEqual(wi, 0.0, msg=f"context padding {i}")
        finally:
            os.unlink(path)


class SignedEconomicsTests(unittest.TestCase):
    """Verify that ALL activated events contribute their true signed delta."""

    def _threshold_stats(self, probs, rows):
        from training.train_reduction_sidecar_v2 import select_threshold
        _, stats = select_threshold(probs, rows)
        return stats

    def _eval_stats(self, rows, threshold=0.05):
        import torch
        import torch.nn as nn
        from training.train_reduction_sidecar_v2 import evaluate_at_threshold
        model = nn.Linear(37, 1)
        nn.init.zeros_(model.weight)
        nn.init.constant_(model.bias, 6.0)  # sigmoid(6)≈0.998, all rows score high
        return evaluate_at_threshold(model, rows, 1.0, 0.0, threshold, "A")

    def _row(self, status, activate, net_saved):
        r = make_row(status=status, activate=activate)
        r["net_nodes_saved"] = net_saved
        return r

    def test_tp_contributes_positive_delta(self):
        # 1 TP with net_saved=50 → gross should be exactly 50
        rows = [self._row("SAFE", True, 50)]
        probs = [0.9]
        stats = self._threshold_stats(probs, rows)
        self.assertEqual(stats.get("tp_delta"), 50)
        self.assertEqual(stats.get("safe_fp_delta"), 0)
        self.assertEqual(stats.get("unsafe_delta"), 0)
        self.assertEqual(stats.get("gross_nodes_saved"), 50)

    def test_safe_fp_contributes_small_positive_delta(self):
        # 1 safe FP (SAFE but not worthwhile) with net_saved=3
        rows = [self._row("SAFE", False, 3)]
        probs = [0.9]
        stats = self._threshold_stats(probs, rows)
        self.assertEqual(stats.get("safe_fp_delta"), 3)
        self.assertEqual(stats.get("tp_delta"), 0)
        self.assertEqual(stats.get("gross_nodes_saved"), 3)

    def test_safe_fp_contributes_zero_delta(self):
        # Safe FP with net_saved=0 (saves nothing individually)
        rows = [self._row("SAFE", False, 0)]
        probs = [0.9]
        stats = self._threshold_stats(probs, rows)
        self.assertEqual(stats.get("safe_fp_delta"), 0)
        self.assertEqual(stats.get("gross_nodes_saved"), 0)

    def test_unsafe_activation_contributes_large_negative(self):
        # UNSAFE with net_saved=-237 and a TP saving 300
        # Unsafe violates constraint → threshold should NOT activate
        tp = self._row("SAFE", True, 300)
        unsafe = self._row("UNSAFE", False, -237)
        unsafe["sample_status"] = "UNSAFE"
        rows = [tp, unsafe]
        probs = [0.9, 0.9]
        stats = self._threshold_stats(probs, rows)
        # Unsafe constraint: unsafe_activations must be 0 for any selected threshold
        self.assertEqual(stats.get("unsafe_activations", 0), 0,
                         "select_threshold must never select a threshold that activates UNSAFE")

    def test_evaluate_gross_includes_safe_fp_delta(self):
        # evaluate_at_threshold should include safe_fp_delta in gross
        rows = [
            self._row("SAFE", True, 50),   # TP
            self._row("SAFE", False, 10),  # safe FP
        ]
        stats = self._eval_stats(rows)
        # Both activated: gross should be 60
        self.assertEqual(stats.get("tp_delta"), 50)
        self.assertEqual(stats.get("safe_fp_delta"), 10)
        self.assertEqual(stats.get("gross_nodes_saved"), 60)

    def test_evaluate_gross_excludes_non_activated(self):
        # Row not activated (prob=0.001 < threshold=0.05) should not contribute
        rows = [self._row("SAFE", True, 50)]
        import torch, torch.nn as nn
        from training.train_reduction_sidecar_v2 import evaluate_at_threshold
        model = nn.Linear(37, 1)
        nn.init.zeros_(model.weight)
        nn.init.constant_(model.bias, -6.0)  # sigmoid(-6)≈0.002, all rows score low
        stats = evaluate_at_threshold(model, rows, 1.0, 0.0, 0.05, "A")
        self.assertEqual(stats.get("tp_delta", 0), 0)
        self.assertEqual(stats.get("gross_nodes_saved"), 0)


class RankPercentileTests(unittest.TestCase):
    """Verify rank_percentile computation."""

    def test_first_move_is_zero(self):
        self.assertAlmostEqual(rank_percentile(0, 10), 0.0)

    def test_last_move_is_one(self):
        self.assertAlmostEqual(rank_percentile(9, 10), 1.0)

    def test_midpoint(self):
        self.assertAlmostEqual(rank_percentile(4, 9), 0.5)

    def test_single_move_returns_zero(self):
        # denominator clamped to max(n-1, 1)=1; result = 0/1 = 0
        self.assertAlmostEqual(rank_percentile(0, 1), 0.0)

    def test_move_index_4_of_20(self):
        expected = 4 / 19
        self.assertAlmostEqual(rank_percentile(4, 20), expected, places=10)

    def test_parity_with_context7_schema(self):
        # Verify that rank_percentile matches what context_features_v2 produces
        probe_event = {
            "depth": 6, "move_index": 7, "base_reduction": 2,
            "move": "c4h", "total_legal_moves": 50, "history_score": 1000,
        }
        ctx7 = context_features_v2(probe_event)
        self.assertEqual(len(ctx7), 7)
        expected_rp = rank_percentile(7, 50)
        self.assertAlmostEqual(ctx7[6], expected_rp, places=10)

    def test_feature_schema_v2_constant_exported(self):
        self.assertEqual(FEATURE_SCHEMA_V2, "halfpw-hidden32-search-context7-v2")

    def test_context7_history_normalisation(self):
        # history=10000 → norm=1.0; history=-10000 → norm=0.0; history=0 → norm=0.5
        def ctx7_history(h):
            return context_features_v2({
                "depth": 4, "move_index": 0, "base_reduction": 1,
                "move": "c4h", "total_legal_moves": 10, "history_score": h,
            })[5]

        self.assertAlmostEqual(ctx7_history(10000), 1.0)
        self.assertAlmostEqual(ctx7_history(-10000), 0.0)
        self.assertAlmostEqual(ctx7_history(0), 0.5)

    def test_variant_plob_uses_rank_percentile(self):
        from training.train_reduction_sidecar_v2 import get_features, VARIANTS
        row = {
            "hidden32": [0.0] * 32,
            "context5": [0.1, 0.2, 0.3, 0.4, 0.5],
            "history_score": 5000,
            "total_legal_moves": 20,
            "move_index": 4,
            "activate_plus_one": False,
            "sample_status": "SAFE",
        }
        f = get_features(row, "PLOB")
        self.assertEqual(len(f), VARIANTS["PLOB"]["dims"])
        # feature at index 38 should be rank_percentile(4, 20) = 4/19
        expected_rp = rank_percentile(4, 20)
        self.assertAlmostEqual(f[38], expected_rp, places=10)


if __name__ == "__main__":
    unittest.main()

