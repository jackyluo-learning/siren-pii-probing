"""
Unit tests for siren/streaming.py
"""

import pytest
import torch

from siren.dataset import SyntheticSafetyDataset
from siren.pipeline import SirenGuard
from siren.streaming import StreamingModerator


def test_streaming_moderator_evaluation():
    synth = SyntheticSafetyDataset(num_samples=100, num_layers=4, hidden_dim=64, safety_neuron_count=8, seed=42)
    layer_acts, labels = synth.generate_synthetic_activations()

    guard = SirenGuard(eta=0.8, c_val=0.1, mlp_hidden_dim=64, device="cpu")
    guard.fit_from_activations(layer_acts, labels, epochs=5)

    streaming_mod = guard.get_streaming_moderator()

    # Test single prefix evaluation
    single_prefix_acts = {l: layer_acts[l][:1] for l in range(1, 5)}
    score, pred = streaming_mod.evaluate_prefix_features(single_prefix_acts)

    assert 0.0 <= score <= 1.0
    assert pred in (0, 1)
