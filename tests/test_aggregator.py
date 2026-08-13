"""
Unit tests for siren/aggregator.py
"""

import pytest
import torch
import numpy as np
from siren.aggregator import AdaptiveNeuronAggregator


def test_adaptive_neuron_aggregator():
    safety_neurons = {1: [0, 1, 2], 2: [5, 6]}
    f1_scores = {1: 0.9, 2: 0.7}

    aggregator = AdaptiveNeuronAggregator(safety_neurons, f1_scores)
    assert aggregator.total_feature_dim == 5
    assert aggregator.layer_weights[1] == 1.0
    assert aggregator.layer_weights[2] == 0.0

    dummy_activations = {
        1: torch.ones(10, 30),
        2: torch.ones(10, 30) * 2.0
    }

    z = aggregator.transform(dummy_activations)
    assert z.shape == (10, 5)
    # Layer 1 has alpha=1.0 -> 1.0
    assert torch.allclose(z[:, :3], torch.ones(10, 3))
    # Layer 2 has alpha=0.0 -> 0.0
    assert torch.allclose(z[:, 3:], torch.zeros(10, 2))
