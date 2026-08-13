"""
Unit tests for siren/probe.py
"""

import pytest
import torch
import numpy as np
from siren.probe import SafetyNeuronProbe
from siren.dataset import SyntheticSafetyDataset


def test_safety_neuron_probe_fitting():
    synth = SyntheticSafetyDataset(num_samples=100, num_layers=4, hidden_dim=64, safety_neuron_count=8, seed=42)
    layer_acts, labels = synth.generate_synthetic_activations()

    probe = SafetyNeuronProbe(eta=0.8, c_val=0.1)
    neurons, f1_scores = probe.fit_all_layers(layer_acts, labels)

    assert len(neurons) == 4
    for l_idx in range(1, 5):
        assert l_idx in neurons
        assert len(neurons[l_idx]) > 0
        assert len(neurons[l_idx]) <= 64
        assert f1_scores[l_idx] >= 0.0 and f1_scores[l_idx] <= 1.0
