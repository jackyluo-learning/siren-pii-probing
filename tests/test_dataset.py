"""
Unit tests for siren/dataset.py
"""

import pytest
import torch
import numpy as np
from siren.dataset import SafetySample, SafetyDataset, SyntheticSafetyDataset


def test_safety_sample_creation():
    sample = SafetySample(text="test text", label=1)
    assert sample.text == "test text"
    assert sample.label == 1


def test_synthetic_safety_dataset_generation():
    synth = SyntheticSafetyDataset(num_samples=20, num_layers=4, hidden_dim=64, safety_neuron_count=4)
    ds = synth.generate_samples()
    assert len(ds) == 20
    assert ds[0].label in (0, 1)

    layer_acts, labels = synth.generate_synthetic_activations()
    assert len(layer_acts) == 4
    assert 1 in layer_acts
    assert layer_acts[1].shape == (20, 64)
    assert labels.shape == (20,)
