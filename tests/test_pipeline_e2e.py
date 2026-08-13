"""
End-to-end integration test for SirenGuard pipeline.
"""

import pytest
import os
import shutil
import numpy as np
import torch
from siren.dataset import SyntheticSafetyDataset
from siren.pipeline import SirenGuard


def test_siren_guard_end_to_end(tmp_path):
    synth = SyntheticSafetyDataset(num_samples=100, num_layers=4, hidden_dim=64, safety_neuron_count=8, seed=42)
    layer_acts, labels = synth.generate_synthetic_activations()

    guard = SirenGuard(eta=0.8, c_val=0.1, mlp_hidden_dim=64, device="cpu")
    fit_info = guard.fit_from_activations(layer_acts, labels, epochs=5, batch_size=16)

    assert fit_info["safety_neurons_count"] > 0
    assert len(fit_info["layer_weights"]) == 4

    preds, probs = guard.predict_from_activations(layer_acts)
    assert len(preds) == 100
    assert len(probs) == 100
    assert set(np.unique(preds)).issubset({0, 1})

    # Save and Load
    save_dir = str(tmp_path / "test_siren_save")
    guard.save(save_dir)
    assert os.path.exists(os.path.join(save_dir, "siren_state.pt"))

    loaded_guard = SirenGuard.load(save_dir, device="cpu")
    loaded_preds, loaded_probs = loaded_guard.predict_from_activations(layer_acts)

    assert np.array_equal(preds, loaded_preds)
    assert np.allclose(probs, loaded_probs, atol=1e-5)
