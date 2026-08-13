"""
Unit tests for siren/classifier.py
"""

import pytest
import torch
from siren.classifier import SirenMLPHead, SirenTrainer


def test_siren_mlp_head_forward():
    model = SirenMLPHead(input_dim=20, hidden_dim=64)
    dummy_z = torch.randn(8, 20)
    logits = model(dummy_z)
    assert logits.shape == (8, 2)

    probs = model.predict_proba(dummy_z)
    assert probs.shape == (8, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8))


def test_siren_trainer_fit():
    model = SirenMLPHead(input_dim=10, hidden_dim=32)
    trainer = SirenTrainer(model=model, lr=1e-3, device="cpu")

    z_train = torch.randn(50, 10)
    y_train = torch.randint(0, 2, (50,))

    z_val = torch.randn(20, 10)
    y_val = torch.randint(0, 2, (20,))

    history = trainer.fit(z_train, y_train, z_val, y_val, epochs=3, batch_size=16)
    assert len(history["train_loss"]) == 3
    assert len(history["val_metrics"]) == 3
    assert "f1" in history["val_metrics"][0]
