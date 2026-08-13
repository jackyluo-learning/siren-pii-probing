"""
Unit tests for siren/extractor.py
"""

import pytest
import torch
import torch.nn as nn
from siren.extractor import InternalStateExtractor


class DummyDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)


class DummyLLM(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(100, hidden_dim)
        self.layers = nn.ModuleList([DummyDecoderLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h


def test_internal_state_extractor_dummy():
    model = DummyLLM(hidden_dim=32, num_layers=4)
    extractor = InternalStateExtractor(model=model, device="cpu")

    input_ids = torch.randint(0, 100, (2, 10))
    attention_mask = torch.ones(2, 10)

    pooled_states = extractor.extract_sequence_pooled(input_ids, attention_mask)

    assert len(pooled_states) == 4
    for l_idx in range(1, 5):
        assert l_idx in pooled_states
        assert pooled_states[l_idx].shape == (2, 32)

    extractor.remove_hooks()
