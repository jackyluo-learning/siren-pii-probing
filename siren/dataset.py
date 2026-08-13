"""
Dataset handling for SIREN safety detection.
Provides loaders for safety benchmarks and a synthetic dataset generator for local testing.
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np


@dataclass
class SafetySample:
    text: str
    label: int  # 0 = Safe, 1 = Harmful
    metadata: Optional[Dict[str, Any]] = None


class SafetyDataset(Dataset):
    """
    Dataset wrapper for safety classification.
    """
    def __init__(self, samples: List[SafetySample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SafetySample:
        return self.samples[idx]

    @classmethod
    def from_jsonl(cls, filepath: str, text_key: str = "text", label_key: str = "label") -> "SafetyDataset":
        """Load dataset from a JSONL file."""
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                samples.append(
                    SafetySample(
                        text=data[text_key],
                        label=int(data[label_key]),
                        metadata={k: v for k, v in data.items() if k not in (text_key, label_key)}
                    )
                )
        return cls(samples)


class SyntheticSafetyDataset:
    """
    Generates synthetic dataset and synthetic layer hidden representations.
    Useful for unit testing, fast CPU verification, and offline execution without downloading LLM weights.
    """
    def __init__(
        self,
        num_samples: int = 200,
        num_layers: int = 12,
        hidden_dim: int = 256,
        safety_neuron_count: int = 16,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.safety_neuron_count = safety_neuron_count
        self.seed = seed

    def generate_samples(self) -> SafetyDataset:
        """Generate text samples (half benign, half harmful)."""
        rng = np.random.RandomState(self.seed)
        samples = []
        
        harmful_keywords = ["bomb", "hack", "exploit", "poison", "illegal", "attack", "steal", "malware"]
        benign_keywords = ["hello", "science", "recipe", "python", "history", "math", "music", "art"]

        for i in range(self.num_samples):
            label = i % 2
            if label == 1:
                kw = rng.choice(harmful_keywords)
                text = f"User asks how to create a {kw} in detail for illegal operations."
            else:
                kw = rng.choice(benign_keywords)
                text = f"User asks how to learn {kw} for educational purposes."
            
            samples.append(SafetySample(text=text, label=label, metadata={"id": i}))
            
        return SafetyDataset(samples)

    def generate_synthetic_activations(self) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Generate synthetic layer-wise pooled hidden states (N, D) and labels (N,).
        Injects strong safety signals into a subset of designated 'safety neurons' across layers.
        
        Returns:
            layer_activations: Dict mapping layer_idx (1..L) to Tensor of shape (num_samples, hidden_dim)
            labels: Tensor of shape (num_samples,)
        """
        rng = np.random.RandomState(self.seed)
        labels_np = np.array([i % 2 for i in range(self.num_samples)], dtype=np.int64)
        
        layer_activations = {}
        
        # Designate specific safety neuron indices per layer
        for layer_idx in range(1, self.num_layers + 1):
            # Base Gaussian noise for all neurons
            acts = rng.normal(loc=0.0, scale=1.0, size=(self.num_samples, self.hidden_dim))
            
            # Middle-to-late layers have stronger safety signal (mimicking LLM internal behavior)
            signal_strength = float(np.sin(np.pi * (layer_idx / self.num_layers))) * 3.0
            
            # Select specific neurons to bear safety signals
            safety_indices = np.arange(self.safety_neuron_count) + (layer_idx * 3) % (self.hidden_dim - self.safety_neuron_count)
            
            for n_idx in safety_indices:
                # Add positive shift for harmful class (label=1), negative for safe (label=0)
                acts[:, n_idx] += (labels_np * 2 - 1) * signal_strength
                
            layer_activations[layer_idx] = torch.tensor(acts, dtype=torch.float32)
            
        labels_tensor = torch.tensor(labels_np, dtype=torch.long)
        return layer_activations, labels_tensor
