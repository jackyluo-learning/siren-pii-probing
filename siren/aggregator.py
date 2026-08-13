"""
Adaptive cross-layer neuron aggregator.
Computes performance-weighted layer importance weights alpha_l
and concatenates cross-layer safety neurons to produce feature vector z.
"""

from typing import Dict, List, Union
import torch
import numpy as np


class AdaptiveNeuronAggregator:
    """
    Aggregates safety neurons across transformer layers weighted by layer validation performance.
    """

    def __init__(
        self,
        safety_neurons: Dict[int, List[int]],
        layer_f1_scores: Dict[int, float]
    ):
        """
        Args:
            safety_neurons: Dict mapping layer_idx to list of selected neuron indices S_l
            layer_f1_scores: Dict mapping layer_idx to validation F1 score f_l
        """
        self.safety_neurons = safety_neurons
        self.layer_f1_scores = layer_f1_scores
        self.layer_weights = self._compute_layer_weights()

    def _compute_layer_weights(self) -> Dict[int, float]:
        """
        Compute alpha_l = (f_l - f_min) / (f_max - f_min)
        Handles edge cases where f_max == f_min.
        """
        f_scores = list(self.layer_f1_scores.values())
        if not f_scores:
            return {}

        f_min = min(f_scores)
        f_max = max(f_scores)
        denom = f_max - f_min

        layer_weights = {}
        for layer_idx, f_l in self.layer_f1_scores.items():
            if denom < 1e-6:
                layer_weights[layer_idx] = 1.0
            else:
                layer_weights[layer_idx] = (f_l - f_min) / denom

        return layer_weights

    def transform(
        self,
        layer_activations: Dict[int, Union[torch.Tensor, np.ndarray]]
    ) -> torch.Tensor:
        """
        Transform layer activations into concatenated feature vector z.
        
        Args:
            layer_activations: Dict mapping layer_idx to Tensor/ndarray of shape (B, D)
            
        Returns:
            z: Concatenated Tensor of shape (B, total_safety_neurons)
        """
        subvectors = []
        sorted_layers = sorted(self.safety_neurons.keys())

        for layer_idx in sorted_layers:
            neurons = self.safety_neurons[layer_idx]
            if not neurons:
                continue

            acts = layer_activations[layer_idx]
            if isinstance(acts, np.ndarray):
                acts = torch.tensor(acts, dtype=torch.float32)
            else:
                acts = acts.float()

            alpha_l = self.layer_weights.get(layer_idx, 1.0)
            
            # Extract safety neuron subvector and weight by alpha_l
            layer_subvector = acts[:, neurons] * alpha_l
            subvectors.append(layer_subvector)

        if not subvectors:
            raise ValueError("No safety neurons selected across any layers.")

        z = torch.cat(subvectors, dim=1)
        return z

    @property
    def total_feature_dim(self) -> int:
        """Get total dimension of aggregated vector z."""
        return sum(len(indices) for indices in self.safety_neurons.values())
