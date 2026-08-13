"""
Safety neuron localization using layer-wise L1-regularized linear probing.
Filters out noise and identifies the sparse set of safety-relevant internal neurons.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


class SafetyNeuronProbe:
    """
    Fits layer-wise L1-regularized linear probes to identify safety neurons.
    """

    def __init__(
        self,
        eta: float = 0.8,
        c_val: float = 0.1,
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Args:
            eta: Cumulative weight threshold for selecting top safety neurons (0 < eta <= 1.0).
            c_val: Inverse of L1 regularization strength in LogisticRegression (smaller C = stronger L1).
            max_iter: Maximum iterations for LogisticRegression solver.
            random_state: Random seed.
        """
        self.eta = eta
        self.c_val = c_val
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.layer_probes: Dict[int, LogisticRegression] = {}
        self.safety_neurons: Dict[int, List[int]] = {}
        self.layer_f1_scores: Dict[int, float] = {}

    def fit_layer(
        self,
        layer_idx: int,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[List[int], float]:
        """
        Train L1 linear probe for a single layer and extract its safety neurons.
        
        Returns:
            selected_neuron_indices: List of selected safety neuron indices
            val_f1: Validation (or training) F1 score for layer probe
        """
        import warnings
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=self.c_val,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            clf.fit(X_train, y_train)

        # Extract weights (1, D)
        weights = np.abs(clf.coef_[0])
        total_weight = np.sum(weights)

        if total_weight == 0 or np.isnan(total_weight):
            # Fallback if L1 pruned all weights: pick top 5% by raw magnitude
            top_k = max(1, int(len(weights) * 0.05))
            selected_indices = list(np.argsort(weights)[::-1][:top_k])
        else:
            normalized_weights = weights / total_weight
            sorted_indices = np.argsort(normalized_weights)[::-1]
            
            cumulative_sum = 0.0
            selected_indices = []
            for idx in sorted_indices:
                selected_indices.append(int(idx))
                cumulative_sum += normalized_weights[idx]
                if cumulative_sum >= self.eta:
                    break

        # Compute F1 score on validation set (or training set fallback)
        if X_val is not None and y_val is not None:
            if isinstance(X_val, torch.Tensor):
                X_val = X_val.detach().cpu().numpy()
            if isinstance(y_val, torch.Tensor):
                y_val = y_val.detach().cpu().numpy()
            preds = clf.predict(X_val)
            val_f1 = float(f1_score(y_val, preds, zero_division=0))
        else:
            preds = clf.predict(X_train)
            val_f1 = float(f1_score(y_train, preds, zero_division=0))

        self.layer_probes[layer_idx] = clf
        self.safety_neurons[layer_idx] = selected_indices
        self.layer_f1_scores[layer_idx] = val_f1

        return selected_indices, val_f1

    def fit_all_layers(
        self,
        layer_activations_train: Dict[int, Union[np.ndarray, torch.Tensor]],
        y_train: Union[np.ndarray, torch.Tensor],
        layer_activations_val: Optional[Dict[int, Union[np.ndarray, torch.Tensor]]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
        """Fit L1 linear probes across all layers."""
        self.safety_neurons.clear()
        self.layer_f1_scores.clear()

        for layer_idx in sorted(layer_activations_train.keys()):
            X_tr = layer_activations_train[layer_idx]
            X_v = layer_activations_val[layer_idx] if layer_activations_val else None
            
            self.fit_layer(layer_idx, X_tr, y_train, X_v, y_val)

        return self.safety_neurons, self.layer_f1_scores
