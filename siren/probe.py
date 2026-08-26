"""
Safety neuron localization using layer-wise L1-regularized linear probing.
Filters out noise and identifies the sparse set of safety-relevant internal neurons.

Paper alignment (arXiv:2604.18519, Appendix A.1 / Table 6):
  - L1-regularized logistic regression per layer.
  - Regularization strength C selected by grid search over {100, 200, 500, 1000},
    choosing the value that maximizes *validation* Macro-F1 (never the test set).
  - Safety neurons = minimal set of neurons whose cumulative normalized |weight|
    exceeds threshold eta.
  - Reported score is Macro-F1 to account for class imbalance.
"""

from typing import Dict, List, Tuple, Optional, Union, Sequence
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier


# Paper Table 6 grid for the L1 inverse-regularization strength C.
PAPER_C_GRID: Tuple[float, ...] = (100.0, 200.0, 500.0, 1000.0)


class SafetyNeuronProbe:
    """
    Fits layer-wise L1-regularized linear probes to identify safety neurons.
    """

    def __init__(
        self,
        eta: float = 0.8,
        c_val: float = 0.1,
        c_grid: Optional[Sequence[float]] = None,
        average: str = "macro",
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Args:
            eta: Cumulative weight threshold for selecting top safety neurons (0 < eta <= 1.0).
            c_val: Inverse L1 regularization strength used when no grid search is run
                (i.e. c_grid is None). Smaller C = stronger L1 sparsity.
            c_grid: Candidate C values for grid search. When provided AND a validation
                set is passed to fit, the C maximizing validation `average`-F1 is chosen
                per layer (paper spec: {100, 200, 500, 1000}). When None, the fixed
                `c_val` is used (backwards-compatible behaviour).
            average: F1 averaging mode for probe selection/reporting. "macro" matches the
                paper; "binary" reproduces the legacy positive-class F1.
            max_iter: Maximum iterations for the LogisticRegression solver.
            random_state: Random seed.
        """
        self.eta = eta
        self.c_val = c_val
        self.c_grid = list(c_grid) if c_grid is not None else None
        self.average = average
        self.max_iter = max_iter
        self.random_state = random_state

        self.layer_probes: Dict[int, LogisticRegression] = {}
        self.safety_neurons: Dict[int, List[int]] = {}
        self.layer_f1_scores: Dict[int, float] = {}
        self.layer_best_c: Dict[int, float] = {}

    # ------------------------------------------------------------------ helpers
    def _fit_single(self, C: float, X: np.ndarray, y: np.ndarray):
        """
        Fit one L1 logistic-regression probe at inverse-strength C.

        The liblinear solver the paper uses refuses n_classes >= 3, so multiclass
        targets are wrapped in a one-vs-rest ensemble: every class still gets an
        L1 + liblinear probe, which keeps the per-class setup identical to the
        binary case (switching to `saga` would change the optimiser as well as
        the multiclass scheme, and is far slower). The stacked per-class weights
        are exposed as `coef_` so neuron selection is agnostic to which path ran.
        """
        import warnings
        base = lambda: LogisticRegression(
            penalty="l1", solver="liblinear", C=C,
            max_iter=self.max_iter, random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(np.unique(y)) > 2:
                clf = OneVsRestClassifier(base())
                clf.fit(X, y)
                clf.coef_ = np.vstack([e.coef_[0] for e in clf.estimators_])
            else:
                clf = base()
                clf.fit(X, y)
        return clf

    def _score(self, clf: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
        """Macro (or configured) F1 of a fitted probe on (X, y). Multiclass-safe."""
        preds = clf.predict(X)
        return float(f1_score(y, preds, average=self.average, zero_division=0))

    def _select_neurons(self, clf: LogisticRegression) -> List[int]:
        """
        Pick the minimal neuron set whose cumulative normalized |weight| >= eta.

        Multiclass note: sklearn stores coef_ as (n_classes, D) for K>2 classes
        (one-vs-rest with the liblinear solver) and (1, D) for binary. A neuron is
        relevant if ANY class leans on it, so the per-neuron score is the maximum
        magnitude across classes -- taking the mean would let one strongly-used
        neuron be diluted by the classes that ignore it.
        """
        coef = np.asarray(clf.coef_)
        weights = np.abs(coef).max(axis=0) if coef.shape[0] > 1 else np.abs(coef[0])
        total_weight = np.sum(weights)

        if total_weight == 0 or np.isnan(total_weight):
            # L1 pruned everything: fall back to top 5% by raw magnitude.
            top_k = max(1, int(len(weights) * 0.05))
            return list(np.argsort(weights)[::-1][:top_k])

        normalized_weights = weights / total_weight
        sorted_indices = np.argsort(normalized_weights)[::-1]

        cumulative_sum = 0.0
        selected_indices: List[int] = []
        for idx in sorted_indices:
            selected_indices.append(int(idx))
            cumulative_sum += normalized_weights[idx]
            if cumulative_sum >= self.eta:
                break
        return selected_indices

    def fit_layer(
        self,
        layer_idx: int,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[List[int], float]:
        """
        Train an L1 linear probe for a single layer and extract its safety neurons.

        If ``c_grid`` was configured and a validation set is supplied, the C value
        maximizing validation Macro-F1 is selected (paper spec). Otherwise the fixed
        ``c_val`` is used. Neurons are always taken from the finally selected probe.

        Returns:
            selected_neuron_indices: List of selected safety neuron indices
            score: Validation (or training) Macro-F1 of the selected layer probe
        """
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()

        has_val = X_val is not None and y_val is not None
        if has_val:
            if isinstance(X_val, torch.Tensor):
                X_val = X_val.detach().cpu().numpy()
            if isinstance(y_val, torch.Tensor):
                y_val = y_val.detach().cpu().numpy()

        # Decide the candidate C values. Grid search only meaningfully selects a
        # value when there is a held-out set to score on; without one we would be
        # tuning on the fit data itself, so fall back to the fixed c_val.
        if self.c_grid and has_val:
            candidates = self.c_grid
        else:
            candidates = [self.c_val]

        best_clf: Optional[LogisticRegression] = None
        best_c: float = candidates[0]
        best_score: float = -1.0

        for C in candidates:
            clf = self._fit_single(C, X_train, y_train)
            # Model selection is done on validation data only (never on test).
            eval_X, eval_y = (X_val, y_val) if has_val else (X_train, y_train)
            score = self._score(clf, eval_X, eval_y)
            if score > best_score:
                best_score, best_clf, best_c = score, clf, C

        selected_indices = self._select_neurons(best_clf)

        self.layer_probes[layer_idx] = best_clf
        self.safety_neurons[layer_idx] = selected_indices
        self.layer_f1_scores[layer_idx] = best_score
        self.layer_best_c[layer_idx] = best_c

        return selected_indices, best_score

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
        self.layer_best_c.clear()

        for layer_idx in sorted(layer_activations_train.keys()):
            X_tr = layer_activations_train[layer_idx]
            X_v = layer_activations_val[layer_idx] if layer_activations_val else None
            
            self.fit_layer(layer_idx, X_tr, y_train, X_v, y_val)

        return self.safety_neurons, self.layer_f1_scores
