"""
SirenGuard: High-level orchestration pipeline for SIREN framework.
Combines extraction, probing, adaptive neuron aggregation, MLP classification, and streaming moderation.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Sequence
import os
import torch
import numpy as np

from siren.extractor import InternalStateExtractor
from siren.probe import SafetyNeuronProbe
from siren.aggregator import AdaptiveNeuronAggregator
from siren.classifier import SirenMLPHead, SirenTrainer
from siren.streaming import StreamingModerator


class SirenGuard:
    """
    Complete end-to-end SIREN Safeguard system.
    """

    def __init__(
        self,
        extractor: Optional[InternalStateExtractor] = None,
        eta: float = 0.8,
        c_val: float = 0.1,
        c_grid: Optional[Sequence[float]] = None,
        average: str = "macro",
        mlp_hidden_dim: int = 512,
        threshold: float = 0.5,
        device: Optional[str] = None
    ):
        self.extractor = extractor
        self.eta = eta
        self.c_val = c_val
        self.c_grid = list(c_grid) if c_grid is not None else None
        self.average = average
        self.mlp_hidden_dim = mlp_hidden_dim
        self.threshold = threshold

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.probe: Optional[SafetyNeuronProbe] = None
        self.aggregator: Optional[AdaptiveNeuronAggregator] = None
        self.classifier: Optional[SirenMLPHead] = None
        self.trainer: Optional[SirenTrainer] = None

    def fit_from_activations(
        self,
        layer_activations_train: Dict[int, Union[torch.Tensor, np.ndarray]],
        y_train: Union[torch.Tensor, np.ndarray],
        layer_activations_val: Optional[Dict[int, Union[torch.Tensor, np.ndarray]]] = None,
        y_val: Optional[Union[torch.Tensor, np.ndarray]] = None,
        epochs: int = 15,
        batch_size: int = 32,
        lr: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Fit SIREN on pre-extracted/pooled layer activations.
        
        Args:
            layer_activations_train: Dict mapping layer_idx to activations (N_train, D)
            y_train: Labels tensor/array of shape (N_train,)
            layer_activations_val: Dict mapping layer_idx to activations (N_val, D)
            y_val: Labels tensor/array of shape (N_val,)
            epochs: MLP training epochs
            batch_size: MLP batch size
            lr: Learning rate
            
        Returns:
            history: Training metrics history
        """
        if isinstance(y_train, np.ndarray):
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        else:
            y_train_tensor = y_train.long()

        if y_val is not None:
            if isinstance(y_val, np.ndarray):
                y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            else:
                y_val_tensor = y_val.long()
        else:
            y_val_tensor = None

        # Step 1: Safety Neuron Localization via L1 linear probing
        self.probe = SafetyNeuronProbe(
            eta=self.eta, c_val=self.c_val, c_grid=self.c_grid, average=self.average
        )
        safety_neurons, layer_f1_scores = self.probe.fit_all_layers(
            layer_activations_train=layer_activations_train,
            y_train=y_train,
            layer_activations_val=layer_activations_val,
            y_val=y_val
        )

        # Step 2: Adaptive Neuron Aggregation
        self.aggregator = AdaptiveNeuronAggregator(
            safety_neurons=safety_neurons,
            layer_f1_scores=layer_f1_scores
        )

        z_train = self.aggregator.transform(layer_activations_train)
        z_val = self.aggregator.transform(layer_activations_val) if layer_activations_val else None

        # Step 3: Train Lightweight MLP Head
        input_dim = z_train.size(1)
        self.classifier = SirenMLPHead(input_dim=input_dim, hidden_dim=self.mlp_hidden_dim)
        self.trainer = SirenTrainer(model=self.classifier, lr=lr, device=self.device)

        history = self.trainer.fit(
            z_train=z_train,
            y_train=y_train_tensor,
            z_val=z_val,
            y_val=y_val_tensor,
            epochs=epochs,
            batch_size=batch_size
        )

        return {
            "history": history,
            "safety_neurons_count": self.aggregator.total_feature_dim,
            "layer_weights": self.aggregator.layer_weights,
            "layer_f1_scores": layer_f1_scores
        }

    def predict_from_activations(
        self,
        layer_activations: Dict[int, Union[torch.Tensor, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict harmfulness from layer activations.
        
        Returns:
            predictions: Array of shape (N,) with binary labels (0 or 1)
            probabilities: Array of shape (N,) with harmfulness probability
        """
        if self.aggregator is None or self.classifier is None:
            raise RuntimeError("SIREN guard must be fitted before calling predict.")

        self.classifier.eval()
        z = self.aggregator.transform(layer_activations).to(self.device)
        with torch.no_grad():
            probs = self.classifier.predict_proba(z)[:, 1].cpu().numpy()
            preds = (probs >= self.threshold).astype(int)

        return preds, probs

    def get_streaming_moderator(self) -> StreamingModerator:
        """Instantiate StreamingModerator for token-level moderation."""
        if self.aggregator is None or self.classifier is None:
            raise RuntimeError("SIREN guard must be fitted before creating StreamingModerator.")

        return StreamingModerator(
            extractor=self.extractor,
            aggregator=self.aggregator,
            classifier=self.classifier,
            threshold=self.threshold,
            device=self.device
        )

    def save(self, dir_path: str):
        """Save SIREN state to directory."""
        os.makedirs(dir_path, exist_ok=True)
        if self.aggregator is None or self.classifier is None:
            raise RuntimeError("Cannot save unfitted SIREN model.")

        state = {
            "eta": self.eta,
            "c_val": self.c_val,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "threshold": self.threshold,
            "safety_neurons": self.aggregator.safety_neurons,
            "layer_f1_scores": self.aggregator.layer_f1_scores,
            "classifier_state_dict": self.classifier.state_dict(),
            "input_dim": self.aggregator.total_feature_dim
        }
        torch.save(state, os.path.join(dir_path, "siren_state.pt"))

    @classmethod
    def load(cls, dir_path: str, extractor: Optional[InternalStateExtractor] = None, device: Optional[str] = None) -> "SirenGuard":
        """Load SIREN state from directory."""
        state_path = os.path.join(dir_path, "siren_state.pt")
        state = torch.load(state_path, map_location="cpu")

        guard = cls(
            extractor=extractor,
            eta=state["eta"],
            c_val=state["c_val"],
            mlp_hidden_dim=state["mlp_hidden_dim"],
            threshold=state["threshold"],
            device=device
        )

        guard.aggregator = AdaptiveNeuronAggregator(
            safety_neurons=state["safety_neurons"],
            layer_f1_scores=state["layer_f1_scores"]
        )

        guard.classifier = SirenMLPHead(
            input_dim=state["input_dim"],
            hidden_dim=state["mlp_hidden_dim"]
        )
        guard.classifier.load_state_dict(state["classifier_state_dict"])
        guard.trainer = SirenTrainer(model=guard.classifier, device=guard.device)

        return guard
