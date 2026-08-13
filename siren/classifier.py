"""
Lightweight Multi-Layer Perceptron (MLP) classifier head and trainer for SIREN.
Processes concatenated cross-layer safety neuron features z for harmfulness prediction.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score


class SirenMLPHead(nn.Module):
    """
    Lightweight MLP Classifier Head (~5M-56M parameters or small custom size).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Input feature tensor of shape (B, input_dim)
            
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        return self.net(z)

    def predict_proba(self, z: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities via Softmax."""
        logits = self.forward(z)
        return torch.softmax(logits, dim=-1)


class SirenTrainer:
    """
    Trainer for SirenMLPHead.
    """

    def __init__(
        self,
        model: SirenMLPHead,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        self.model = model
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train model for 1 epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_z, batch_y in dataloader:
            batch_z = batch_z.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_z)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_z.size(0)

        return total_loss / len(dataloader.dataset)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset and return metrics."""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        total_loss = 0.0

        for batch_z, batch_y in dataloader:
            batch_z = batch_z.to(self.device)
            batch_y = batch_y.to(self.device)

            logits = self.model(batch_z)
            loss = self.criterion(logits, batch_y)
            total_loss += loss.item() * batch_z.size(0)

            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)

        metrics = {
            "loss": total_loss / len(dataloader.dataset),
            "accuracy": float(accuracy_score(all_targets, all_preds)),
            "precision": float(precision_score(all_targets, all_preds, zero_division=0)),
            "recall": float(recall_score(all_targets, all_preds, zero_division=0)),
            "f1": float(f1_score(all_targets, all_preds, zero_division=0))
        }

        try:
            metrics["auroc"] = float(roc_auc_score(all_targets, all_probs))
        except ValueError:
            metrics["auroc"] = 0.5

        return metrics

    def fit(
        self,
        z_train: torch.Tensor,
        y_train: torch.Tensor,
        z_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        epochs: int = 15,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Fit SirenMLPHead model for specified epochs."""
        train_dataset = TensorDataset(z_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if z_val is not None and y_val is not None:
            val_dataset = TensorDataset(z_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_metrics": []}

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            history["train_loss"].append(loss)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_metrics"].append(val_metrics)

        return history
