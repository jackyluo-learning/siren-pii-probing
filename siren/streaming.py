"""
Streaming moderation engine for SIREN.
Enables real-time, token-by-token harmfulness detection with zero-shot prefix pooling
and millisecond-level early stopping.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
import torch
from siren.extractor import InternalStateExtractor
from siren.aggregator import AdaptiveNeuronAggregator
from siren.classifier import SirenMLPHead


class StreamingModerator:
    """
    Evaluates harmfulness score at each token step during LLM generation.
    """

    def __init__(
        self,
        extractor: Optional[InternalStateExtractor],
        aggregator: AdaptiveNeuronAggregator,
        classifier: SirenMLPHead,
        threshold: float = 0.5,
        device: Optional[str] = None
    ):
        self.extractor = extractor
        self.aggregator = aggregator
        self.classifier = classifier
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

        self.classifier.to(self.device)
        self.classifier.eval()

    @torch.no_grad()
    def evaluate_prefix_features(
        self,
        prefix_activations: Dict[int, torch.Tensor]
    ) -> Tuple[float, int]:
        """
        Evaluate harmfulness score given pre-extracted/pooled prefix activations.
        
        Args:
            prefix_activations: Dict mapping layer_idx to Tensor of shape (1, D)
            
        Returns:
            harmful_score: Probablity of harmfulness in [0.0, 1.0]
            prediction: 1 if harmful_score >= threshold else 0
        """
        z_t = self.aggregator.transform(prefix_activations).to(self.device)
        probs = self.classifier.predict_proba(z_t)
        harmful_prob = float(probs[0, 1].item())
        pred = 1 if harmful_prob >= self.threshold else 0
        return harmful_prob, pred

    @torch.no_grad()
    def score_token_stream(
        self,
        input_ids: torch.Tensor
    ) -> "np.ndarray":
        """
        Harmfulness score at EVERY token position (paper Eq. 7-9), via a single
        forward pass over the full sequence.

        The classifier trained on full-sequence features is applied unchanged to
        prefix-restricted features -- no parameters are updated, matching the
        paper's strict zero-shot streaming protocol.

        Args:
            input_ids: Tensor of shape (1, T).

        Returns:
            scores: np.ndarray of shape (T,) with P(harmful) after each token.
        """
        if self.extractor is None:
            raise ValueError("An extractor is required to score a token stream.")

        # (T, D) per layer -- row t-1 is the mean over prefix s<=t.
        prefix_states = self.extractor.extract_all_prefix_pooled(input_ids)
        # Aggregator treats the leading axis as batch, so all T prefixes are
        # aggregated and classified in one shot.
        z = self.aggregator.transform(prefix_states).to(self.device)
        probs = self.classifier.predict_proba(z)[:, 1]
        return probs.detach().cpu().numpy()

    @torch.no_grad()
    def moderate_stream(
        self,
        input_ids: torch.Tensor,
        start_token_pos: int = 1,
        early_stop: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate harmfulness score token-by-token over input_ids sequence.
        
        Args:
            input_ids: Input token tensor of shape (1, T)
            start_token_pos: Token index to begin monitoring (default 1)
            early_stop: If True, halts evaluation on first token crossing threshold.
            
        Returns:
            results: Dict containing token_scores, flagged_step, early_stopped, final_score
        """
        if self.extractor is None:
            raise ValueError("Extractor must be provided to run moderate_stream with input_ids.")

        seq_len = input_ids.size(1)
        token_scores = []
        flagged_step = None
        early_stopped = False

        for t in range(start_token_pos, seq_len + 1):
            prefix_activations = self.extractor.extract_prefix_pooled(input_ids, prefix_len=t)
            score, pred = self.evaluate_prefix_features(prefix_activations)
            
            token_scores.append({"step": t, "score": score, "harmful": bool(pred)})

            if pred == 1 and flagged_step is None:
                flagged_step = t
                if early_stop:
                    early_stopped = True
                    break

        final_score = token_scores[-1]["score"] if token_scores else 0.0

        return {
            "token_scores": token_scores,
            "flagged_step": flagged_step,
            "early_stopped": early_stopped,
            "is_harmful": flagged_step is not None,
            "final_score": final_score
        }
