"""
SIREN: LLM Safety From Within - Detecting Harmful Content with Internal Representations
Official-spec local reproduction framework (ACL 2026 / arXiv:2604.18519).
"""

__version__ = "0.1.0"
__author__ = "SIREN Local Reproduction Suite"

from siren.dataset import SafetyDataset, SyntheticSafetyDataset
from siren.extractor import InternalStateExtractor
from siren.probe import SafetyNeuronProbe
from siren.aggregator import AdaptiveNeuronAggregator
from siren.classifier import SirenMLPHead, SirenTrainer
from siren.streaming import StreamingModerator
from siren.pipeline import SirenGuard

__all__ = [
    "SafetyDataset",
    "SyntheticSafetyDataset",
    "InternalStateExtractor",
    "SafetyNeuronProbe",
    "AdaptiveNeuronAggregator",
    "SirenMLPHead",
    "SirenTrainer",
    "StreamingModerator",
    "SirenGuard",
]
