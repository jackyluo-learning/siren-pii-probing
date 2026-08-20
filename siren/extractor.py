"""
Internal state extractor for Transformer-based LLMs.
Uses PyTorch forward hooks to capture hidden layer activations (residual stream / FFN)
and compute sequence-level or prefix-level mean-pooled representations.
"""

from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm


class InternalStateExtractor:
    """
    Extracts internal hidden representations from specified transformer layers of an LLM.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_names: Optional[List[str]] = None,
        extraction_point: str = "residual",  # "residual" or "ffn"
        device: Optional[str] = None
    ):
        """
        Args:
            model: PyTorch LLM model (e.g. AutoModelForCausalLM).
            target_layer_names: Explicit list of layer submodule names. If None, auto-detects transformer layers.
            extraction_point: Component to hook into ("residual" or "ffn").
            device: Computing device ('cpu', 'cuda', 'mps').
        """
        self.model = model
        self.extraction_point = extraction_point

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
        self.model.eval()

        self.hooks = []
        self.captured_states: Dict[int, torch.Tensor] = {}
        self.target_layers = self._locate_target_layers(target_layer_names)
        self._register_hooks()

    def _locate_target_layers(self, layer_names: Optional[List[str]]) -> List[Tuple[int, nn.Module]]:
        """Auto-detect transformer layer modules if layer_names is None."""
        layers = []
        
        if layer_names is not None:
            for i, name in enumerate(layer_names, start=1):
                module = dict(self.model.named_modules()).get(name)
                if module is not None:
                    layers.append((i, module))
            return layers

        # Common layer attributes in standard HuggingFace architectures
        candidate_parents = [
            getattr(self.model, "model", None),
            getattr(self.model, "transformer", None),
            getattr(self.model, "gpt_neox", None),
            self.model
        ]

        found_layer_list = None
        for parent in candidate_parents:
            if parent is None:
                continue
            for attr in ["layers", "h", "blocks"]:
                if hasattr(parent, attr):
                    found_layer_list = getattr(parent, attr)
                    break
            if found_layer_list is not None:
                break

        if found_layer_list is not None and isinstance(found_layer_list, (nn.ModuleList, list)):
            for idx, module in enumerate(found_layer_list, start=1):
                layers.append((idx, module))

        if not layers:
            raise ValueError(
                "Could not automatically detect transformer layers in model. "
                "Please specify target_layer_names explicitly."
            )

        return layers

    def _register_hooks(self):
        """Register PyTorch forward hooks on detected layers."""
        self.remove_hooks()
        self.captured_states = {}

        for layer_idx, module in self.target_layers:
            target_module = module
            if self.extraction_point == "ffn":
                # Attempt to find FFN module within transformer block
                for ffn_attr in ["mlp", "ffn", "feed_forward"]:
                    if hasattr(module, ffn_attr):
                        target_module = getattr(module, ffn_attr)
                        break

            def get_hook(idx: int):
                def hook(mod, input_args, output):
                    # Handle layer output formats (tensor vs tuple)
                    if isinstance(output, tuple):
                        tensor_output = output[0]
                    else:
                        tensor_output = output
                    # Store tensor on CPU to avoid CUDA/MPS memory overflow
                    self.captured_states[idx] = tensor_output.detach().cpu()
                return hook

            hook_handle = target_module.register_forward_hook(get_hook(layer_idx))
            self.hooks.append(hook_handle)

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @torch.no_grad()
    def extract_sequence_pooled(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Extract mean-pooled internal states over full sequence for input_ids.
        
        Args:
            input_ids: Tensor of shape (B, T)
            attention_mask: Tensor of shape (B, T)
            
        Returns:
            Dict mapping layer_idx (1..L) to pooled tensor of shape (B, D)
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Guard against an empty tokenization (0-length sequence), which would
        # otherwise crash attention with a 0-element reshape. Substitute a single
        # token so a (meaningless but finite) pooled vector is produced and the
        # per-sample feature/label alignment is preserved.
        if input_ids.shape[-1] == 0:
            input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)

        self.captured_states.clear()

        # Forward pass through base LLM (frozen)
        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        pooled_results = {}
        for layer_idx, hidden_state in self.captured_states.items():
            # hidden_state: (B, T, D)
            if attention_mask is not None:
                mask_expanded = attention_mask.cpu().unsqueeze(-1).float()  # (B, T, 1)
                sum_hidden = torch.sum(hidden_state * mask_expanded, dim=1)  # (B, D)
                lengths = torch.clamp(mask_expanded.sum(dim=1), min=1.0)
                mean_pooled = sum_hidden / lengths
            else:
                mean_pooled = torch.mean(hidden_state, dim=1)  # (B, D)
                
            pooled_results[layer_idx] = mean_pooled

        return pooled_results

    @torch.no_grad()
    def extract_prefix_pooled(
        self,
        input_ids: torch.Tensor,
        prefix_len: int
    ) -> Dict[int, torch.Tensor]:
        """
        Extract mean-pooled internal states restricted to sequence prefix <= prefix_len.

        Args:
            input_ids: Tensor of shape (1, T) or (B, T)
            prefix_len: Active prefix cutoff length t

        Returns:
            Dict mapping layer_idx (1..L) to pooled tensor of shape (B, D)
        """
        prefix_ids = input_ids[:, :prefix_len]
        return self.extract_sequence_pooled(prefix_ids)

    @torch.no_grad()
    def extract_all_prefix_pooled(
        self,
        input_ids: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Prefix mean-pooled states for EVERY prefix length t = 1..T (paper Eq. 8),
        computed with a single forward pass.

        In a causal decoder-only LM, position tau attends only to positions <= tau,
        so the hidden state at tau from one full-sequence forward is identical to
        the one obtained by feeding only s<=t (for any t >= tau). The prefix mean
        is therefore a cumulative mean along the sequence axis. This makes
        streaming evaluation O(T) forward cost instead of O(T^2).

        Args:
            input_ids: Tensor of shape (1, T) -- a single, unpadded sequence.

        Returns:
            Dict mapping layer_idx (1..L) to a tensor of shape (T, D), where row t-1
            is the mean-pooled representation of the prefix s<=t.
        """
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError("extract_all_prefix_pooled expects input_ids of shape (1, T).")

        input_ids = input_ids.to(self.device)
        if input_ids.shape[-1] == 0:
            raise ValueError("Cannot stream-score an empty sequence.")

        attention_mask = torch.ones_like(input_ids)
        self.captured_states.clear()
        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        seq_len = input_ids.shape[-1]
        denom = torch.arange(1, seq_len + 1, dtype=torch.float32).unsqueeze(1)  # (T, 1)

        prefix_pooled = {}
        for layer_idx, hidden_state in self.captured_states.items():
            # hidden_state: (1, T, D) on CPU (hooks detach to CPU)
            hs = hidden_state[0].float()
            prefix_pooled[layer_idx] = torch.cumsum(hs, dim=0) / denom

        return prefix_pooled
