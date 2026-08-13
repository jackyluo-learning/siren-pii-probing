"""
SIREN Streaming Moderation Demonstration Script.
Simulates real-time token generation and demonstrates millisecond-level early stopping.
"""

import time
import torch
import numpy as np
from siren import SyntheticSafetyDataset, SirenGuard, StreamingModerator


def simulate_token_stream(is_harmful_sequence: bool = True, seq_len: int = 15, num_layers: int = 12, hidden_dim: int = 256):
    """
    Simulates streaming tokens and generating prefix mean-pooled hidden states step by step.
    """
    rng = np.random.RandomState(42 if is_harmful_sequence else 100)
    
    tokens = [
        "How", " to", " build", " a", " dangerous", " explosive", " device", " at", " home", " step", " by", " step", " guide", " for", " attacks"
    ] if is_harmful_sequence else [
        "How", " to", " make", " a", " delicious", " chocolate", " cake", " at", " home", " step", " by", " step", " guide", " for", " baking"
    ]

    prefix_history = []
    
    for t in range(1, len(tokens) + 1):
        # Generate synthetic activations for step t
        step_acts = {}
        for layer_idx in range(1, num_layers + 1):
            acts = rng.normal(loc=0.0, scale=1.0, size=(1, hidden_dim))
            # Inject harmful signal if sequence is harmful and token step >= 4 (around 'dangerous')
            if is_harmful_sequence and t >= 4:
                safety_indices = np.arange(16) + (layer_idx * 3) % (hidden_dim - 16)
                acts[:, safety_indices] += 2.5 * (t / seq_len)
            step_acts[layer_idx] = torch.tensor(acts, dtype=torch.float32)
            
        prefix_history.append((tokens[:t], step_acts))
        
    return prefix_history


def main():
    print("=" * 60)
    print("SIREN: Real-Time Streaming Moderation Demo")
    print("=" * 60)

    print("1. Preparing SirenGuard Model...")
    synth = SyntheticSafetyDataset(num_samples=200, num_layers=12, hidden_dim=256)
    layer_acts, labels = synth.generate_synthetic_activations()
    
    guard = SirenGuard(eta=0.8, c_val=0.1, mlp_hidden_dim=256)
    guard.fit_from_activations(layer_acts, labels, epochs=10)

    streaming_mod = guard.get_streaming_moderator()

    for is_harmful, prompt_desc in [(False, "Benign Query"), (True, "Harmful Attack Query")]:
        print("\n" + "-" * 50)
        print(f"Simulating Generation for: [{prompt_desc}]")
        print("-" * 50)

        prefix_stream = simulate_token_stream(is_harmful_sequence=is_harmful)
        
        for step, (tokens_so_far, prefix_activations) in enumerate(prefix_stream, start=1):
            start_time = time.perf_counter()
            score, pred = streaming_mod.evaluate_prefix_features(prefix_activations)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            current_text = "".join(tokens_so_far)
            status_symbol = "🚨 [FLAGGED / UNSAFE]" if pred == 1 else "✅ [SAFE]"

            print(f"Step {step:2d} | Token: '{tokens_so_far[-1]:>12s}' | Score: {score:.4f} | Status: {status_symbol} | Latency: {latency_ms:.2f}ms")

            if pred == 1:
                print(f"\n⚡ Early Stopping Triggered at Token #{step} ('{tokens_so_far[-1]}')!")
                print(f"   Generated Text Cutoff: \"{current_text}\"")
                break

    print("\n" + "=" * 60)
    print("SIREN Streaming Moderation Demo Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
