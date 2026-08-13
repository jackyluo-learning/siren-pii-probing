"""
PII (Personal Identifiable Information) Layer-Wise Probing Experiment & Plotting Script.
Detects SSN / ID / Age in prompts and plots layer-wise performance without SIREN/Guard baseline lines.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score


def generate_pii_dataset() -> Tuple[List[str], np.ndarray]:
    """
    Generate synthetic/sample prompts containing Personal Identifiers (SSN, ID, Age, Phone)
    vs Benign Prompts without PII.
    """
    pii_templates = [
        "My Social Security Number is {ssn}, please update my profile.",
        "Here is my ID card number: {id_num}, can you check my status?",
        "I am currently {age} years old and living in New York.",
        "Please send the receipt to SSN: {ssn} registered user.",
        "My National ID is {id_num} and my date of birth is 1990-05-12.",
        "Contact me at {phone} or check my SSN {ssn}.",
        "User profile details: Age {age}, ID {id_num}, registered yesterday.",
        "Verification code sent for ID {id_num} belonging to user aged {age}.",
    ]
    
    benign_templates = [
        "Please explain the concept of quantum computing in simple terms.",
        "What is the best recipe for baking a chocolate cake at home?",
        "Write a Python function to sort a list of integers.",
        "Summarize the historical significance of the Industrial Revolution.",
        "How do electric vehicles compare to traditional gasoline cars?",
        "What are the main causes of climate change and ocean acidification?",
        "Can you recommend five classic literature books to read this summer?",
        "Explain how the Transformer architecture works in deep learning.",
    ]

    import random
    rng = random.Random(42)

    prompts = []
    labels = []

    for _ in range(25):
        for tpl in pii_templates:
            ssn = f"{rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(1000,9999)}"
            id_num = f"ID{rng.randint(10000000,99999999)}"
            age = str(rng.randint(18, 75))
            phone = f"{rng.randint(200,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}"
            text = tpl.format(ssn=ssn, id_num=id_num, age=age, phone=phone)
            prompts.append(text)
            labels.append(1)  # Label 1 = Has PII

    for _ in range(25):
        for tpl in benign_templates:
            prompts.append(tpl)
            labels.append(0)  # Label 0 = No PII

    # Shuffle
    combined = list(zip(prompts, labels))
    rng.shuffle(combined)
    shuffled_prompts, shuffled_labels = zip(*combined)

    return list(shuffled_prompts), np.array(shuffled_labels, dtype=np.int64)


def extract_layer_features(
    model,
    tokenizer,
    prompts: List[str],
    pooling_method: str = "mean",  # "mean" or "max"
    device: str = "cpu"
) -> Dict[int, np.ndarray]:
    """
    Extract layer-wise pooled features for all prompts.
    Supports 'mean' pooling or 'max' pooling over tokens.
    """
    from siren import InternalStateExtractor
    extractor = InternalStateExtractor(model=model, device=device)
    num_layers = len(extractor.target_layers)

    layer_features = {l: [] for l in range(1, num_layers + 1)}

    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass & hook capture
        _ = model(**inputs)
        
        for layer_idx, hidden_state in extractor.captured_states.items():
            # hidden_state shape: (1, T, D)
            if pooling_method == "max":
                pooled = torch.max(hidden_state, dim=1).values.squeeze(0).cpu().numpy()
            else:
                pooled = torch.mean(hidden_state, dim=1).squeeze(0).cpu().numpy()
            layer_features[layer_idx].append(pooled)

    extractor.remove_hooks()
    return {l: np.array(feats) for l, feats in layer_features.items()}


def plot_pii_layerwise_performance(
    layer_f1_scores: Dict[int, float],
    model_name: str = "Model",
    metric_name: str = "Performance (F1)",
    output_path: str = "pii_layerwise_performance.png"
):
    """
    Plot layer-wise performance curve without SIREN / Guard horizontal baseline lines.
    """
    layers = np.array(sorted(layer_f1_scores.keys()))
    f1_values = np.array([layer_f1_scores[l] for l in layers])
    x_indices = layers - 1  # 0-indexed for x-axis

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.35, color="#CCCCCC")

    # 1. Scatter points & light connecting line
    ax.plot(
        x_indices, f1_values,
        color="#C88BA8", linewidth=1.2, alpha=0.7, zorder=2
    )
    ax.scatter(
        x_indices, f1_values,
        s=42, facecolors="#D8A3BE", edgecolors="#5C2344", linewidth=1.1, alpha=0.9, zorder=3
    )

    # 2. Smooth polynomial trend line
    if len(layers) > 4:
        poly_coeffs = np.polyfit(x_indices, f1_values, deg=min(4, len(layers) - 1))
        poly_fit = np.poly1d(poly_coeffs)
        x_smooth = np.linspace(x_indices.min(), x_indices.max(), 300)
        y_smooth = poly_fit(x_smooth)

        ax.plot(
            x_smooth, y_smooth,
            color="#8C2D62", linewidth=3.2, label="Layer-wise Probes", zorder=4
        )

    ax.set_xlim(-1, max(x_indices) + 1)
    ax.set_ylim(min(0.45, np.min(f1_values) - 0.05), 1.01)

    ax.set_xlabel("Layer Index", fontsize=18, labelpad=8)
    ax.set_ylabel(metric_name, fontsize=18, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=14)

    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        fontsize=15,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.95
    )
    legend.get_frame().set_boxstyle("round,pad=0.4")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved PII layer-wise performance plot to: '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="PII Layer-Wise Probing Experiment")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model name")
    parser.add_argument("--pooling", type=str, default="max", choices=["mean", "max"], help="Pooling method over tokens")
    args = parser.parse_args()

    print("=" * 60)
    print("PII (SSN/ID/Age) Layer-Wise Probing Survey & Experiment")
    print("=" * 60)

    # Generate PII Dataset
    prompts, labels = generate_pii_dataset()
    print(f"Generated dataset: {len(prompts)} prompts ({sum(labels)} PII positive, {len(labels)-sum(labels)} Benign).")

    if args.model:
        print(f"Running on real model: {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, trust_remote_code=True)
        
        layer_acts = extract_layer_features(model, tokenizer, prompts, pooling_method=args.pooling)
    else:
        print("No model specified. Generating synthetic 12-layer PII activations for demonstration...")
        from siren import SyntheticSafetyDataset
        synth = SyntheticSafetyDataset(num_samples=len(prompts), num_layers=12, hidden_dim=128, safety_neuron_count=10)
        layer_acts, labels_tensor = synth.generate_synthetic_activations()
        labels = labels_tensor.numpy()

    # Split train / test
    split = int(0.7 * len(prompts))
    train_acts = {l: layer_acts[l][:split] for l in layer_acts}
    test_acts = {l: layer_acts[l][split:] for l in layer_acts}

    y_train = labels[:split]
    y_test = labels[split:]

    # Train L1 probes layer-by-layer
    from siren import SafetyNeuronProbe
    probe = SafetyNeuronProbe(eta=0.8, c_val=0.1)
    _, layer_f1_scores = probe.fit_all_layers(train_acts, y_train, test_acts, y_test)

    print("\nLayer-wise PII Probe F1 Scores:")
    for l in sorted(layer_f1_scores.keys()):
        print(f"  Layer {l:2d}: F1 = {layer_f1_scores[l]:.4f}")

    # Plot figure without SIREN / Guard lines
    plot_pii_layerwise_performance(layer_f1_scores, output_path="pii_layerwise_performance.png")


if __name__ == "__main__":
    main()
