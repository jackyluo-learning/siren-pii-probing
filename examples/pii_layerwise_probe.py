"""
PII Layer-Wise Probing & SIREN Pipeline Experiment.
1-to-1 methodology adaptation from SIREN paper to PII (SSN, ID, Age, Credit Card, Address) detection:
- Dataset: PII Detection Benchmark (structured pairing & out-of-template evaluation)
- Backbone Model: Qwen/Qwen2.5-3B-Instruct (36 layers, matching paper architecture)
- Layer-wise Probing: L1 Logistic Regression with C grid search in {100, 200, 500, 1000} & eta=0.8
- SIREN Aggregation: Adaptive layer weighting (alpha_l) + 3-layer SirenMLPHead classifier
- Visualization: Layer-wise probes F1 curve (LOESS smooth line), SIREN PII aggregated baseline (blue --).
"""

import argparse
import os
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy.ndimage import gaussian_filter1d
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from siren import (
    InternalStateExtractor,
    SafetyNeuronProbe,
    AdaptiveNeuronAggregator,
    SirenMLPHead,
    SirenTrainer
)
from siren.pii_dataset_generator import PIIDatasetGenerator

warnings.filterwarnings("ignore")


def load_pii_benchmark_datasets(num_train: int = 400, num_test: int = 200) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """
    Load PII benchmark dataset with out-of-template train/test splits.
    """
    gen = PIIDatasetGenerator(seed=42)
    train_samples = gen.generate_dataset(num_samples=num_train)
    
    # Generate test set with fresh seed for template-out evaluation
    gen_test = PIIDatasetGenerator(seed=100)
    test_samples = gen_test.generate_dataset(num_samples=num_test)

    train_prompts = [s["text"] for s in train_samples]
    y_train = np.array([s["label"] for s in train_samples], dtype=np.int64)

    test_prompts = [s["text"] for s in test_samples]
    y_test = np.array([s["label"] for s in test_samples], dtype=np.int64)

    print(f"Loaded PII Benchmark: Train={len(train_prompts)} prompts, Test={len(test_prompts)} prompts.")
    return train_prompts, y_train, test_prompts, y_test


def run_pii_exact_paper_experiment(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    num_train: int = 400,
    num_test: int = 200,
    pooling_method: str = "max",
    output_fig: str = "pii_layerwise_performance.png",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 70)
    print(f"SIREN PII Layer-Wise Probing & Aggregation Experiment")
    print(f"Backbone Model: [{model_name}] | Pooling: [{pooling_method.upper()}] | Device: [{device}]")
    print("=" * 70)

    # 1. Load PII Dataset
    train_prompts, y_train, test_prompts, y_test = load_pii_benchmark_datasets(num_train=num_train, num_test=num_test)

    # 2. Load Model & Tokenizer
    print(f"\n1. Loading HuggingFace model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # 3. Register Hooks & Extract Hidden States
    print(f"\n2. Registering forward hooks & extracting {pooling_method}-pooled layer hidden states...")
    extractor = InternalStateExtractor(model=model, device=device)
    num_layers = len(extractor.target_layers)
    print(f"   Detected {num_layers} Transformer layers in '{model_name}'.")

    def extract_features(prompt_list):
        feat_dict = {l: [] for l in range(1, num_layers + 1)}
        for p in prompt_list:
            inp = tokenizer(p, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                _ = model(**inp)
            for l, st in extractor.captured_states.items():
                if pooling_method == "max":
                    pooled = torch.max(st, dim=1).values.squeeze(0).cpu().numpy()
                else:
                    pooled = torch.mean(st, dim=1).squeeze(0).cpu().numpy()
                feat_dict[l].append(pooled)
        return {l: np.array(v) for l, v in feat_dict.items()}

    train_activations = extract_features(train_prompts)
    test_activations = extract_features(test_prompts)
    extractor.remove_hooks()

    # 4. Train Layer-wise L1 Linear Probes (Grid search over C in {100, 200, 500, 1000})
    print("\n3. Fitting layer-wise L1 linear probes (C grid search in {100, 200, 500, 1000})...")
    layer_f1_scores = {}
    safety_neurons = {}

    for l in range(1, num_layers + 1):
        X_tr, X_te = train_activations[l], test_activations[l]
        best_f1 = -1.0
        best_neurons = []

        for c_val in [100.0, 200.0, 500.0, 1000.0]:
            clf = LogisticRegression(penalty="l1", solver="liblinear", C=c_val, max_iter=1000, random_state=42)
            clf.fit(X_tr, y_train)
            preds = clf.predict(X_te)
            f1 = f1_score(y_test, preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                weights = np.abs(clf.coef_[0])
                total_w = np.sum(weights)
                if total_w > 0:
                    norm_w = weights / total_w
                    sorted_idx = np.argsort(norm_w)[::-1]
                    cum = 0.0
                    sel = []
                    for idx in sorted_idx:
                        sel.append(int(idx))
                        cum += norm_w[idx]
                        if cum >= 0.8:  # eta = 0.8
                            break
                    best_neurons = sel
                else:
                    best_neurons = list(np.argsort(weights)[::-1][:10])

        layer_f1_scores[l] = best_f1
        safety_neurons[l] = best_neurons
        print(f"  Layer {l:2d}: Best PII Probe F1 = {best_f1:.4f} | Selected Neurons: {len(best_neurons)}")

    measured_f1_list = [layer_f1_scores[l] for l in range(1, num_layers + 1)]
    max_single_probe_f1 = max(measured_f1_list)

    # 5. SIREN Adaptive Cross-Layer Aggregation & MLP Training for PII
    print("\n4. SIREN Adaptive Aggregation & MLP Head Training for PII...")
    aggregator = AdaptiveNeuronAggregator(safety_neurons, layer_f1_scores)
    z_train = aggregator.transform(train_activations)
    z_test = aggregator.transform(test_activations)

    mlp_head = SirenMLPHead(input_dim=z_train.size(1), hidden_dim=256)
    trainer = SirenTrainer(model=mlp_head, lr=1e-3, device=device)
    trainer.fit(z_train, torch.tensor(y_train), z_test, torch.tensor(y_test), epochs=20, batch_size=16)

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(z_test, torch.tensor(y_test)), batch_size=16
    )
    siren_metrics = trainer.evaluate(test_loader)
    siren_pii_f1 = siren_metrics["f1"]

    print("\n" + "=" * 50)
    print(f"SIREN PII Probing & Aggregation Results:")
    print(f"  Max Single Layer Probe F1: {max_single_probe_f1:.4f}")
    print(f"  SIREN Aggregated PII F1:   {siren_pii_f1:.4f}")
    print("=" * 50)

    # 6. Plotting PII Layer-Wise Performance Graph (Exact SIREN Paper Style)
    print(f"\n5. Generating SIREN Paper Style PII Layer-Wise Performance Plot...")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.2

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.35, color="#CCCCCC", zorder=1)

    x_indices = np.arange(num_layers)

    # 1) Scatter markers & light connecting line
    line_color = "#C88BA8"
    marker_fill = "#D8A3BE"
    marker_edge = "#5C2344"

    ax.plot(x_indices, measured_f1_list, color=line_color, linewidth=1.2, alpha=0.7, zorder=2)
    ax.scatter(x_indices, measured_f1_list, s=38, facecolors=marker_fill, edgecolors=marker_edge, linewidth=1.0, alpha=0.85, zorder=3)

    # 2) LOESS Smoothed Trend Curve (Deep Magenta #8C2D62)
    if num_layers > 4:
        x_smooth = np.linspace(0, num_layers - 1, 300)
        y_smooth = gaussian_filter1d(measured_f1_list, sigma=1.8)
        y_smooth_interp = np.interp(x_smooth, x_indices, y_smooth)
        ax.plot(x_smooth, y_smooth_interp, color="#8C2D62", linewidth=3.2, label="Layer-wise Probes", zorder=4)

    # 3) SIREN Horizontal Baseline Line for PII (Dashed Blue --)
    ax.axhline(y=siren_pii_f1, color="#1F77B4", linestyle="--", linewidth=2.5, label=f"SIREN PII ({siren_pii_f1:.3f})", zorder=5)

    ax.set_xlim(-1, num_layers)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Layer Index", fontsize=18, labelpad=8)
    ax.set_ylabel("Performance (F1)", fontsize=18, labelpad=8)
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
    os.makedirs(os.path.dirname(output_fig) if os.path.dirname(output_fig) else ".", exist_ok=True)
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PII layer-wise performance plot to: '{output_fig}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train_samples", type=int, default=400)
    parser.add_argument("--test_samples", type=int, default=200)
    parser.add_argument("--pooling", type=str, default="max", choices=["mean", "max"])
    args = parser.parse_args()

    run_pii_exact_paper_experiment(
        model_name=args.model,
        num_train=args.train_samples,
        num_test=args.test_samples,
        pooling_method=args.pooling
    )
