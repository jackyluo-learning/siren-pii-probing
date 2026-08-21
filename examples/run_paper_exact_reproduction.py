"""
SIREN Paper Reproduction — real Qwen3-4B backbone, real seven-benchmark corpus.

Fully replicates the paper's Figure 7 setting (arXiv:2604.18519):
  - Backbone: Qwen3-4B (36 layers), the paper's Figure 7 model.
  - Training data: the seven safety benchmarks (ToxicChat, OpenAIModeration,
    Aegis, Aegis2.0, WildGuard, SafeRLHF, BeaverTails) aggregated to binary.
  - Methodology: mean pooling, train/val/test split, per-layer L1 probe with C
    grid {100,200,500,1000} selected on validation Macro-F1, adaptive alpha_l
    from validation performance, Macro-F1 reporting.

The paper's officially reported SIREN=0.867 / Guard=0.834 (Qwen3-4B) are drawn
as dotted reference lines; the solid curve and "SIREN (measured)" line are what
this run actually computes.

Gated datasets (WildGuard, and possibly NVIDIA Aegis) require a one-time
`huggingface_hub.notebook_login()` and accepting each dataset's terms. Any
benchmark that cannot be loaded is skipped with a message and the run continues.
"""

import argparse
import warnings

import torch

from run_real_layerwise_experiment import (
    measure_layerwise_f1_presplit, plot_layerwise, save_siren_state)
from siren.safety_benchmarks import load_safety_benchmarks

warnings.filterwarnings("ignore")

# Paper-reported reference numbers (Qwen3-4B, 7-benchmark average).
PAPER_SIREN_F1 = 0.867
PAPER_GUARD_F1 = 0.834


def run_reproduction(
    model_name: str = "Qwen/Qwen3-4B",
    cap_per_dataset: int = 2000,
    only=None,
    response_only: bool = False,
    probe_train_cap: int = 4000,
    save_state: str = "checkpoints/siren_figure7_state.pt",
    output_fig: str = "exact_figure7_reproduction.png",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 72)
    print("SIREN: Paper Reproduction — real model + real seven-benchmark corpus")
    print(f"Backbone: [{model_name}] | Device: [{device}] | cap/dataset: {cap_per_dataset}")
    print("=" * 72)

    print("\n1. Loading the seven safety benchmarks ...")
    corpus = load_safety_benchmarks(cap_per_dataset=cap_per_dataset, only=only,
                                    response_only=response_only, seed=42)

    print("\n2. Extracting mean-pooled features and running the SIREN pipeline ...")
    res = measure_layerwise_f1_presplit(
        model_name,
        corpus.train_texts, corpus.y_train,
        corpus.val_texts, corpus.y_val,
        corpus.test_texts, corpus.y_test,
        mlp_hidden_dim=256, epochs=20, probe_train_cap=probe_train_cap, device=device,
    )

    print("\nPer-layer TEST Macro-F1 (C selected on validation):")
    for l in sorted(res["test_f1"]):
        print(f"  Layer {l:2d}: F1={res['test_f1'][l]:.4f} | C={res['best_c'][l]:g} "
              f"| neurons={res['safety_neurons'][l]}")
    print(f"\nMeasured max single-layer Macro-F1 : {max(res['test_f1'].values()):.4f}")
    print(f"Measured SIREN aggregated Macro-F1 : {res['siren_test_f1']:.4f}")
    print(f"Total safety neurons (z dim)       : {res['total_neurons']}")
    print(f"Datasets loaded                    : {corpus.meta['loaded_datasets']}")
    print(f"Paper reference (Qwen3-4B)         : SIREN={PAPER_SIREN_F1}, Guard={PAPER_GUARD_F1}")

    if save_state:
        save_siren_state(res, model_name, save_state)

    plot_layerwise(
        res["test_f1"], res["siren_test_f1"], output_fig,
        title=f"SIREN reproduction on {model_name.split('/')[-1]} (7-benchmark, measured)",
        reference_lines=[
            ("Paper SIREN (Qwen3-4B)", PAPER_SIREN_F1, "#1F77B4"),
            ("Paper Guard (Qwen3-4B)", PAPER_GUARD_F1, "#FF7F0E"),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--cap", type=int, default=2000,
                        help="max rows streamed per benchmark")
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated subset, e.g. 'ToxicChat,BeaverTails'")
    parser.add_argument("--response-only", action="store_true",
                        help="train on response-level labels only (drops ToxicChat / "
                             "OpenAIModeration). Makes the guard answer 'did the assistant "
                             "produce harmful content', so a plain absolute threshold can "
                             "tell a refusal apart from a compliance.")
    parser.add_argument("--probe-cap", type=int, default=4000,
                        help="balanced train-row cap for L1 probe fitting (0 = use all)")
    parser.add_argument("--save-state", type=str, default="checkpoints/siren_figure7_state.pt",
                        help="where to persist the fitted SIREN (for streaming eval); '' to skip")
    parser.add_argument("--output", type=str, default="exact_figure7_reproduction.png")
    args = parser.parse_args()
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    run_reproduction(args.model, cap_per_dataset=args.cap, only=only,
                     response_only=args.response_only,
                     probe_train_cap=args.probe_cap, save_state=args.save_state,
                     output_fig=args.output)
