"""
SIREN Pipeline End-to-End Training and Evaluation Example Script.
Demonstrates training SIREN on synthetic activations or real Transformer model hidden states.
"""

import argparse
import os
import torch
import numpy as np
from siren import SyntheticSafetyDataset, SirenGuard


def main():
    parser = argparse.ArgumentParser(description="Train SIREN Guard Model")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic dataset and activations")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of synthetic samples")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size per layer")
    parser.add_argument("--eta", type=float, default=0.8, help="Cumulative threshold for safety neuron selection")
    parser.add_argument("--c_val", type=float, default=0.1, help="Inverse L1 regularization strength")
    parser.add_argument("--epochs", type=int, default=15, help="MLP classifier training epochs")
    parser.add_argument("--save_dir", type=str, default="checkpoints/siren_guard", help="Save directory")
    args = parser.parse_args()

    print("=" * 60)
    print("SIREN: LLM Safety From Within - Training Pipeline")
    print("=" * 60)

    if args.synthetic:
        print(f"Generating synthetic dataset: {args.num_samples} samples, {args.num_layers} layers, hidden_dim={args.hidden_dim}...")
        synth = SyntheticSafetyDataset(
            num_samples=args.num_samples,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            safety_neuron_count=16,
            seed=42
        )
        layer_acts, labels = synth.generate_synthetic_activations()

        # Split 70% train, 15% val, 15% test
        n_train = int(0.7 * args.num_samples)
        n_val = int(0.15 * args.num_samples)

        train_acts = {l: v[:n_train] for l, v in layer_acts.items()}
        val_acts = {l: v[n_train:n_train + n_val] for l, v in layer_acts.items()}
        test_acts = {l: v[n_train + n_val:] for l, v in layer_acts.items()}

        y_train = labels[:n_train]
        y_val = labels[n_train:n_train + n_val]
        y_test = labels[n_train + n_val:]

        print(f"Dataset split: Train={n_train}, Val={n_val}, Test={len(y_test)}")

    print("\nInitializing SirenGuard...")
    guard = SirenGuard(eta=args.eta, c_val=args.c_val, mlp_hidden_dim=256)

    print("\nStep 1 & 2: Localizing Safety Neurons & Computing Adaptive Layer Weights...")
    fit_info = guard.fit_from_activations(
        layer_activations_train=train_acts,
        y_train=y_train,
        layer_activations_val=val_acts,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=32,
        lr=1e-3
    )

    print("\nSafety Neurons Selection Summary:")
    print(f"Total Selected Safety Neurons: {fit_info['safety_neurons_count']} / {args.num_layers * args.hidden_dim}")
    print(f"Sparsity Reduction Ratio: {args.num_layers * args.hidden_dim / fit_info['safety_neurons_count']:.1f}x reduction")

    print("\nLayer Weights (alpha_l):")
    for layer_idx, alpha in fit_info['layer_weights'].items():
        f1 = fit_info['layer_f1_scores'][layer_idx]
        print(f"  Layer {layer_idx:2d}: Validation F1 = {f1:.3f} | Weight alpha_l = {alpha:.3f}")

    print("\nEvaluating on Test Set...")
    preds, probs = guard.predict_from_activations(test_acts)
    test_y_np = y_test.numpy()
    
    from sklearn.metrics import classification_report, roc_auc_score
    print("\nClassification Report:")
    print(classification_report(test_y_np, preds, target_names=["Safe", "Harmful"]))
    print(f"Test AUROC: {roc_auc_score(test_y_np, probs):.4f}")

    print(f"\nSaving model state to '{args.save_dir}'...")
    guard.save(args.save_dir)
    print("Saved successfully!")

    print("\nTesting Model Loading...")
    loaded_guard = SirenGuard.load(args.save_dir)
    loaded_preds, _ = loaded_guard.predict_from_activations(test_acts)
    assert np.array_equal(preds, loaded_preds), "Loaded model predictions do not match!"
    print("Model load and save sanity check PASSED!")

    print("\n" + "=" * 60)
    print("SIREN End-to-End Training Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
