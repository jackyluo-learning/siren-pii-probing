"""
SIREN streaming (generation-time) harmfulness detection.

Reproduces the paper's Additional Results / Appendix B.1 streaming mode
(arXiv:2604.18519): a SIREN trained on FULL-SEQUENCE features is applied,
completely unchanged, to progressively longer generation prefixes.

    x_{l,<=t} = LLM_l(s_<=t)                              (Eq. 7)
    x*_{l,<=t} = (1/t) sum_{tau<=t} x_{l,tau}             (Eq. 8)   <- same pooling as training
    z_<=t = concat_l  alpha_l * [x*_{l,<=t}]_{S_l}        (Eq. 9)
    h_t = clf(z_<=t)                                                <- classifier NOT retrained

No parameters of the LLM, the safety-neuron probes, or the classifier are
updated -- a strict zero-shot test of whether sentence-level safety information
already manifests in prefix-level representations.

Efficiency note: in a causal decoder-only LM the hidden state at position tau
depends only on tokens <= tau, so ONE full-sequence forward pass yields every
prefix's states; the prefix mean is then a cumulative mean along the sequence
axis. Streaming scoring is therefore O(T), not O(T^2).

Outputs
  1. Detection-rate vs. token-position curve (paper Figure 3 analogue), with the
     false-positive rate on safe samples on the same axis.
  2. A token-level harmfulness heat strip for one example (Figure 8 analogue).

Protocol difference to state when citing: the paper measures detection latency
relative to a MANUALLY ANNOTATED unsafe span boundary (timely, +32, +64, +128,
+256 tokens) on the Think/Qwen3GuardTest benchmark. Those span annotations are
not publicly available here, so latency is measured from the start of the
sequence instead. The mechanism under test is identical; the x-axis origin differs.
"""

import argparse
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from siren.progress import track
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from siren import (
    InternalStateExtractor,
    AdaptiveNeuronAggregator,
    SirenMLPHead,
    StreamingModerator,
)
from siren.safety_benchmarks import load_safety_benchmarks
from run_real_layerwise_experiment import load_model_and_extractor

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Load a SIREN fitted by module 4                                              #
# --------------------------------------------------------------------------- #
def load_streaming_moderator(
    state_path: str,
    device: str,
    threshold: float = 0.5,
) -> Tuple[StreamingModerator, object, object, str]:
    """Rebuild the fitted SIREN and attach it to a live model + extractor."""
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model_name = state["model_name"]

    tokenizer, model, extractor, num_layers = load_model_and_extractor(model_name, device)

    aggregator = AdaptiveNeuronAggregator(
        safety_neurons=state["safety_neurons"],
        layer_f1_scores=state["layer_f1_scores"],
    )
    classifier = SirenMLPHead(
        input_dim=state["input_dim"], hidden_dim=state["mlp_hidden_dim"]
    )
    classifier.load_state_dict(state["classifier_state_dict"])

    moderator = StreamingModerator(
        extractor=extractor, aggregator=aggregator, classifier=classifier,
        threshold=threshold, device=device,
    )
    print(f"Loaded fitted SIREN: {num_layers} layers, z-dim {state['input_dim']}, "
          f"threshold {threshold}")
    return moderator, tokenizer, model, model_name


# --------------------------------------------------------------------------- #
# Streaming evaluation                                                         #
# --------------------------------------------------------------------------- #
def score_texts(
    moderator: StreamingModerator,
    tokenizer,
    texts: Sequence[str],
    max_length: int = 256,
    show_progress: bool = True,
) -> List[np.ndarray]:
    """Per-token harmfulness scores for each text (one forward pass per text)."""

    out: List[np.ndarray] = []
    seqs = list(texts)
    for text in (track(seqs, desc="streaming", unit="条") if show_progress else seqs):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"]
        if ids.shape[-1] == 0:
            out.append(np.zeros(0, dtype=np.float32))
            continue
        out.append(moderator.score_token_stream(ids))
    return out


def first_flag_positions(
    scores: Sequence[np.ndarray], threshold: float = 0.5, min_prefix: int = 8
) -> np.ndarray:
    """
    1-based token index where the score first crosses threshold; inf if never.

    Positions before `min_prefix` are ignored. The first token of a decoder-only
    LM carries a massive-activation "attention sink" residual (measured here:
    ~9x the full-sequence norm at mid depth), so a 1-3 token prefix mean is far
    outside the distribution the sequence-level classifier was fitted on and its
    output there is not meaningful. The paper does not hit this regime because it
    scores latency around an unsafe span deep inside a long reasoning trace.
    """
    pos = []
    for s in scores:
        usable = np.full_like(s, -np.inf, dtype=float)
        if len(s) >= min_prefix:
            usable[min_prefix - 1:] = s[min_prefix - 1:]
        hit = np.nonzero(usable >= threshold)[0]
        pos.append(float(hit[0] + 1) if len(hit) else np.inf)
    return np.asarray(pos, dtype=float)


def detection_curve(first_flags: np.ndarray, max_pos: int) -> np.ndarray:
    """Fraction of sequences flagged by token position t, for t = 1..max_pos."""
    ts = np.arange(1, max_pos + 1)
    return np.array([(first_flags <= t).mean() for t in ts])


def evaluate_streaming(
    moderator: StreamingModerator,
    tokenizer,
    harmful_texts: Sequence[str],
    safe_texts: Sequence[str],
    max_length: int = 256,
    threshold: float = 0.5,
    min_prefix: int = 8,
    min_tokens: int = 32,
) -> Dict[str, object]:
    """Detection rate on harmful vs. false-positive rate on safe, by token position."""
    def long_enough(texts):
        keep = [t for t in texts
                if len(tokenizer(t, truncation=True, max_length=max_length)["input_ids"]) >= min_tokens]
        return keep

    n_h_all, n_s_all = len(harmful_texts), len(safe_texts)
    harmful_texts, safe_texts = long_enough(harmful_texts), long_enough(safe_texts)
    print(f"\nKept sequences with >= {min_tokens} tokens: "
          f"harmful {len(harmful_texts)}/{n_h_all}, safe {len(safe_texts)}/{n_s_all} "
          f"(latency on a handful of tokens is not a meaningful streaming test)", flush=True)
    print(f"Scoring prefix-wise up to {max_length} tokens, warm-up {min_prefix} ...", flush=True)

    harmful_scores = score_texts(moderator, tokenizer, harmful_texts, max_length)
    safe_scores = score_texts(moderator, tokenizer, safe_texts, max_length)

    ff_harm = first_flag_positions(harmful_scores, threshold, min_prefix)
    ff_safe = first_flag_positions(safe_scores, threshold, min_prefix)

    det = detection_curve(ff_harm, max_length)
    fpr = detection_curve(ff_safe, max_length)

    detected = np.isfinite(ff_harm)
    return {
        "harmful_scores": harmful_scores,
        "safe_scores": safe_scores,
        "detection_curve": det,
        "fpr_curve": fpr,
        "first_flag_harmful": ff_harm,
        "first_flag_safe": ff_safe,
        "final_detection_rate": float(det[-1]),
        "final_fpr": float(fpr[-1]),
        "median_first_flag": float(np.median(ff_harm[detected])) if detected.any() else float("nan"),
        "min_prefix": min_prefix,
        "n_harmful": len(harmful_texts),
        "n_safe": len(safe_texts),
        "harmful_texts": list(harmful_texts),
    }


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #
def plot_detection_latency(res: Dict[str, object], output_fig: str, model_name: str):
    """Detection rate and false-positive rate as a function of token position."""
    det, fpr = res["detection_curve"], res["fpr_curve"]
    x = np.arange(1, len(det) + 1)

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.2

    fig, ax = plt.subplots(figsize=(8, 5.0), dpi=200)
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.35, color="#CCCCCC", zorder=1)

    ax.plot(x, det * 100, color="#8C2D62", linewidth=3.0,
            label="Detection rate (harmful)", zorder=4)
    ax.plot(x, fpr * 100, color="#C25E00", linewidth=2.2, linestyle="--",
            label="False-positive rate (safe)", zorder=4)

    warm = int(res.get("min_prefix", 0) or 0)
    if warm > 1:
        ax.axvspan(1, warm, color="#E9EDF1", zorder=1)
        ax.text(warm, 96, f" warm-up <{warm} tok", fontsize=9, color="#7A8894", va="top")

    for mark in (32, 64, 128, 256):
        if mark < len(det):
            ax.axvline(mark, color="#B9C6D2", linewidth=0.9, linestyle=":", zorder=2)
            ax.text(mark, 2, f" {mark}", fontsize=9, color="#8A98A6", va="bottom")

    ax.set_xlim(1, len(det))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Token position in stream", fontsize=15, labelpad=6)
    ax.set_ylabel("Rate (%)", fontsize=15, labelpad=6)
    ax.set_title(f"SIREN streaming detection on {model_name.split('/')[-1]} "
                 f"(zero-shot, prefix-pooled)", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc="lower right", fontsize=11, frameon=True, facecolor="white",
              edgecolor="#CCCCCC", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved streaming detection-latency plot to: '{output_fig}'")


def plot_token_heatstrip(
    tokens: Sequence[str],
    scores: np.ndarray,
    output_fig: str,
    threshold: float = 0.5,
    max_tokens: int = 80,
    min_prefix: int = 8,
):
    """Token-level harmfulness strip (paper Figure 8 analogue)."""
    tokens = list(tokens)[:max_tokens]
    scores = np.asarray(scores)[:max_tokens]

    cmap = LinearSegmentedColormap.from_list(
        "siren", ["#F1F4F7", "#F6D9A8", "#D98A5B", "#A32020"])

    per_row = 16
    n_rows = int(np.ceil(len(tokens) / per_row))
    fig, ax = plt.subplots(figsize=(min(14, per_row * 0.95), max(2.0, n_rows * 0.72)), dpi=200)
    ax.axis("off")

    for i, (tok, sc) in enumerate(zip(tokens, scores)):
        r, c = divmod(i, per_row)
        warm = i < min_prefix - 1
        ax.add_patch(plt.Rectangle(
            (c, -r), 0.94, 0.82, facecolor=cmap(float(sc)),
            edgecolor="#7A8894" if warm else "#DDE4EA",
            linestyle=":" if warm else "-", linewidth=1.1 if warm else 0.6))
        label = tok.replace("Ġ", " ").replace("Ċ", "\\n").strip() or "·"
        if len(label) > 9:
            label = label[:8] + "…"
        ax.text(c + 0.47, -r + 0.41, label, ha="center", va="center", fontsize=7.5,
                color="#16283D" if sc < 0.6 else "white")

    ax.set_xlim(-0.2, per_row + 0.2)
    ax.set_ylim(-n_rows + 0.1, 1.15)
    ax.set_title(f"Token-level harmfulness (threshold {threshold}); light = safe, dark = harmful\n"
                 f"dotted border = warm-up (<{min_prefix} tokens, not used for flagging)",
                 fontsize=10, pad=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.05, pad=0.04, aspect=40)
    cbar.set_label("P(harmful) after this token", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved token-level heat strip to: '{output_fig}'")


# --------------------------------------------------------------------------- #
# Live generation demo                                                         #
# --------------------------------------------------------------------------- #
def generation_demo(
    moderator: StreamingModerator,
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 48,
    threshold: float = 0.5,
    device: str = "cpu",
):
    """
    Actually generate a continuation, then score every prefix of prompt+generation.
    This is the deployment-shaped view: what SIREN would see as the model speaks.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    full_ids = out[:, : enc["input_ids"].shape[1] + max_new_tokens].cpu()
    scores = moderator.score_token_stream(full_ids)
    tokens = tokenizer.convert_ids_to_tokens(full_ids[0])
    prompt_len = enc["input_ids"].shape[1]

    hit = np.nonzero(scores >= threshold)[0]
    flag = int(hit[0] + 1) if len(hit) else None
    print(f"\n  prompt   : {prompt!r}")
    print(f"  generated: {tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)!r}")
    print(f"  score at end of prompt : {scores[prompt_len - 1]:.4f}")
    print(f"  final score            : {scores[-1]:.4f}")
    print(f"  first flagged position : {flag if flag else 'never'}"
          + (f" (token {tokens[flag-1]!r})" if flag else ""))
    return tokens, scores, prompt_len


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="checkpoints/siren_figure7_state.pt",
                        help="fitted SIREN saved by run_paper_exact_reproduction.py")
    parser.add_argument("--cap", type=int, default=2000, help="rows streamed per benchmark")
    parser.add_argument("--only", type=str, default=None, help="benchmark subset")
    parser.add_argument("--n-eval", type=int, default=60,
                        help="sequences per class for the latency evaluation")
    parser.add_argument("--max-length", type=int, default=256, help="max tokens scored per sequence")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-prefix", type=int, default=8,
                        help="warm-up: ignore flags before this token (attention-sink OOD)")
    parser.add_argument("--min-tokens", type=int, default=32,
                        help="skip eval sequences shorter than this")
    parser.add_argument("--demo-prompt", type=str, default=None,
                        help="optional prompt for the live generation demo")
    parser.add_argument("--output", type=str, default="streaming_detection_latency.png")
    parser.add_argument("--output-strip", type=str, default="streaming_token_strip.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("SIREN: Streaming (generation-time) token-level detection")
    print(f"Device: [{device}] | threshold: {args.threshold} | max tokens: {args.max_length}")
    print("=" * 72)

    print("\n1. Loading the SIREN fitted on full sequences (zero-shot reuse) ...")
    moderator, tokenizer, model, model_name = load_streaming_moderator(
        args.state, device, args.threshold)

    print("\n2. Loading held-out evaluation text ...")
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    corpus = load_safety_benchmarks(cap_per_dataset=args.cap, only=only, seed=42)
    te_txt, te_y = corpus.test_texts, corpus.y_test
    harmful = [t for t, y in zip(te_txt, te_y) if y == 1][: args.n_eval]
    safe = [t for t, y in zip(te_txt, te_y) if y == 0][: args.n_eval]

    print("\n3. Streaming evaluation (prefix-wise, classifier unchanged) ...")
    res = evaluate_streaming(moderator, tokenizer, harmful, safe,
                             max_length=args.max_length, threshold=args.threshold,
                             min_prefix=args.min_prefix, min_tokens=args.min_tokens)

    print("\n" + "=" * 72)
    print("Streaming results (zero-shot transfer from sequence-level SIREN):")
    print(f"  Sequences used: {res['n_harmful']} harmful / {res['n_safe']} safe "
          f"(>= {args.min_tokens} tokens)")
    print(f"  Detection rate on harmful (by token {args.max_length}) : "
          f"{res['final_detection_rate']*100:.1f}%")
    print(f"  False-positive rate on safe                        : {res['final_fpr']*100:.1f}%")
    print(f"  Median first-flag position (detected harmful)      : {res['median_first_flag']:.0f} tokens")
    for mark in (32, 64, 128):
        if mark <= args.max_length:
            print(f"  Detection @ {mark:3d} tokens : {res['detection_curve'][mark-1]*100:5.1f}%  "
                  f"(FPR {res['fpr_curve'][mark-1]*100:.1f}%)")
    print("=" * 72)

    plot_detection_latency(res, args.output, model_name)

    # Token strip for the earliest-detected harmful example.
    ff = res["first_flag_harmful"]
    if np.isfinite(ff).any():
        # Prefer a sequence that starts benign and crosses the threshold mid-stream:
        # it shows the streaming mechanism far better than one that is harmful from
        # the first word. Fall back to the earliest detection if none qualifies.
        mid = [(f, i) for i, f in enumerate(ff)
               if np.isfinite(f) and args.min_prefix < f <= 64]
        idx = max(mid)[1] if mid else int(np.argmin(np.where(np.isfinite(ff), ff, np.inf)))
        enc = tokenizer(res["harmful_texts"][idx], return_tensors="pt",
                        truncation=True, max_length=args.max_length)
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        plot_token_heatstrip(toks, res["harmful_scores"][idx], args.output_strip,
                             threshold=args.threshold, min_prefix=args.min_prefix)

    if args.demo_prompt:
        print("\n4. Live generation demo ...")
        generation_demo(moderator, tokenizer, model, args.demo_prompt,
                        threshold=args.threshold, device=device)


if __name__ == "__main__":
    main()
