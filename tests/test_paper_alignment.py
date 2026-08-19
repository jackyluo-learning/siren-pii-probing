"""
Regression tests for paper-methodology alignment (arXiv:2604.18519).
"""
import numpy as np
from siren.probe import SafetyNeuronProbe, PAPER_C_GRID
from siren.pii_dataset_generator import build_layerwise_pii_benchmark


def test_c_grid_selected_from_grid_on_validation():
    # Grid search must pick a C strictly from PAPER_C_GRID, using the val set.
    rng = np.random.RandomState(0)
    d = 32
    def make(n):
        X = rng.randn(n, d)
        y = (X[:, 0] + 0.3 * rng.randn(n) > 0).astype(int)
        return X, y
    Xtr, ytr = make(120); Xva, yva = make(60)
    probe = SafetyNeuronProbe(eta=0.8, c_grid=PAPER_C_GRID, average="macro")
    probe.fit_layer(1, Xtr, ytr, Xva, yva)
    assert probe.layer_best_c[1] in PAPER_C_GRID
    assert 0.0 <= probe.layer_f1_scores[1] <= 1.0


def test_no_grid_without_val_uses_fixed_c():
    # Backwards-compatible: no val => fixed c_val, no grid search.
    rng = np.random.RandomState(1)
    X = rng.randn(80, 16); y = rng.randint(0, 2, 80)
    probe = SafetyNeuronProbe(eta=0.8, c_val=0.1, c_grid=PAPER_C_GRID)
    probe.fit_layer(1, X, y)  # no validation set
    assert probe.layer_best_c[1] == 0.1


def test_pii_benchmark_cue_holdout_and_digit_alignment():
    b = build_layerwise_pii_benchmark(n_train=40, n_val=20, n_test=40, seed=7)
    # Cue vocabularies disjoint across train/val vs test.
    seen = " ".join(b.train_prompts + b.val_prompts)
    for cue in ("Passport", "Driver License", "Invoice", "Tracking"):
        assert cue not in seen, f"test cue leaked into train/val: {cue}"
    heldout = " ".join(b.test_prompts)
    for cue in ("SSN", "Tax ID", "Order ID", "Product SKU", "Item Batch"):
        assert cue not in heldout, f"train cue leaked into test: {cue}"
    # Balanced labels.
    assert b.y_train.mean() == 0.5 and b.y_test.mean() == 0.5
    # Digit alignment: consecutive positive/negative pairs share the identifier.
    import re
    num = lambda s: re.search(r"\d{3}-\d{2}-\d{4}", s).group(0)
    assert num(b.train_prompts[0]) == num(b.train_prompts[1])


def test_safety_benchmark_label_mappers():
    # Verified schemas -> binary label (1=harmful/unsafe).
    from siren.safety_benchmarks import (
        _toxicchat, _openai_moderation, _aegis, _wildguard, _saferlhf, _beavertails)
    assert _toxicchat({"user_input": "hi", "toxicity": 0}) == [("hi", 0)]
    assert _toxicchat({"user_input": "x", "toxicity": 1}) == [("x", 1)]
    assert _openai_moderation({"prompt": "p", "H": 1}) == [("p", 1)]
    assert _openai_moderation({"prompt": "p"}) == [("p", 0)]
    # Aegis majority vote of annotator labels.
    assert _aegis({"text": "t", "labels_0": "Safe", "labels_1": "Safe", "labels_2": "Violence"}) == [("t", 0)]
    assert _aegis({"text": "t", "labels_0": "Violence", "labels_1": "Hate", "labels_2": "Safe"}) == [("t", 1)]
    # WildGuard yields up to two samples (prompt, prompt+response).
    wg = _wildguard({"prompt": "p", "response": "r",
                     "prompt_harm_label": "unharmful", "response_harm_label": "harmful"})
    assert wg == [("p", 0), ("p\nr", 1)]
    # SafeRLHF: one sample per response, label = not safe.
    sr = _saferlhf({"prompt": "q", "response_0": "a", "is_response_0_safe": True,
                    "response_1": "b", "is_response_1_safe": False})
    assert sr == [("q\na", 0), ("q\nb", 1)]
    assert _beavertails({"prompt": "u", "response": "v", "is_safe": False}) == [("u\nv", 1)]


def test_safety_benchmark_aggregation_balances_and_splits(monkeypatch):
    # Fake two datasets so no download is needed; check balance + 3-way split.
    import siren.safety_benchmarks as sb
    fake = {
        "ToxicChat": [("safe%d" % i, 0) for i in range(80)] + [("bad%d" % i, 1) for i in range(20)],
        "BeaverTails": [("ok%d" % i, 0) for i in range(10)] + [("harm%d" % i, 1) for i in range(60)],
    }
    monkeypatch.setattr(sb, "_load_one", lambda spec, cap: fake.get(spec.name, []))
    corpus = sb.load_safety_benchmarks(
        cap_per_dataset=100, only=["ToxicChat", "BeaverTails"], seed=1, verbose=False)
    import numpy as np
    ys = np.concatenate([corpus.y_train, corpus.y_val, corpus.y_test])
    # balanced -> equal classes overall
    assert abs(ys.mean() - 0.5) < 1e-9
    # disjoint, non-empty splits
    assert len(corpus.train_texts) > 0 and len(corpus.val_texts) > 0 and len(corpus.test_texts) > 0
    assert set(corpus.meta["loaded_datasets"]) == {"ToxicChat", "BeaverTails"}
