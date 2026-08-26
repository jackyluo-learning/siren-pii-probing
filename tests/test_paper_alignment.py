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
    monkeypatch.setattr(sb, "_load_one",
                        lambda spec, cap, response_only=False: fake.get(spec.name, []))
    corpus = sb.load_safety_benchmarks(
        cap_per_dataset=100, only=["ToxicChat", "BeaverTails"], seed=1, verbose=False)
    import numpy as np
    ys = np.concatenate([corpus.y_train, corpus.y_val, corpus.y_test])
    # balanced -> equal classes overall
    assert abs(ys.mean() - 0.5) < 1e-9
    # disjoint, non-empty splits
    assert len(corpus.train_texts) > 0 and len(corpus.val_texts) > 0 and len(corpus.test_texts) > 0
    assert set(corpus.meta["loaded_datasets"]) == {"ToxicChat", "BeaverTails"}


def test_response_only_filters_prompt_level_datasets():
    """response_only must keep response-judgement rows and drop prompt-level ones."""
    from siren.safety_benchmarks import (
        _toxicchat, _openai_moderation, _aegis, _wildguard, _saferlhf, _beavertails)

    # Prompt-level datasets contribute nothing.
    assert _toxicchat({"user_input": "x", "toxicity": 1}, response_only=True) == []
    assert _openai_moderation({"prompt": "x", "H": 1}, response_only=True) == []
    # ...but still work in the default mixed mode.
    assert _toxicchat({"user_input": "x", "toxicity": 1}) == [("x", 1)]

    # WildGuard: drop the prompt-only sample, keep prompt+response.
    row = {"prompt": "p", "response": "r",
           "prompt_harm_label": "harmful", "response_harm_label": "unharmful"}
    assert _wildguard(row) == [("p", 1), ("p\nr", 0)]
    assert _wildguard(row, response_only=True) == [("p\nr", 0)]

    # Aegis: keep llm_response / combined, drop user_message.
    base = {"text": "t", "labels_0": "Safe", "labels_1": "Safe"}
    assert _aegis({**base, "text_type": "user_message"}, response_only=True) == []
    assert _aegis({**base, "text_type": "llm_response"}, response_only=True) == [("t", 0)]
    assert _aegis({**base, "text_type": "combined"}, response_only=True) == [("t", 0)]

    # Always-response-level datasets are unaffected, and the text still carries
    # the full prompt+response prefix (response_only changes labels, not scope).
    sr = _saferlhf({"prompt": "q", "response_0": "a", "is_response_0_safe": True},
                   response_only=True)
    assert sr == [("q\na", 0)]
    bt = _beavertails({"prompt": "u", "response": "v", "is_safe": False}, response_only=True)
    assert bt == [("u\nv", 1)]


def test_paired_response_adapters_keep_prompt_response_boundary():
    """Analyses need the boundary that the training corpus builders collapse."""
    from siren.safety_benchmarks import (
        _paired_beavertails, _paired_saferlhf, _paired_wildguard)

    bt = _paired_beavertails({"prompt": "p", "response": "r", "is_safe": False})
    assert len(bt) == 1 and bt[0].prompt == "p" and bt[0].response == "r" and bt[0].label == 1

    sr = _paired_saferlhf({"prompt": "q", "response_0": "a", "is_response_0_safe": True,
                           "response_1": "b", "is_response_1_safe": False})
    assert [(s.response, s.label) for s in sr] == [("a", 0), ("b", 1)]

    wg = _paired_wildguard({"prompt": "p", "response": "r", "response_harm_label": "harmful"})
    assert wg[0].label == 1
    # Missing pieces yield nothing rather than a malformed pair.
    assert _paired_beavertails({"prompt": "p", "response": "", "is_safe": False}) == []
    assert _paired_wildguard({"prompt": "p", "response": "r"}) == []


def test_multiclass_probe_selects_neurons_across_all_classes():
    """liblinear refuses K>=3, so the probe must wrap OvR and stack coef_."""
    import numpy as np
    from siren.probe import SafetyNeuronProbe, PAPER_C_GRID

    rng = np.random.RandomState(0)
    D, K, BLOCK = 40, 4, 5

    def make(n):
        y = rng.randint(0, K, n)
        X = rng.randn(n, D) * 0.4
        for k in range(K):                      # each class driven by its own block
            X[y == k, k * BLOCK:(k + 1) * BLOCK] += 2.5
        return X, y

    Xtr, ytr = make(400)
    Xva, yva = make(200)
    probe = SafetyNeuronProbe(eta=0.8, c_grid=PAPER_C_GRID, average="macro")
    neurons, f1 = probe.fit_all_layers({1: Xtr}, ytr, {1: Xva}, yva)

    assert probe.layer_probes[1].coef_.shape == (K, D)   # stacked per-class weights
    assert f1[1] > 0.9
    # Every class's driving block must be represented, which is the point of
    # taking the max magnitude across classes rather than the mean.
    for k in range(K):
        assert any(i in neurons[1] for i in range(k * BLOCK, (k + 1) * BLOCK))


def test_binary_probe_path_unchanged_by_multiclass_support():
    import numpy as np
    from siren.probe import SafetyNeuronProbe, PAPER_C_GRID

    rng = np.random.RandomState(1)
    X = rng.randn(300, 20)
    y = (X[:, 0] > 0).astype(int)
    Xv = rng.randn(120, 20)
    yv = (Xv[:, 0] > 0).astype(int)
    probe = SafetyNeuronProbe(eta=0.8, c_grid=PAPER_C_GRID, average="macro")
    probe.fit_layer(1, X, y, Xv, yv)
    assert probe.layer_probes[1].coef_.shape == (1, 20)   # still the plain path


def test_pii_task_builders_filter_and_label_correctly(monkeypatch):
    import numpy as np
    import siren.pii_benchmark as B

    rows = ([("mail a@b.com", {"EMAIL"})] * 30
            + [("ssn 123-45-6789", {"SOCIALNUM"})] * 30
            + [("both a@b.com 123-45-6789", {"EMAIL", "SOCIALNUM"})] * 10
            + [("city Beijing", {"CITY"})] * 10)
    monkeypatch.setattr(B, "_rows", lambda cap, language=None, verbose=True: iter(rows))

    # Binary: negatives are other-PII texts from the same corpus, never plain text.
    b = B.build_pii_binary_task(target="SOCIALNUM", cap=100, verbose=False)
    assert b.n_classes == 2 and set(np.unique(b.y_train)) <= {0, 1}

    # Multiclass: texts carrying several target categories are dropped, not guessed.
    m = B.build_pii_multiclass_task(categories=["EMAIL", "SOCIALNUM"], cap=100,
                                    min_per_class=10, verbose=False)
    assert m.class_names == ["EMAIL", "SOCIALNUM"]
    assert m.meta["dropped_mixed"] == 10     # the 10 two-label rows
    assert m.meta["dropped_none"] == 10      # the 10 CITY-only rows
