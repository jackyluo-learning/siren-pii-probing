"""
Loader for the seven safety benchmarks used to train SIREN in the paper
(arXiv:2604.18519, Sec. 4.1): ToxicChat, OpenAIModeration, Aegis, Aegis2.0,
WildGuard, SafeRLHF, BeaverTails.

Each benchmark is mapped to a binary (text, label) list with the paper's
convention **label 1 = harmful/unsafe, 0 = safe**, aggregating any
multi-category taxonomy into a single binary label. Datasets are streamed and
capped so the notebook stays runnable on a single Colab GPU.

Some sources are gated (WildGuard, and the NVIDIA Aegis sets may require
accepting terms). Loading is resilient: a benchmark that errors or is
unauthorized is skipped with a clear message, and the run continues on whatever
loaded. To use the gated sets on Colab, run once:

    from huggingface_hub import notebook_login; notebook_login()

and accept each dataset's terms on its Hugging Face page.
"""

from dataclasses import dataclass, field
from itertools import islice
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


Sample = Tuple[str, int]


def _truthy_unsafe(v) -> int:
    """Map a per-category flag to 1 (harmful) if it is set."""
    if v is None:
        return 0
    if isinstance(v, bool):
        return int(v)
    try:
        return int(float(v) > 0)
    except (TypeError, ValueError):
        s = str(v).strip().lower()
        return int(s in {"1", "true", "yes", "unsafe", "harmful"})


# --------------------------------------------------------------------------- #
# Per-dataset row -> [(text, label)] adapters (schemas verified on HF viewer)  #
# --------------------------------------------------------------------------- #
def _toxicchat(row, response_only: bool = False) -> List[Sample]:
    # Prompt-level only: `toxicity` judges the user's message, not any reply.
    if response_only:
        return []
    text = row.get("user_input")
    if not text:
        return []
    return [(text, int(_truthy_unsafe(row.get("toxicity"))))]


_OPENAI_CATS = ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")


def _openai_moderation(row, response_only: bool = False) -> List[Sample]:
    # Prompt-level only: the category flags describe the prompt itself.
    if response_only:
        return []
    text = row.get("prompt")
    if not text:
        return []
    label = int(any(_truthy_unsafe(row.get(c)) for c in _OPENAI_CATS))
    return [(text, label)]


def _aegis(row, response_only: bool = False) -> List[Sample]:
    # Aegis 1.0/2.0: `text` + up to 5 annotator votes labels_0..labels_4
    # (value "Safe" vs a harm category). Also tolerate a single `prompt_label`.
    # `text_type` marks what the annotation is about: user_message, llm_response,
    # combined, multi_turn. Under response_only we keep the ones whose judgement
    # involves the assistant's reply.
    if response_only:
        tt = str(row.get("text_type", "")).strip().lower()
        if tt not in ("llm_response", "combined"):
            return []
    text = row.get("text") or row.get("prompt")
    if not text:
        return []
    votes = [row.get(f"labels_{i}") for i in range(5)]
    votes = [v for v in votes if v not in (None, "")]
    if votes:
        unsafe = sum(1 for v in votes if str(v).strip().lower() not in ("safe", "needs caution"))
        return [(text, int(unsafe > len(votes) / 2))]
    lab = row.get("prompt_label") or row.get("response_label") or row.get("label")
    if lab is None:
        return []
    return [(text, int(str(lab).strip().lower() in ("unsafe", "harmful", "1", "true")))]


def _wildguard(row, response_only: bool = False) -> List[Sample]:
    # Emits up to two samples: the prompt alone (prompt-level) and
    # prompt+response (response-level). response_only keeps just the latter.
    out: List[Sample] = []
    prompt = (row.get("prompt") or "").strip()
    if prompt and not response_only and row.get("prompt_harm_label") is not None:
        out.append((prompt, int(str(row["prompt_harm_label"]).strip().lower() == "harmful")))
    resp = (row.get("response") or "").strip()
    if resp and row.get("response_harm_label") is not None:
        text = f"{prompt}\n{resp}".strip()
        out.append((text, int(str(row["response_harm_label"]).strip().lower() == "harmful")))
    return out


def _saferlhf(row, response_only: bool = False) -> List[Sample]:
    # Always response-level: is_response_k_safe judges the reply.
    prompt = (row.get("prompt") or "").strip()
    out: List[Sample] = []
    for k in (0, 1):
        resp = row.get(f"response_{k}")
        safe = row.get(f"is_response_{k}_safe")
        if resp and safe is not None:
            out.append((f"{prompt}\n{resp}".strip(), int(not bool(safe))))
    return out


def _beavertails(row, response_only: bool = False) -> List[Sample]:
    # Always response-level: is_safe judges the reply given the prompt.
    prompt = (row.get("prompt") or "").strip()
    resp = (row.get("response") or "").strip()
    is_safe = row.get("is_safe")
    if not (prompt or resp) or is_safe is None:
        return []
    return [(f"{prompt}\n{resp}".strip(), int(not bool(is_safe)))]


@dataclass
class BenchmarkSpec:
    name: str
    hf_id: str
    config: Optional[str]
    split: str
    to_samples: Callable[..., List[Sample]]
    gated: bool = False


# Registry, in the paper's listed order. Ids/configs/splits/fields verified on
# the HF dataset viewer (Aug 2026); WildGuard is gated, NVIDIA Aegis may be.
PAPER_BENCHMARKS: List[BenchmarkSpec] = [
    BenchmarkSpec("ToxicChat", "lmsys/toxic-chat", "toxicchat0124", "train", _toxicchat),
    BenchmarkSpec("OpenAIModeration", "mmathys/openai-moderation-api-evaluation", "default", "train", _openai_moderation),
    BenchmarkSpec("Aegis", "nvidia/Aegis-AI-Content-Safety-Dataset-1.0", "default", "train", _aegis, gated=True),
    BenchmarkSpec("Aegis2.0", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "default", "train", _aegis, gated=True),
    BenchmarkSpec("WildGuard", "allenai/wildguardmix", "wildguardtrain", "train", _wildguard, gated=True),
    BenchmarkSpec("SafeRLHF", "PKU-Alignment/PKU-SafeRLHF", "default", "train", _saferlhf),
    BenchmarkSpec("BeaverTails", "PKU-Alignment/BeaverTails", "default", "30k_train", _beavertails),
]


@dataclass
class SafetyCorpus:
    train_texts: List[str]
    y_train: np.ndarray
    val_texts: List[str]
    y_val: np.ndarray
    test_texts: List[str]
    y_test: np.ndarray
    meta: Dict = field(default_factory=dict)


def _load_one(spec: BenchmarkSpec, cap: int, response_only: bool = False) -> List[Sample]:
    """Stream up to `cap` rows from one benchmark and map to (text, label)."""
    from datasets import load_dataset  # local import so `import siren` stays light

    ds = load_dataset(spec.hf_id, spec.config, split=spec.split, streaming=True)
    samples: List[Sample] = []
    for row in islice(ds, cap):
        for text, label in spec.to_samples(row, response_only):
            if isinstance(text, str):
                t = text.strip()
                if t:  # drop empty / whitespace-only texts (would 0-length crash)
                    samples.append((t, int(label)))
    return samples


def load_safety_benchmarks(
    cap_per_dataset: int = 2000,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    balance: bool = True,
    seed: int = 42,
    only: Optional[Sequence[str]] = None,
    response_only: bool = False,
    verbose: bool = True,
) -> SafetyCorpus:
    """
    Load and aggregate the paper's safety benchmarks into a binary corpus with a
    train/val/test split (stratified by a shuffled global split).

    Args:
        cap_per_dataset: max rows streamed per benchmark (keeps Colab runnable).
        val_frac, test_frac: split fractions.
        balance: downsample the majority class to a 1:1 ratio.
        only: restrict to a subset of benchmark names (default: all seven).
        response_only: keep only samples whose LABEL judges the assistant's
            reply, dropping prompt-level ones. The text is still the full
            prompt+response prefix -- this changes the QUESTION the classifier
            learns to answer, not what it gets to see.

            Why it matters: mixing prompt-level and response-level labels teaches
            contradictory things about the same prefix ("bomb question" -> 1 from
            ToxicChat, but "bomb question + refusal" -> 0 from BeaverTails). The
            blend makes the score track the PROMPT, so a refusal and a compliance
            to the same harmful prompt become indistinguishable (measured: 0.9999
            vs 0.9997) -- and that distinction is exactly what an early-stop
            decision needs. Training on response-level labels only lets a plain
            absolute threshold separate them.

            Kept: BeaverTails, SafeRLHF, WildGuard (response half), Aegis /
            Aegis2.0 (llm_response + combined). Dropped: ToxicChat,
            OpenAIModeration (prompt-level by construction).
        verbose: print per-benchmark load status and label balance.

    Resilient: any benchmark that errors (gated/unauthorized/schema change) is
    skipped with a message; the run continues on whatever loaded.
    """
    specs = PAPER_BENCHMARKS if only is None else [s for s in PAPER_BENCHMARKS if s.name in set(only)]
    all_samples: List[Sample] = []
    per_ds: Dict[str, Dict[str, int]] = {}

    for spec in specs:
        try:
            s = _load_one(spec, cap_per_dataset, response_only)
            if not s:
                if response_only:
                    if verbose:
                        print(f"  [skip] {spec.name:16s} no response-level samples "
                              f"(prompt-level dataset)")
                    per_ds[spec.name] = {"loaded": 0, "note": "no response-level rows"}
                    continue
                raise ValueError("no usable rows (schema mismatch?)")
            pos = sum(l for _, l in s)
            per_ds[spec.name] = {"loaded": len(s), "harmful": pos, "safe": len(s) - pos}
            all_samples.extend(s)
            if verbose:
                print(f"  [ok]   {spec.name:16s} {len(s):6d} rows  (harmful={pos}, safe={len(s)-pos})")
        except Exception as e:  # gated / offline / schema drift
            gate = " (GATED — run huggingface_hub.notebook_login and accept terms)" if spec.gated else ""
            per_ds[spec.name] = {"loaded": 0, "error": f"{type(e).__name__}: {str(e)[:120]}"}
            if verbose:
                print(f"  [skip] {spec.name:16s} {type(e).__name__}: {str(e)[:80]}{gate}")

    if not all_samples and response_only:
        raise RuntimeError(
            "No response-level samples. Every selected benchmark is prompt-level "
            "(ToxicChat / OpenAIModeration contribute nothing under response_only). "
            "Include at least one of BeaverTails, SafeRLHF, WildGuard, Aegis, Aegis2.0 "
            "-- or drop response_only."
        )
    if not all_samples:
        raise RuntimeError(
            "No benchmarks could be loaded. On Colab, run "
            "`from huggingface_hub import notebook_login; notebook_login()` and accept "
            "the gated datasets' terms, or pass only=[...] with the open ones "
            "(ToxicChat, OpenAIModeration, SafeRLHF, BeaverTails)."
        )

    rng = np.random.RandomState(seed)

    # De-duplicate and (optionally) class-balance.
    seen = set()
    uniq: List[Sample] = []
    for text, label in all_samples:
        key = (text, label)
        if key not in seen:
            seen.add(key)
            uniq.append((text, label))

    if balance:
        pos = [s for s in uniq if s[1] == 1]
        neg = [s for s in uniq if s[1] == 0]
        k = min(len(pos), len(neg))
        rng.shuffle(pos)
        rng.shuffle(neg)
        uniq = pos[:k] + neg[:k]

    rng.shuffle(uniq)
    texts = [t for t, _ in uniq]
    labels = np.array([l for _, l in uniq], dtype=np.int64)

    n = len(texts)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    test_texts, test_y = texts[:n_test], labels[:n_test]
    val_texts, val_y = texts[n_test:n_test + n_val], labels[n_test:n_test + n_val]
    train_texts, train_y = texts[n_test + n_val:], labels[n_test + n_val:]

    meta = {
        "response_only": response_only,
        "per_dataset": per_ds,
        "loaded_datasets": [k for k, v in per_ds.items() if v.get("loaded", 0) > 0],
        "total_after_balance": n,
        "label_convention": "1=harmful/unsafe, 0=safe",
    }
    if verbose:
        scope = "response-level labels only" if response_only else "prompt+response labels mixed"
        print(f"\nAggregated corpus [{scope}]: train={len(train_texts)} "
              f"val={len(val_texts)} test={len(test_texts)} (balanced={balance})")
        print(f"datasets loaded: {meta['loaded_datasets']}")

    return SafetyCorpus(train_texts, train_y, val_texts, val_y, test_texts, test_y, meta)


# --------------------------------------------------------------------------- #
# Paired (prompt, response) access                                             #
# --------------------------------------------------------------------------- #
# The corpus builders above concatenate prompt and response into one string,
# which is what training needs. Some analyses instead need the boundary: e.g.
# "score the prompt alone, then watch the score evolve across the response".
# Only datasets that genuinely separate the two are supported here.

@dataclass
class PairedSample:
    prompt: str
    response: str
    label: int          # response-level: 1 = harmful reply, 0 = safe reply
    source: str


def _paired_beavertails(row) -> List["PairedSample"]:
    p, r, safe = (row.get("prompt") or "").strip(), (row.get("response") or "").strip(), row.get("is_safe")
    if not (p and r) or safe is None:
        return []
    return [PairedSample(p, r, int(not bool(safe)), "BeaverTails")]


def _paired_saferlhf(row) -> List["PairedSample"]:
    p = (row.get("prompt") or "").strip()
    out = []
    for k in (0, 1):
        r, safe = row.get(f"response_{k}"), row.get(f"is_response_{k}_safe")
        if p and r and safe is not None:
            out.append(PairedSample(p, str(r).strip(), int(not bool(safe)), "SafeRLHF"))
    return out


def _paired_wildguard(row) -> List["PairedSample"]:
    p, r = (row.get("prompt") or "").strip(), (row.get("response") or "").strip()
    lab = row.get("response_harm_label")
    if not (p and r) or lab is None:
        return []
    return [PairedSample(p, r, int(str(lab).strip().lower() == "harmful"), "WildGuard")]


PAIRED_SOURCES = [
    ("BeaverTails", "PKU-Alignment/BeaverTails", "default", "30k_test", _paired_beavertails),
    ("SafeRLHF", "PKU-Alignment/PKU-SafeRLHF", "default", "test", _paired_saferlhf),
    ("WildGuard", "allenai/wildguardmix", "wildguardtest", "test", _paired_wildguard),
]


def load_paired_response_samples(
    cap_per_dataset: int = 1000,
    only: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> List["PairedSample"]:
    """
    Load (prompt, response, response-level label) triples from the datasets that
    keep the two apart. Uses the TEST splits, so these are held out from the
    corpora that train/validate the guards.
    """
    from datasets import load_dataset

    picked = [s for s in PAIRED_SOURCES if only is None or s[0] in set(only)]
    out: List[PairedSample] = []
    for name, hf_id, config, split, adapter in picked:
        try:
            ds = load_dataset(hf_id, config, split=split, streaming=True)
            got = []
            for row in islice(ds, cap_per_dataset):
                got.extend(adapter(row))
            out.extend(got)
            if verbose:
                pos = sum(s.label for s in got)
                print(f"  [ok]   {name:14s} {len(got):5d} pairs  (harmful reply={pos})")
        except Exception as e:
            if verbose:
                print(f"  [skip] {name:14s} {type(e).__name__}: {str(e)[:70]}")
    return out
