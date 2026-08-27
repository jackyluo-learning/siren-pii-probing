"""
Regression tests for the streaming (generation-time) detection path.
"""
import numpy as np
import torch
import pytest

from siren.aggregator import AdaptiveNeuronAggregator
from siren.classifier import SirenMLPHead
from siren.streaming import StreamingModerator


class _StubExtractor:
    """Returns canned prefix-pooled states, so no LLM download is needed."""
    def __init__(self, states):
        self._states = states

    def extract_all_prefix_pooled(self, input_ids):
        return self._states


def test_prefix_mean_is_cumulative_mean():
    # The efficiency trick: prefix mean == cumsum / t along the sequence axis.
    T, D = 7, 5
    hs = torch.randn(T, D)
    cum = torch.cumsum(hs, dim=0) / torch.arange(1, T + 1).unsqueeze(1).float()
    for t in range(1, T + 1):
        assert torch.allclose(cum[t - 1], hs[:t].mean(dim=0), atol=1e-6)


def test_score_token_stream_returns_one_score_per_token():
    T, D = 11, 8
    layers = {1: torch.randn(T, D), 2: torch.randn(T, D)}
    agg = AdaptiveNeuronAggregator(
        safety_neurons={1: [0, 1, 2], 2: [3, 4]},
        layer_f1_scores={1: 0.9, 2: 0.6},
    )
    clf = SirenMLPHead(input_dim=agg.total_feature_dim, hidden_dim=16)
    mod = StreamingModerator(_StubExtractor(layers), agg, clf, threshold=0.5, device="cpu")

    scores = mod.score_token_stream(torch.zeros(1, T, dtype=torch.long))
    assert scores.shape == (T,)                       # one score per prefix
    assert np.all((scores >= 0.0) & (scores <= 1.0))  # probabilities


def test_first_flag_respects_warmup_and_detection_curve_is_monotonic():
    from run_streaming_evaluation import first_flag_positions, detection_curve

    # Crosses at position 2 (inside warm-up) and again at 10.
    s = np.zeros(12)
    s[1] = 0.99
    s[9:] = 0.99
    # No warm-up -> flags at 2; warm-up 8 -> the early spike is ignored, flags at 10.
    assert first_flag_positions([s], 0.5, min_prefix=1)[0] == 2
    assert first_flag_positions([s], 0.5, min_prefix=8)[0] == 10

    # Never crossing -> inf, and a sequence shorter than the warm-up never flags.
    assert np.isinf(first_flag_positions([np.zeros(12)], 0.5, 8)[0])
    assert np.isinf(first_flag_positions([np.ones(3)], 0.5, 8)[0])

    curve = detection_curve(np.array([3.0, 10.0, np.inf]), max_pos=12)
    assert np.all(np.diff(curve) >= 0)          # cumulative -> non-decreasing
    assert curve[2] == pytest.approx(1 / 3)     # only the t=3 one flagged by pos 3
    assert curve[-1] == pytest.approx(2 / 3)    # the never-flagged one stays out


def test_balanced_subsample_keeps_every_class():
    """Regression: the binary-only version dropped classes 2..K-1 entirely."""
    import numpy as np
    from run_real_layerwise_experiment import _balanced_subsample

    y = np.repeat(np.arange(6), 500)          # 6 classes x 500
    sel = _balanced_subsample(y, cap=1200)
    counts = np.bincount(y[sel], minlength=6)
    assert len(sel) <= 1200
    assert (counts == 200).all(), counts       # cap // 6 from each class

    # Binary behaviour is unchanged.
    yb = np.repeat([0, 1], 500)
    cb = np.bincount(yb[_balanced_subsample(yb, cap=400)], minlength=2)
    assert (cb == 200).all(), cb

    # A class thinner than the quota contributes everything it has.
    ys = np.concatenate([np.zeros(500), np.ones(30), np.full(500, 2)]).astype(int)
    cs = np.bincount(ys[_balanced_subsample(ys, cap=600)], minlength=3)
    assert cs.tolist() == [200, 30, 200], cs


def test_track_emits_flushed_lines_when_stdout_is_not_a_terminal():
    """Colab pipes `!python` output; tqdm's \\r frames vanish, so we need lines."""
    import io
    from siren.progress import track

    buf = io.StringIO()                       # StringIO.isatty() is False
    assert list(track(range(50), "阶段", unit="步", stream=buf)) == list(range(50))
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]

    assert len(lines) > 3, "非终端时必须有多行进度，否则看起来像卡死"
    assert len(lines) <= 25, f"行数应被节流，实际 {len(lines)}"
    assert lines[0].startswith("阶段: 开始")
    assert "50/50 (100%)" in lines[-1] and "完成" in lines[-1]
    assert "\r" not in buf.getvalue(), "回车重画会被 Colab 吞掉"


def test_track_handles_unsized_iterables():
    import io
    from siren.progress import track

    buf = io.StringIO()
    assert list(track((x for x in range(5)), "生成器", stream=buf)) == list(range(5))
    assert buf.getvalue().strip(), "无 total 时也要有输出"


def test_track_defaults_to_lines_even_on_a_terminal():
    """
    Colab runs `!python` under a pty, so isatty() is True while the frontend
    still shows only the last \\r frame. Keying on isatty() therefore picked the
    bar and fixed nothing; lines must be the default regardless.
    """
    import io
    import os
    from unittest import mock
    from siren.progress import track

    class _FakeTTY(io.StringIO):
        def isatty(self):
            return True

    buf = _FakeTTY()
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SIREN_PROGRESS", None)
        list(track(range(30), "阶段", unit="步", stream=buf))
    assert "\r" not in buf.getvalue(), "伪终端下仍必须走行模式"
    assert "30/30 (100%)" in buf.getvalue()


def test_heartbeat_reports_liveness_when_items_are_slow():
    """
    Progress can only be emitted between items. The shuffled-label control fits
    one layer roughly every 15s, so the stage announced itself and then went
    quiet long enough to look stuck; a heartbeat has to cover that gap.
    """
    import io
    import time
    from siren.progress import _LineProgress

    buf = io.StringIO()
    bar = _LineProgress(total=2, desc="慢阶段", unit="层", heartbeat=0.2,
                        note="预计约 1s", stream=buf)
    time.sleep(0.75)                      # no item finishes in this window
    bar.update()
    bar.close()

    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    assert "预计约 1s" in lines[0], "开跑前要说明预计耗时"
    beats = [l for l in lines if "运行中…" in l]
    assert beats, "长时间无输出时必须报活"
    assert any("已用" in b for b in beats)


def test_last_token_pooling_indexes_by_mask_not_by_position():
    """
    The tokenizer pads on the right, so hidden_state[:, -1] is a pad token for
    every sequence shorter than the batch maximum. Indexing by position instead
    of by the attention mask silently pools padding -- no error, just wrong
    vectors -- so this pins the mask lookup.
    """
    import torch
    from siren.extractor import InternalStateExtractor

    ex = object.__new__(InternalStateExtractor)          # skip __init__/model load
    ex.device = "cpu"
    ex.captured_states = {}

    B, T, D = 3, 5, 4
    # Distinct value per position so the wrong pick is unmistakable.
    hidden = torch.arange(B * T * D, dtype=torch.float32).reshape(B, T, D)
    mask = torch.tensor([[1, 1, 1, 0, 0],       # real length 3 -> index 2
                         [1, 1, 0, 0, 0],       # real length 2 -> index 1
                         [1, 1, 1, 1, 1]])      # real length 5 -> index 4

    def fake_forward(**kw):
        ex.captured_states = {1: hidden}
        return None
    ex.model = fake_forward

    out = ex.extract_sequence_pooled(torch.zeros(B, T, dtype=torch.long),
                                     mask, pooling="last")[1]
    expected = torch.stack([hidden[0, 2], hidden[1, 1], hidden[2, 4]])
    assert torch.allclose(out, expected), f"取到了错误位置:\n{out}\n应为\n{expected}"

    # Mean pooling must ignore the padded tail too, and differ from last-token.
    mean = ex.extract_sequence_pooled(torch.zeros(B, T, dtype=torch.long),
                                      mask, pooling="mean")[1]
    assert torch.allclose(mean[1], hidden[1, :2].mean(dim=0))
    assert not torch.allclose(mean, out)

    import pytest
    with pytest.raises(ValueError, match="pooling"):
        ex.extract_sequence_pooled(torch.zeros(B, T, dtype=torch.long), mask,
                                   pooling="cls")


def _fake_qwen_block():
    """A block with Qwen/Llama's child names, enough to resolve hook targets."""
    import torch.nn as nn

    blk = nn.Module()
    blk.self_attn = nn.Linear(4, 4)
    blk.mlp = nn.Linear(4, 4)
    blk.input_layernorm = nn.LayerNorm(4)
    blk.post_attention_layernorm = nn.LayerNorm(4)
    return blk


def test_extraction_point_selects_the_intended_submodule():
    """
    Each tap point is resolved by a candidate name list, because families disagree
    on naming. Picking the wrong submodule produces plausible-looking activations
    with no error at all, so the mapping is pinned here.
    """
    import pytest
    import torch.nn as nn
    from siren.extractor import InternalStateExtractor

    blk = _fake_qwen_block()
    ex = object.__new__(InternalStateExtractor)
    ex.hooks = []
    ex.captured_states = {}
    ex.target_layers = [(1, blk)]

    for point, expected in (("residual", blk),
                            ("ffn", blk.mlp),
                            ("post_ln", blk.post_attention_layernorm)):
        ex.extraction_point = point
        ex._register_hooks()
        # A forward hook registers on the module it targets.
        hooked = {id(m) for m in (blk, blk.mlp, blk.post_attention_layernorm,
                                  blk.self_attn)
                  if m._forward_hooks}
        assert hooked == {id(expected)}, f"{point} 挂到了错误的子模块"
        ex.remove_hooks()

    # A block missing the target must fail loudly, naming what it does have.
    bare = nn.Module()
    bare.attention = nn.Linear(4, 4)
    ex2 = object.__new__(InternalStateExtractor)
    ex2.hooks = []
    ex2.captured_states = {}
    ex2.target_layers = [(1, bare)]
    ex2.extraction_point = "post_ln"
    with pytest.raises(ValueError, match="post_ln"):
        ex2._register_hooks()

    with pytest.raises(ValueError, match="extraction_point"):
        InternalStateExtractor.__init__(
            object.__new__(InternalStateExtractor), model=nn.Module(),
            extraction_point="attention_out")
