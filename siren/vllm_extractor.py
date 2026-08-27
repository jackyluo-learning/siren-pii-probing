"""
Layer-wise activation extraction through IBM's vLLM-Hook, as a drop-in
alternative to the HuggingFace forward-hook path in run_real_layerwise_experiment.

Why this exists: the HF path runs one padded batch at a time through
AutoModelForCausalLM and reads the residual stream with PyTorch hooks. vLLM-Hook
instead hooks vLLM's own decoder layers, so the same activations come out of a
serving-grade engine -- continuous batching, paged attention, no padding waste --
and the same hooks keep working during generation, which the HF path cannot do
without re-running the model per prefix.

What it does NOT change: `pool_layers` returns exactly what `pool_texts` returns,
{layer_index: (N, hidden)} with layer 1 nearest the input, so probing,
aggregation, the shuffled-label control and the lexical baseline are untouched.
That is deliberate -- the only way to trust a new extraction path is to hold
everything downstream fixed and compare the numbers.

Two correctness details this wraps up:

  * vLLM fuses the residual add into the next layer's norm, so its decoder blocks
    return (hidden_states, residual) with the add still pending. vLLM-Hook
    reconstructs `output[0] + output[1]` before storing, which is what makes its
    tensors comparable to HuggingFace's output_hidden_states. Verified by reading
    probe_hidden_states_worker.py, not assumed.
  * Layer numbering is 1-based in vLLM-Hook ("layer N = output after the Nth
    block"), matching the convention used throughout this project.

Pooling maps onto the plugin's own modes rather than being redone here:
  mean -> mode "all_tokens" + reduce "mean". vLLM stores the real sequence with
      no padding, so this is paper Eq. 2 exactly, with no mask arithmetic.
  last -> mode "last_token". The plugin takes hidden[end - 1] per request, which
      is the final real token; the right-padding trap in the HF path does not
      exist here because there is no padding.

Untested against a live vLLM at the time of writing -- this machine has no CUDA.
Run `parity_report` before trusting any result from this path.
"""

import json
import os
import tempfile
from typing import Dict, List, Optional, Sequence

import numpy as np

from .progress import track


def check_runtime() -> None:
    """
    Fail with one actionable line instead of a forty-line import traceback.

    Measured on Colab (Python 3.13, Tesla T4, a CUDA-12 package ecosystem):
    installing vLLM-Hook's requirement.txt resolves `vllm>=0.5,<=0.21` to 0.21.0,
    which pins torch==2.11.0 -- a CUDA 13 build -- and `import vllm._C` then dies
    on a missing libcudart.so.13. The failure surfaces deep inside vLLM's
    platform detection, where nothing points at the version pin that caused it.
    """
    try:
        import vllm  # noqa: F401
    except ImportError as exc:
        msg = str(exc)
        if "libcudart" in msg:
            want = msg.split("libcudart.so.")[-1].split(":")[0]
            raise RuntimeError(
                f"vLLM 装的是 CUDA {want} 构建，但运行时没有对应的 CUDA runtime。\n"
                f"  原始错误: {msg}\n"
                f"  路 A（不动 torch）: pip install nvidia-cuda-runtime-cu{want}\n"
                f"  路 B（换 cu12 版本，需重启运行时）: pip install 'vllm==0.11.0'\n"
                f"  注意 vLLM-Hook 的 requirement.txt 会解析到 vllm 0.21.0，"
                f"它钉死 torch==2.11.0（cu13）——不要用它装 vllm。") from exc
        raise RuntimeError(f"导入 vllm 失败：{msg}") from exc

    try:
        from vllm_hook_plugins import HookLLM  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"导入 vllm_hook_plugins 失败：{exc}\n"
            "  先跑: pip install -e /content/vLLM-Hook/vllm_hook_plugins") from exc

    # vLLM 0.11's get_cached_tokenizer reads tokenizer.all_special_tokens_extended,
    # which transformers removed in the v5 tokenizer refactor. vLLM declares only
    # transformers>=4.55.2, so pip happily leaves a v5 in place and the failure
    # lands inside HookLLM(...) -- after the 8 GB model download. Check the
    # capability itself rather than comparing version numbers, and check it here
    # so it costs a second instead of ten minutes.
    import transformers
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        import vllm
        raise RuntimeError(
            f"transformers {transformers.__version__} 移除了 all_special_tokens_extended，"
            f"而 vLLM {vllm.__version__} 的分词器缓存要读它。\n"
            "  修法: pip install 'transformers==4.57.6'   （4.x 最后一版，Qwen3 从 4.51 起支持）\n"
            "  若换用更新的 vLLM 已不再依赖该属性，可删掉这段检查。")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("没有可用 GPU，vLLM 无法运行。")
    cc = torch.cuda.get_device_capability(0)
    if cc[0] < 8:
        print(f"⚠️ GPU 算力 {cc[0]}.{cc[1]}（如 T4 = 7.5）：不支持 bfloat16，"
              "也没有 FlashAttention，vLLM 会回退到较慢的实现。本项目固定用 float16，可以跑。")


def _write_model_config(model: str, num_layers: int, mode: str, out_dir: str) -> str:
    """vLLM-Hook reads which layers to capture from a per-model JSON file."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model.split('/')[-1]}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "model_info": {"name": model},
            "hidden_states": {"layers": list(range(1, num_layers + 1)), "mode": mode},
        }, fh, indent=2)
    return path


def _layer_index(layer_name: str, meta_num: Optional[int]) -> int:
    """
    Recover the 1-based layer index from what the analyzer keys its dict by.

    The plugin exposes an explicit layer_num, so prefer it; the name parse is a
    fallback for shapes like 'model.layers.11' (0-based there, hence + 1).
    """
    if meta_num is not None:
        return int(meta_num)
    tail = [p for p in layer_name.replace("__", ".").split(".") if p.isdigit()]
    if not tail:
        raise ValueError(f"无法从层名 {layer_name!r} 解析层号")
    return int(tail[-1]) + 1


def suggest_chunk_size(num_layers: int, max_len: int, hidden: int, mode: str,
                       fraction: float = 0.25) -> int:
    """
    Size the chunk from free GPU memory instead of a fixed guess.

    The hook cache lives on the GPU until retrieval: in all_tokens mode it holds
    chunk x layers x seq_len x hidden fp16 values on top of the weights and KV
    cache vLLM has already reserved. A fixed chunk of 32 was calibrated for 128
    tokens and then OOM'd at 512 -- it asked for 3.36 GiB with 1.96 GiB free --
    because the cost scales with max_model_len, which the caller sets separately.
    So compute it, and only spend `fraction` of what is free.
    """
    import torch
    per_text = num_layers * hidden * 2
    if mode == "all_tokens":
        per_text *= max_len
    free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 2 << 30
    n = max(1, int(free * fraction // max(per_text, 1)))
    print(f"  hook 缓存约 {per_text/2**20:.1f} MB/条，空闲显存 {free/2**30:.1f} GiB"
          f" → 块大小取 {n}", flush=True)
    return n


def build_hook_llm(model: str, num_layers: int, pooling: str = "mean",
                   max_model_len: int = 128, gpu_memory_utilization: float = 0.62,
                   hook_dir: str = "/dev/shm/vllm_hook", cache_dir: str = "./cache/"):
    """
    Construct a HookLLM configured to capture every layer.

    max_model_len defaults to 128 to match the HuggingFace path's max_length. That
    is a correctness requirement, not a memory one: truncating at different points
    feeds the two paths different token sequences, so parity would fail even with
    the extraction itself perfectly right.

    gpu_memory_utilization is deliberately below vLLM's usual 0.9. Whatever vLLM
    reserves is unavailable to the hook cache, which needs GPU room of its own --
    on a 14.56 GiB T4, 0.75 left too little and the capture OOM'd.
    """
    check_runtime()
    import torch
    from vllm_hook_plugins import HookLLM

    mode = "all_tokens" if pooling == "mean" else "last_token"
    cfg = _write_model_config(model, num_layers, mode,
                              tempfile.mkdtemp(prefix="siren_hookcfg_"))
    llm = HookLLM(
        model=model,
        worker_name="probe_hidden_states",
        analyzer_name="hidden_states",
        config_file=cfg,
        download_dir=cache_dir,
        hook_dir=hook_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype=torch.float16,
        enable_prefix_caching=False,   # prefix reuse would skip the prefill we need
        enable_hook=True,
        tensor_parallel_size=1,
    )
    return llm, mode


def pool_layers(llm, texts: Sequence[str], num_layers: int, mode: str,
                chunk_size: int = 0, show_progress: bool = True,
                max_len: int = 128, hidden: int = 0) -> Dict[int, np.ndarray]:
    """
    Pool every layer for every text. Returns {layer: (N, hidden)}, order preserved.

    Chunking is not about throughput -- vLLM batches internally -- but about the
    hook cache. In all_tokens mode each text holds num_layers x seq_len x hidden
    values before reduction: at 36 layers, 128 tokens and 2560 dims that is ~24 MB
    of fp16 per text, so a whole split at once would be tens of gigabytes.
    """
    from vllm import SamplingParams

    if chunk_size <= 0:
        if not hidden:
            hidden = int(llm.llm_engine.vllm_config.model_config.get_hidden_size())
        chunk_size = suggest_chunk_size(num_layers, max_len, hidden, mode)

    texts = list(texts)
    per_layer: Dict[int, List[np.ndarray]] = {l: [] for l in range(1, num_layers + 1)}
    # One token is the minimum vLLM will generate; the prefill is what we want and
    # the single decode step is negligible beside it.
    params = SamplingParams(temperature=0.0, max_tokens=1)
    reduce = "mean" if mode == "all_tokens" else "none"

    starts = range(0, len(texts), chunk_size)
    for start in (track(starts, desc="vLLM 抽取", unit="块") if show_progress else starts):
        batch = texts[start:start + chunk_size]
        llm.generate(batch, params, save_to_disk=True)
        stats = llm.analyze(analyzer_spec={"reduce": reduce})
        hs = stats["hidden_states"]
        if not hs:
            raise RuntimeError("vLLM-Hook 没有返回任何隐藏状态；"
                               "检查 config_file 里的 layers 与 enable_hook。")
        seen = set()
        for layer_name, tensors in hs.items():
            idx = _layer_index(layer_name, getattr(tensors, "layer_num", None))
            if idx not in per_layer:
                continue
            if len(tensors) != len(batch):
                raise RuntimeError(
                    f"层 {idx} 返回 {len(tensors)} 条，与本块的 {len(batch)} 条不符——"
                    "特征与标签会错位，中止。")
            seen.add(idx)
            per_layer[idx].extend(
                np.asarray(t.float().cpu().numpy(), dtype=np.float32) for t in tensors)
        missing = set(per_layer) - seen
        if missing:
            raise RuntimeError(f"这些层没有被捕获：{sorted(missing)[:8]}…"
                               "（配置里的 layers 与模型层数不一致？）")
        llm.llm_engine.reset_prefix_cache()

    return {l: np.stack(v) for l, v in per_layer.items()}


def parity_report(model: str, texts: Sequence[str], num_layers: int,
                  pooling: str = "mean", max_length: int = 128,
                  device: str = "cuda", **hook_kwargs) -> Dict:
    """
    Extract the same texts both ways and report how far apart they land.

    This is the gate, not a nicety. Two engines can disagree on dtype, on where a
    layer's output is read, on whether the residual was added, and on how padding
    is masked -- all of which change the numbers without raising anything. Run it
    before reading any score produced through the vLLM path.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples"))
    from run_real_layerwise_experiment import load_model_and_extractor, pool_texts

    # The two engines cannot share the GPU: Qwen3-4B in fp16 is ~7.5 GiB, and the
    # first version of this function left the HuggingFace copy resident, so vLLM
    # started with 5.9 of 14.56 GiB free and refused to allocate. remove_hooks()
    # detaches the hooks but frees nothing, so the model has to be dropped and the
    # allocator's cache released before the second engine is built.
    import gc
    import torch

    tok, hf_model, ex, hf_layers = load_model_and_extractor(model, device)
    hf = pool_texts(tok, ex, texts, hf_layers, max_length, pooling=pooling,
                    show_progress=False)
    ex.remove_hooks()
    del ex, hf_model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        print(f"  已释放 HuggingFace 模型，空闲显存回到 {free:.1f} GiB", flush=True)

    hook_kwargs.setdefault("max_model_len", max_length)   # 必须与 HF 的截断一致
    llm, mode = build_hook_llm(model, num_layers, pooling, **hook_kwargs)
    vl = pool_layers(llm, texts, num_layers, mode, show_progress=False,
                     max_len=max_length)

    rows = []
    for l in sorted(set(hf) & set(vl)):
        a, b = hf[l].astype(np.float64), vl[l].astype(np.float64)
        cos = float(np.mean(np.sum(a * b, 1) /
                            (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))
        rel = float(np.mean(np.linalg.norm(a - b, axis=1) /
                            (np.linalg.norm(a, axis=1) + 1e-12)))
        rows.append({"layer": l, "cos": cos, "rel_diff": rel})
    return {"n_texts": len(texts), "pooling": pooling, "rows": rows,
            "worst_cos": min(r["cos"] for r in rows) if rows else None,
            "worst_rel": max(r["rel_diff"] for r in rows) if rows else None}
