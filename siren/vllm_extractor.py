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
import shutil
import tempfile
from typing import Dict, List, Optional, Sequence

import numpy as np


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
                       gpu_mem_util: float = 0.62) -> int:
    """
    Size the first chunk from what is left OUTSIDE vLLM's reservation.

    Not from torch.cuda.mem_get_info(): the hook cache is allocated in the engine
    subprocess, and the parent sees a different picture entirely. Measured on a
    T4, the parent reported 5.3 GiB free and chose 60, while the engine process
    had already given vLLM everything but 0.83 GiB of KV cache and died asking
    for 2.17 GiB.

    So budget analytically -- total x (1 - gpu_mem_util) -- and take a small
    slice of it, because the save path pads and stacks the whole chunk, roughly
    doubling peak usage. This is a starting point, not a guarantee: pool_layers
    halves it and retries on OOM.
    """
    import torch
    per_text = num_layers * hidden * 2
    if mode == "all_tokens":
        per_text *= max_len
    total = (torch.cuda.get_device_properties(0).total_memory
             if torch.cuda.is_available() else 8 << 30)
    outside = total * (1.0 - gpu_mem_util)
    # 0.15 was still fragmenting: the pad+stack buffer is a second copy of
    # the same size, so a chunk costs about twice what this arithmetic says.
    n = max(1, int(outside * 0.08 // max(per_text, 1)))
    # Cap it. The chunk exists to bound the hook cache, not to raise throughput --
    # vLLM batches internally either way -- and an enormous chunk only means a
    # retry after OOM throws away more work, and pushes the engine into scheduler
    # edge cases that a modest one never reaches.
    n = min(n, 256)
    print(f"  hook 缓存约 {per_text/2**20:.1f} MB/条；vLLM 预留之外约 "
          f"{outside/2**30:.1f} GiB → 起始块大小 {n}（OOM 会自动减半重试）", flush=True)
    return n


def build_hook_llm(model: str, num_layers: int, pooling: str = "mean",
                   max_model_len: int = 128, gpu_memory_utilization: float = 0.62,
                   hook_dir: str = "/dev/shm/vllm_hook", cache_dir: str = "./cache/",
                   max_num_seqs: int = 32):
    """
    Construct a HookLLM configured to capture every layer.

    max_model_len is the PROMPT truncation length, matching the HuggingFace path's
    max_length. That is a correctness requirement, not a memory one: truncating at
    different points feeds the two paths different token sequences, so parity
    would fail even with the extraction itself perfectly right.

    vLLM is given a slightly larger window, because its limit covers prompt plus
    generated tokens. Passing the two as equal asserted "Sampled token IDs exceed
    the max model length. Total number of tokens: 129 > max_model_len: 128" on the
    first text that filled the window exactly -- 9 of 6000 did. Parity had missed
    it because none of its 16 texts reached the cap.

    gpu_memory_utilization is deliberately below vLLM's usual 0.9. Whatever vLLM
    reserves is unavailable to the hook cache, which needs GPU room of its own --
    on a 14.56 GiB T4, 0.75 left too little and the capture OOM'd.
    """
    check_runtime()
    import torch
    from vllm_hook_plugins import HookLLM

    # Show the memory arithmetic before the engine spends minutes proving it
    # wrong. On a 14.56 GiB T4 with Qwen3-4B this is where it becomes visible
    # that all_tokens capture over 36 layers has almost nothing left to live in.
    # Each chunk allocates a transient pad+stack buffer of roughly
    # chunk x layers x seq_len x hidden before the copy to CPU, then frees it.
    # Repeating that beside vLLM's fixed pool fragments the free space until even
    # a 2 MiB request fails -- which is how the run died at 40%, not from any one
    # allocation being too large. Expandable segments is the allocator mode meant
    # for exactly this pattern; it is inherited by the engine subprocess.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    reserved = total * gpu_memory_utilization
    print(f"  显存账目: 总 {total:.1f} GiB → vLLM 预留 {reserved:.1f} GiB"
          f"（权重+KV）, 余下 {total - reserved:.1f} GiB 供 hook 缓存", flush=True)

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
        # +4 covers the single sampled token with slack; prompts are still cut
        # at max_model_len, so what the probes see is unchanged.
        max_model_len=max_model_len + 4,
        trust_remote_code=True,
        dtype=torch.float16,
        enable_prefix_caching=False,   # prefix reuse would skip the prefill we need
        enable_hook=True,
        tensor_parallel_size=1,
        # Bound concurrency explicitly. With last_token pooling a chunk costs
        # almost nothing in hook memory, so the auto-sizer offered 3902 texts at
        # once; the scheduler then ran 175 sequences concurrently and vLLM 0.11
        # died in its sampled-token copy with "The size of tensor a (132) must
        # match the size of tensor b (175)" -- 132 being max_model_len + 4, the
        # short window this project uses to match the HuggingFace truncation.
        # Concurrency buys nothing here anyway: every request is one prefill plus
        # one token, and vLLM batches internally regardless of chunk size.
        max_num_seqs=max_num_seqs,
    )
    llm._siren_hook_dir = hook_dir     # pool_layers 需要它来删除每块的产物
    return llm, mode


def pool_layers(llm, texts: Sequence[str], num_layers: int, mode: str,
                chunk_size: int = 0, show_progress: bool = True,
                max_len: int = 128, hidden: int = 0,
                tokenizer=None) -> Dict[int, np.ndarray]:
    """
    Pool every layer for every text. Returns {layer: (N, hidden)}, order preserved.

    Two things here exist because of how the first runs failed.

    Truncation is done here, not by either engine. HuggingFace's tokenizer
    silently truncates at max_length; vLLM refuses -- "The decoder prompt (length
    147) is longer than the maximum model length of 128" -- and kills the engine.
    Tokenising once and passing prompt_token_ids also removes the last way the
    two paths could see different inputs, which is what parity is actually
    testing.

    Chunk size backs off on OOM instead of trusting an estimate. Two estimates
    have already been wrong: one measured the parent process rather than the
    engine, the other ignored that saving pads and stacks the chunk.
    """
    import torch
    from vllm import SamplingParams
    try:                                    # 位置随 vLLM 版本变动过
        from vllm.inputs import TokensPrompt
    except ImportError:
        from vllm import TokensPrompt

    if tokenizer is None:
        from transformers import AutoTokenizer
        name = llm.llm_engine.vllm_config.model_config.tokenizer
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if not hidden:
        hidden = int(llm.llm_engine.vllm_config.model_config.get_hidden_size())
    if chunk_size <= 0:
        chunk_size = suggest_chunk_size(num_layers, max_len, hidden, mode)

    texts = list(texts)
    enc = tokenizer(texts, truncation=True, max_length=max_len)["input_ids"]
    enc = [ids if ids else [tokenizer.eos_token_id or 0] for ids in enc]
    over = sum(1 for t in tokenizer(texts)["input_ids"] if len(t) > max_len)
    if over:
        print(f"  {over}/{len(texts)} 条超过 {max_len} 词元，已按与 HF 路径相同的方式截断",
              flush=True)

    per_layer: Dict[int, List[np.ndarray]] = {l: [] for l in range(1, num_layers + 1)}
    params = SamplingParams(temperature=0.0, max_tokens=1)
    reduce = "mean" if mode == "all_tokens" else "none"

    def run_chunk(ids_batch):
        llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids_batch],
                     params, save_to_disk=True)
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
            if len(tensors) != len(ids_batch):
                raise RuntimeError(
                    f"层 {idx} 返回 {len(tensors)} 条，与本块的 {len(ids_batch)} 条不符——"
                    "特征与标签会错位，中止。")
            seen.add(idx)
            per_layer[idx].extend(
                np.asarray(t.float().cpu().numpy(), dtype=np.float32) for t in tensors)
        missing = set(per_layer) - seen
        if missing:
            raise RuntimeError(f"这些层没有被捕获：{sorted(missing)[:8]}…"
                               "（配置里的 layers 与模型层数不一致？）")
        # Artifacts are written per run_id and never cleaned up by the plugin,
        # while hook_dir defaults to /dev/shm -- RAM. In all_tokens mode a chunk
        # is over a gigabyte, so leaving them behind fills memory as surely as any
        # GPU leak. Delete each one as soon as it has been read.
        run_id = getattr(llm, "_last_run_id", None)
        hook_dir = getattr(llm, "_siren_hook_dir", None)
        if run_id and hook_dir:
            shutil.rmtree(os.path.join(hook_dir, run_id), ignore_errors=True)
        llm.llm_engine.reset_prefix_cache()

    # Not track(): the chunk size changes underfoot when a chunk is retried, so a
    # bar built from a fixed range would count wrong. Plain lines, same throttle.
    i, last_report, step = 0, 0, max(1, len(enc) // 20)
    if show_progress:
        print(f"vLLM 抽取: 开始，共 {len(enc)} 条", flush=True)
    while i < len(enc):
        n = min(chunk_size, len(enc) - i)
        try:
            run_chunk(enc[i:i + n])
        except torch.OutOfMemoryError:
            if chunk_size == 1:
                raise RuntimeError(
                    "块大小已降到 1 仍然显存不足。这张卡放不下"
                    f"「{num_layers} 层 × {max_len} 词元」的 hook 缓存加上模型权重。\n"
                    "  可选：--pooling last（每条只存一个向量，内存需求降两个数量级）、"
                    "换更小的模型、或换显存更大的 GPU。")
            chunk_size = max(1, chunk_size // 2)
            torch.cuda.empty_cache()
            print(f"  显存不足，块大小减半到 {chunk_size} 后重试", flush=True)
            for l in per_layer:                      # 丢掉本块可能写入的半截数据
                del per_layer[l][i:]
            continue
        i += n
        if show_progress and (i - last_report >= step or i == len(enc)):
            last_report = i
            print(f"vLLM 抽取: {i}/{len(enc)} ({100*i/len(enc):.0f}%)", flush=True)

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
    # One length governs both sides. Letting the caller also pass max_model_len
    # through hook_kwargs made them silently divergent: the runner set it while
    # max_length kept its own default, so any value other than 128 would have
    # truncated the two paths differently and reported a mismatch that was the
    # harness's fault, not the extraction's -- the worst possible false alarm from
    # a gate whose whole job is to catch exactly that.
    if "max_model_len" in hook_kwargs:
        raise TypeError("parity_report 只接受 max_length；max_model_len 由它推导，"
                        "分开传会让两条路径的截断点不一致。")
    hook_kwargs["max_model_len"] = max_length

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
