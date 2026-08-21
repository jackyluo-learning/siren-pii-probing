"""
Preflight: which of the paper's seven safety benchmarks can this environment
actually load, and how many samples survive under --response-only?

Reports, per benchmark, the number of samples produced in each labelling mode,
so you can see before a 30-minute run whether the corpus is complete and whether
the response-level guard (module 4b / 6) will have enough data.

Run:  PYTHONPATH=.:examples python -u examples/check_datasets.py
"""

import argparse
import warnings

from siren.safety_benchmarks import PAPER_BENCHMARKS, _load_one

warnings.filterwarnings("ignore")

OK, NO, DASH = "✅", "❌", "—"


def hf_login_status():
    """Return (logged_in, description)."""
    try:
        from huggingface_hub import whoami
        info = whoami()
        return True, info.get("name") or info.get("fullname") or "unknown"
    except Exception as e:
        return False, type(e).__name__


def classify(exc) -> str:
    """Why did this dataset fail? The remediation differs per cause."""
    name, msg = type(exc).__name__, str(exc).lower()
    if "offline" in msg or "connection" in msg or name in ("ConnectionError",):
        return "network"
    if ("gated" in msg or "401" in msg or "restricted" in msg or "authenticated" in msg
            or name in ("GatedRepoError",)):
        return "gated"
    return "other"


REASON = {
    "network": "网络不通 / 离线模式",
    "gated":   "需登录并接受条款",
    "other":   "加载失败（schema 或其它）",
}


def probe(spec, cap, response_only):
    """Try to pull a few rows. Returns (count, (kind, detail) or None)."""
    try:
        return len(_load_one(spec, cap, response_only)), None
    except Exception as e:
        return 0, (classify(e), f"{type(e).__name__}: {str(e)[:60]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200,
                    help="rows streamed per benchmark for the probe (small = fast)")
    args = ap.parse_args()

    print("=" * 78)
    print("SIREN 数据集预检 — 七个安全基准的可用性与标签口径")
    print("=" * 78)

    logged_in, who = hf_login_status()
    if logged_in:
        print(f"\nHugging Face 登录: {OK} 已登录（{who}）")
    else:
        print(f"\nHugging Face 登录: {NO} 未登录（{who}）")
        print("   gated 数据集会被跳过。在 Colab 里执行下面一行后重跑本检查：")
        print("   from huggingface_hub import notebook_login; notebook_login()")

    print(f"\n逐个基准探测（每个流式读取 {args.rows} 行）...\n")
    header = f"  {'基准':<18}{'默认(混合)':>12}{'response_only':>16}   {'说明'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    mixed_ok, resp_ok, blocked = [], [], []
    for spec in PAPER_BENCHMARKS:
        n_mixed, err_m = probe(spec, args.rows, False)
        if err_m:
            kind, detail = err_m
            blocked.append((spec.name, kind, detail))
            print(f"  {spec.name:<18}{NO+' 不可用':>12}{DASH:>16}   {REASON[kind]}")
            continue

        n_resp, err_r = probe(spec, args.rows, True)
        mixed_ok.append(spec.name)
        if n_resp > 0:
            resp_ok.append(spec.name)
            note = "prompt+response，标签判回复"
            print(f"  {spec.name:<18}{n_mixed:>12}{n_resp:>16}   {note}")
        else:
            note = "纯 prompt 级，response_only 下不贡献"
            print(f"  {spec.name:<18}{n_mixed:>12}{'0':>16}   {note}")

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)

    total = len(PAPER_BENCHMARKS)
    print(f"\n  模块 4 / 5（复现，混合标签）：{len(mixed_ok)}/{total} 个基准可用")
    print(f"    → {mixed_ok}")
    if len(mixed_ok) < total:
        print(f"    {NO} 缺 {[n for n, _, _ in blocked]}，与论文的七基准设定有差距，")
        print(f"       Figure 7 仍能跑出形状，但数值不完全可比。")
    else:
        print(f"    {OK} 七个齐全，与论文设定一致。")

    print(f"\n  模块 4b / 6（拦截，response 级标签）：{len(resp_ok)} 个基准可贡献")
    print(f"    → {resp_ok}")
    if not resp_ok:
        print(f"    {NO} 没有任何 response 级语料，模块 4b 无法训练。")
    elif len(resp_ok) < 3:
        print(f"    ⚠️  语料偏少。建议登录以拉入 WildGuard，让「有害提问+拒答=安全」")
        print(f"       这类样本更充分，绝对阈值才分得开拒答与顺从。")
    else:
        print(f"    {OK} 语料充足，可以训练 response 级守卫。")

    if blocked:
        print("\n  被阻挡的数据集及原因：")
        for name, kind, detail in blocked:
            print(f"    · {name:<18}[{REASON[kind]}] {detail}")

        gated = [n for n, k, _ in blocked if k == "gated"]
        network = [n for n, k, _ in blocked if k == "network"]
        other = [n for n, k, _ in blocked if k == "other"]

        if gated:
            print(f"\n  {len(gated)} 个因权限被挡 —— 处理方式：")
            print("    1) from huggingface_hub import notebook_login; notebook_login()")
            print("    2) 逐个打开页面点 Agree 接受条款：")
            for name in gated:
                spec = next(s for s in PAPER_BENCHMARKS if s.name == name)
                print(f"       https://huggingface.co/datasets/{spec.hf_id}")
            print("    3) 重跑本检查确认变为可用")
        if network:
            print(f"\n  {len(network)} 个因网络被挡（{network}）——")
            print("    与登录无关。检查网络，或取消 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 后重试。")
        if other:
            print(f"\n  {len(other)} 个因其它原因失败（{other}）——")
            print("    可能是数据集 schema 变更，需要更新 siren/safety_benchmarks.py 的适配器。")

    print()


if __name__ == "__main__":
    main()
