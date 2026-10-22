"""对第一级缓存（baselines/gen_cache.py 的输出）做离线评测，不调用 LLM。

逐候选、逐阶段（raw / compiled）跑隐藏测试 + 航天安全 MISRA 检查，结果写回
<pool_dir>/_eval.json（已评过的候选按代码哈希复用，不重复跑 docker）。

题目级指标（每题 N 份候选）：
  pass@N / safe-pass@N ：至少一份候选通过（且安全）——候选池上限
  E[pass@1] / E[safe-pass@1]：随机挑一份的期望 = 通过候选占比
候选级指标：compile / pass / safe / safe-pass 比例。

用法：python3 eval/eval_cache.py --dataset dataset/gnc-adcs-100.jsonl \
          --pool-dir logs/cache/deepseek-v4-flash/gnc100/pool1 [--pool-dir ...]
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "utils"))
from eval import load_jsonl, run_case  # noqa: E402
from record import safe_filename  # noqa: E402

STAGES = ("raw", "compiled", "misra")  # misra：第二级缓存（misra_cache.py），没有时自动跳过
PRIORITY = {"compiled": 0, "misra": 1, "raw": 2}


def evaluate_pool(refs, pool_dir, workers):
    cache_path = os.path.join(pool_dir, "_eval.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    jobs = []
    entries = {}
    for tid, ref in refs.items():
        p = os.path.join(pool_dir, safe_filename(tid) + ".json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        entries[tid] = d
        for c in d["candidates"]:
            for st in STAGES:
                if st not in c:
                    continue
                key = hashlib.sha1((tid + "\0" + c[st]).encode("utf-8")).hexdigest()
                if key not in cache:
                    jobs.append((key, tid, c[st], PRIORITY[st]))

    def run(job):
        key, tid, code = job
        with tempfile.TemporaryDirectory(prefix="ec_") as wd:
            r = run_case(refs[tid], code, 10.0, wd)
        return key, {"outcome": r["outcome"], "safe": bool(r["safe"]), "violations": r["violations"]}

    def save():
        with open(cache_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(cache_path + ".tmp", cache_path)

    jobs.sort(key=lambda j: j[3])  # 先全部 compiled，再 raw
    jobs = [j[:3] for j in jobs]
    if jobs:
        with ThreadPoolExecutor(workers) as ex:
            for i, (key, res) in enumerate(ex.map(run, jobs), 1):
                cache[key] = res
                if i % 50 == 0:  # 增量落盘：中途被杀也不丢已评结果
                    save()
        save()

    per_task = {}
    for tid, d in entries.items():
        per_task[tid] = {st: [cache[hashlib.sha1((tid + "\0" + c[st]).encode("utf-8")).hexdigest()]
                              for c in d["candidates"]] for st in STAGES
                         if all(st in c for c in d["candidates"])}
    return per_task, entries


def summarize(per_task, refs):
    def agg(tids, st):
        n = len(tids)
        if n == 0:
            return None
        cand = [r for t in tids for r in per_task[t][st]]
        passed = lambda r: r["outcome"] == "passed"  # noqa: E731
        sp = lambda r: passed(r) and r["safe"]  # noqa: E731
        return {
            "n_tasks": n,
            "cand_compile": sum(r["outcome"] != "compile_error" for r in cand) / len(cand),
            "cand_pass": sum(passed(r) for r in cand) / len(cand),
            "cand_safe": sum(r["safe"] for r in cand) / len(cand),
            "pass@N": sum(any(passed(r) for r in per_task[t][st]) for t in tids) / n,
            "safe-pass@N": sum(any(sp(r) for r in per_task[t][st]) for t in tids) / n,
            "E[pass@1]": sum(sum(passed(r) for r in per_task[t][st]) / len(per_task[t][st]) for t in tids) / n,
            "E[safe-pass@1]": sum(sum(sp(r) for r in per_task[t][st]) / len(per_task[t][st]) for t in tids) / n,
        }

    out = {}
    tids = sorted(per_task)
    for st in STAGES:
        if not tids or any(st not in per_task[t] for t in tids):
            continue
        out[st] = {"all": agg(tids, st)}
        by = defaultdict(list)
        for t in tids:
            by[refs[t].get("difficulty", "?")].append(t)
        for k, v in sorted(by.items()):
            out[st][k] = agg(v, st)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pool-dir", action="append", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    refs = {r["task_id"]: r for r in load_jsonl(args.dataset)}
    for pd in args.pool_dir:
        per_task, entries = evaluate_pool(refs, pd, args.workers)
        s = summarize(per_task, refs)
        tok = defaultdict(int)
        for d in entries.values():
            for k, v in d["tokens"].items():
                tok[k] += v
        s["tokens"] = dict(tok)
        s["n_cached_tasks"] = len(entries)
        json.dump(s, open(os.path.join(pd, "_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"== {pd}  ({len(entries)}/{len(refs)} 题已缓存)  tokens={dict(tok)}")
        cols = ["n_tasks", "cand_compile", "cand_pass", "cand_safe", "pass@N", "safe-pass@N", "E[pass@1]", "E[safe-pass@1]"]
        print("stage     group   " + "  ".join(f"{c:>13}" for c in cols))
        for st in STAGES:
            for g, m in s.get(st, {}).items():
                if m:
                    print(f"{st:9} {g:7} " + "  ".join(
                        f"{m[c]:>13}" if c == "n_tasks" else f"{m[c]:>13.3f}" for c in cols))


if __name__ == "__main__":
    main()
