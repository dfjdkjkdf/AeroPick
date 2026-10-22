"""实验①：单点程序化计算（2026-10-03）。

对候选间有分歧的条款探针用例：让被测模型只依据需求（及上下文中的常量/宏），针对该用例的具体输入写一小段
只用 math 的 Python 代码，算出分歧字段的值；在 Docker 中执行；与各候选输出比对，匹配最多者胜出。
算不出/平局时沿用原选择。只读需求与输入，不看任何候选代码（避免被候选带偏）。

用法：python3 eval/point_compute.py --dataset dataset/gnc-adcs-100.jsonl --ds-dir logs/cache/deepseek-v4-flash/gnc100
      [--only-mixed]   （评估用：只在有对有错的池上调用，其他池选择不变）
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "../utils"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub)))

import evidence_judge as ej  # noqa: E402
import fmv_probe  # noqa: E402
import random_probe as rp  # noqa: E402
from eval import docker_run, load_jsonl  # noqa: E402
from llm_generate import call_llm, extract_requirement, strip_requirement_comment  # noqa: E402
from record import safe_filename  # noqa: E402

MAX_CASES = 2
MAX_FIELDS = 6
CTX_LIMIT = 12000
REL_TOL = 1e-3
_lock = threading.Lock()
_tok = Counter()

PROMPT = """你是一名航天 GNC 算法工程师。请**只依据需求**，在下面给定的一组具体输入下，计算指定输出字段的值。
写一段 Python3 程序完成计算（只能 import math），不要写成通用函数，只算这一组输入。

#需求
{requirement}

{glossary}#C代码上下文（用于查常量、宏、结构体字段含义）
```c
{context}
```

#这一组输入（C 赋值语句；未列出的输入一律为 0）
{inputs}
{steps}
#需要计算的输出字段（每个字段打印一行，格式严格为 "F序号: 数值"）
{fields}

#要求
- 严格按需求中的公式和分支逐步计算；需求中的条件判断按这组输入的具体数值判断走哪个分支。
- 宏/常量的数值从上下文中取（如 PI、DEG2RAD、TRUE32=0xEB90146F）。
- 若函数返回值或标志为宏，打印其数值。
- 只输出一个 ```python 代码块。
"""


CTX_FULL = 30000


def build_context(prompt, mode):
    """tail：旧做法，只保留尾部 CTX_LIMIT 字符（会截掉前部的宏定义）。
    full：不超过 CTX_FULL 时给全文；超过时把全部 #define 行放在最前，再接尾部 CTX_FULL//2 字符。"""
    ctx = strip_requirement_comment(prompt)
    if mode == "tail":
        return ctx[-CTX_LIMIT:] if len(ctx) > CTX_LIMIT else ctx
    if len(ctx) <= CTX_FULL:
        return ctx
    defines = []
    for ln in ctx.splitlines():
        if ln.lstrip().startswith("#define") and ln not in defines:
            defines.append(ln)
    return "/* 宏定义（全部） */\n" + "\n".join(defines) + "\n/* …（中间省略）… */\n" + ctx[-(CTX_FULL // 2):]


def h(*parts):
    return hashlib.sha1("\0".join(parts).encode("utf-8")).hexdigest()


def pretty(field):
    f = field.replace(rp.PREFIX + "arg_", "")
    return re.sub(r"^(\w+)\[0\]", r"\1", f)


def run_python(code):
    with tempfile.TemporaryDirectory(prefix="pc_") as wd:
        open(os.path.join(wd, "s.py"), "w", encoding="utf-8").write(code)
        r = docker_run(wd, "timeout -s KILL 10 python3 s.py", 30)
    vals = {}
    for k, v in re.findall(r"^F(\d+)\s*:\s*(\S+)", r.stdout or "", re.M):
        try:
            vals[int(k)] = float(v)
        except ValueError:
            pass
    return vals, (r.stderr or "")[-300:]


def match(a, b):
    """按量级的相对容差，且正负号不同判为不一致（2026-10-03 修正：原 1e-3*(1+|a|+|b|) 对小量级字段过宽）。"""
    if a is None or b is None:
        return False
    if a == b:  # 2026-10-04：含 inf==inf（原先 inf-inf=nan 判为不一致）
        return True
    if a in (float("inf"), float("-inf")) or b in (float("inf"), float("-inf")):
        return False  # 无穷与有限值（或异号无穷）不一致（原先 inf 与大数会判为一致）
    if a != a or b != b:
        return (a != a) and (b != b)
    big = max(abs(a), abs(b))
    if big > 1e-9 and min(abs(a), abs(b)) > 1e-12 and (a > 0) != (b > 0):
        return False
    return abs(a - b) <= 1e-4 * big + 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--pools", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--only-mixed", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--context", choices=["tail", "full"], default="tail")
    ap.add_argument("--glossary", default=None, help="项目约定词表文件")
    ap.add_argument("--tag", default="", help="结果文件后缀，区分不同变体")
    args = ap.parse_args()
    glossary = ""
    if args.glossary:
        glossary = open(args.glossary, encoding="utf-8").read().strip() + "\n\n"
    D = args.ds_dir.rstrip("/") + "/"
    refs = {r["task_id"]: r for r in load_jsonl(args.dataset)}
    sel = ej.load_json(D + "_selection_summary_misra.json")["picks"]
    EV = {p: ej.load_json(D + f"pool{p}/_eval.json") for p in args.pools}
    cache_p = D + "_pointcomp_cache.json"
    cache = ej.load_json(cache_p, {})

    def run(job):
        p, t = job
        pk = sel[str(p)].get(t)
        if pk is None:
            return None
        ref = refs[t]
        d = ej.load_json(D + f"pool{p}/{safe_filename(t)}.json")
        codes = [c["misra"] for c in d["candidates"]]
        res = [EV[p][h(t, c)] for c in codes]
        P = [r["outcome"] == "passed" for r in res]
        S = [P[i] and r["safe"] for i, r in enumerate(res)]
        ours, tied = pk["ours"][0], pk["misra_random"]
        out = {"p": p, "t": t, "P": P, "S": S, "ours": ours, "pick": ours, "path": "unique"}
        mixed = any(S[i] for i in tied) and not all(S[i] for i in tied)
        if len(tied) < 2 or (args.only_mixed and not mixed):
            out["path"] = "skipped" if len(tied) >= 2 else "unique"
            return out
        cl = ej.load_json(D + f"_clause/{safe_filename(t)}.json")
        if not (cl and cl.get("ok")):
            out["path"] = "no_clause_harness"
            return out
        spec = (ej.load_json(D + f"_clause/{safe_filename(t)}.spec.json") or {}).get("spec") or {}
        cmap = {k: tuple(v) for k, v in cl["case_map"].items()}
        outs = [fmv_probe.probe_candidate(ref, cl["harness"], codes[i]) for i in tied]
        div = ej.divergences(outs)
        if not div:
            out["path"] = "no_divergence"
            return out
        score = [0] * len(tied)
        computed_any = False
        details = []
        for case, fields in list(div.items())[:MAX_CASES]:
            desc, assigns = ej.clause_desc(spec, cmap, case)
            nominal = spec.get("nominal") or []
            inputs = "\n".join(f"{a.get('lvalue')} = {a.get('value')};" for a in nominal + assigns if isinstance(a, dict))
            cid, step = cmap.get(case, (None, None))
            steps_txt = ""
            for c in spec.get("clauses") or []:
                if str(c.get("id")) == str(cid):
                    st = (c.get("steps") or [])[: (step or 0) + 1]
                    if len(st) > 1 or any(int(s.get("repeat", 1) or 1) > 1 for s in st if isinstance(s, dict)):
                        steps_txt = "\n#调用过程（函数有状态时按此顺序多次调用，计算最后一次调用后的输出）\n" + "\n".join(
                            f"第{i + 1}步：赋值 {json.dumps(s.get('assign'), ensure_ascii=False)}，连续调用 {s.get('repeat', 1)} 次"
                            for i, s in enumerate(st) if isinstance(s, dict))
            flist = list(fields.items())[:MAX_FIELDS]
            fields_txt = "\n".join(f"F{i + 1}: {pretty(f)}" for i, (f, _v) in enumerate(flist))
            ctx = build_context(ref["prompt"], args.context)
            prompt = PROMPT.format(requirement=extract_requirement(ref["prompt"]), context=ctx, inputs=inputs,
                                   steps=steps_txt, fields=fields_txt, glossary=glossary)
            key = h(prompt)
            with _lock:
                hit = cache.get(key)
            if hit is None:
                content, usage = call_llm(args.model, [{"role": "user", "content": prompt}])
                with _lock:
                    _tok["llm"] += (usage or {}).get("total_tokens", 0)
                m = re.search(r"```(?:python|py)?\s*\n(.*?)```", content, re.S)
                code = m.group(1) if m else ""
                vals, err = run_python(code) if code else ({}, "no_code")
                hit = {"vals": {str(k): v for k, v in vals.items()}, "err": err}
                with _lock:
                    cache[key] = hit
            vals = {int(k): v for k, v in hit["vals"].items()}
            details.append({"case": case, "desc": desc, "key": key, "err": hit.get("err"),
                            "fields": [{"f": pretty(f), "cand": cv, "exp": vals.get(i + 1)} for i, (f, cv) in enumerate(flist)]})
            for i, (_f, cand_vals) in enumerate(flist):
                exp = vals.get(i + 1)
                if exp is None:
                    continue
                computed_any = True
                for j in range(len(tied)):
                    if match(cand_vals[j], exp):
                        score[j] += 1
        out["details"] = details
        out["tied"] = tied
        out["score"] = score
        if not computed_any or max(score) == 0:
            out["path"] = "no_computation"
            return out
        best = [tied[j] for j in range(len(tied)) if score[j] == max(score)]
        out["path"] = "computed"
        out["pick"] = ours if ours in best else best[0]
        out["tie"] = len(best) > 1
        return out

    jobs = [(p, t) for p in args.pools for t in sorted(refs)]
    with ThreadPoolExecutor(args.workers) as ex:
        R = [r for r in ex.map(run, jobs) if r]
    json.dump(cache, open(cache_p, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(R, open(D + f"_pointcomp_results{args.tag}.json", "w"), ensure_ascii=False, indent=1)
    print(f"tokens={dict(_tok)} paths={dict(Counter(r['path'] for r in R))}")
    for name, key in (("ours", "ours"), ("point_compute", "pick")):
        sv = [sum(r["S"][r[key]] for r in R if r["p"] == p) / max(1, sum(1 for r in R if r["p"] == p)) for p in args.pools]
        print(f"  {name:14} safe {100 * statistics.mean(sv):.1f}  {[round(100 * x, 1) for x in sv]}")
    mixed = [r for r in R if any(r["S"][i] for i in range(len(r["S"]))) and not all(r["S"])]
    comp = [r for r in mixed if r["path"] == "computed"]
    print(f"  mixed pools {len(mixed)}; computed {len(comp)}; correct pick: ours {sum(r['S'][r['ours']] for r in comp)} "
          f"-> point {sum(r['S'][r['pick']] for r in comp)}; changed-to-correct "
          f"{sum(r['S'][r['pick']] and not r['S'][r['ours']] for r in comp)}, changed-to-wrong "
          f"{sum(r['S'][r['ours']] and not r['S'][r['pick']] for r in comp)}; ties {sum(r.get('tie', False) for r in comp)}")


if __name__ == "__main__":
    main()
