"""思路 3：带分歧证据的裁判（2026-10-03 试验）。

在合规修复后的候选池上：沿用现有选拔（编译 → 违规最少 → 并列集合 tied）；对 tied 中的候选，用条款探针
（必要时补随机探针）找出输出不一致的用例，把"条款原文 + 该用例的输入 + 各候选在分歧字段上的输出 + 各候选代码"
交给 LLM 裁判，让它依据需求逐个判断哪一方符合需求，再给出最终选择。候选之间无分歧时不调用裁判，沿用原选择。

用法：python3 eval/evidence_judge.py --dataset dataset/gnc-adcs-100.jsonl --ds-dir logs/cache/deepseek-v4-flash/gnc100
      [--pools 1 2 3] [--limit N]
缓存：<ds-dir>/_evjudge_cache.json（按 需求+候选+证据 的哈希）
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "../utils"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub)))

import fmv_probe  # noqa: E402
import random_probe as rp  # noqa: E402
from eval import load_jsonl  # noqa: E402
from llm_generate import call_llm, extract_requirement  # noqa: E402
from record import safe_filename  # noqa: E402

MAX_CASES = 4
MAX_FIELDS = 6
_lock = threading.Lock()
_tok = Counter()

PROMPT = """你是一名资深航天 GNC 软件评审专家。下面有同一需求的 {n} 份实现（{labels}），它们在若干测试输入上的输出不一致。
请**只依据需求原文**判断：在每个分歧用例上，哪一份（或哪几份）实现的输出符合需求。最后选出最符合需求的一份。

#需求
{requirement}

#各份实现
{codes}

#分歧证据（同一输入下各实现的输出；只列出不一致的字段）
{evidence}

#判断要求
1. 对每个分歧用例：先根据需求原文写出该输入下"应该怎样"——走哪个分支、输出的符号/大小关系/是否为零/取哪个错误码或模式，
   必要时可做粗略手算；不要求精确数值。再对照各实现的输出，指出哪些符合、哪些不符合，并给出依据（引用需求原句或公式）。
2. 不要因为多数实现一致就认为它们对；也不要依据代码风格、注释或长度判断。
3. 如果某个用例根据需求无法判断，写 "unknown"。
4. 最后给出 best：最符合需求的一份的标签。

#只在最后输出一个 JSON：
{{"cases": [{{"case": "用例编号", "should": "需求要求的行为", "consistent": ["A"], "inconsistent": ["B"]}}], "best": "A"}}
"""


def h(*parts):
    return hashlib.sha1("\0".join(parts).encode("utf-8")).hexdigest()


def close(a, b):
    return ((a != a) and (b != b)) or (a == a and b == b and rp.values_close(a, b))


def load_json(p, d=None):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else d


def divergences(outs):
    """outs: [dict 或 None]；返回 {用例: {字段: [各候选值]}}（仅不一致的字段）。"""
    valid = [o for o in outs if o is not None]
    if len(valid) < 2:
        return {}
    keys = set().union(*[set(o) for o in valid])
    by_case = defaultdict(dict)
    for k in sorted(keys):
        vals = [(o or {}).get(k) for o in outs]
        present = [v for v in vals if v is not None]
        if len(present) < 2:
            continue
        if any(v is None for v in vals) or any(not close(present[0], v) for v in present[1:]):
            case, _, field = k.partition("::")
            by_case[case][field] = vals
    return by_case


def clause_desc(spec, case_map, case):
    """条款探针用例的说明：条款原文 + 相对名义输入的改写。"""
    cid, step = case_map.get(case, (None, None))
    if cid in (None, 0):
        return "名义输入", spec.get("nominal") or []
    for c in spec.get("clauses") or []:
        if str(c.get("id")) == str(cid):
            steps = c.get("steps") or []
            assigns = []
            for s in steps[:(step or 0) + 1]:
                assigns += s.get("assign") or []
            return f"条款 {cid}（{c.get('type')}）：{c.get('text')}", assigns
    return f"条款 {cid}", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--pools", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    D = args.ds_dir.rstrip("/") + "/"
    refs = {r["task_id"]: r for r in load_jsonl(args.dataset)}
    sel = load_json(D + "_selection_summary_misra.json")["picks"]
    cache_p = D + "_evjudge_cache.json"
    cache = load_json(cache_p, {})
    tids = sorted(refs)[:args.limit] if args.limit else sorted(refs)
    log = []

    def run(job):
        p, t = job
        pk = sel[str(p)].get(t)
        if pk is None:
            return None
        ref = refs[t]
        d = load_json(D + f"pool{p}/{safe_filename(t)}.json")
        ev = EV[p]
        codes = [c["misra"] for c in d["candidates"]]
        res = [ev[h(t, c)] for c in codes]
        P = [r["outcome"] == "passed" for r in res]
        S = [P[i] and r["safe"] for i, r in enumerate(res)]
        tied, ours = pk["misra_random"], pk["ours"][0]
        out = {"p": p, "t": t, "P": P, "S": S, "ours": ours, "pick": ours, "path": "unique"}
        if len(tied) < 2:
            return out
        # 证据：条款探针优先，随机探针补充
        evid, cmap, spec = {}, {}, {}
        cl = load_json(D + f"_clause/{safe_filename(t)}.json")
        if cl and cl.get("ok"):
            spec = (load_json(D + f"_clause/{safe_filename(t)}.spec.json") or {}).get("spec") or {}
            cmap = {k: tuple(v) for k, v in cl["case_map"].items()}
            outs = [fmv_probe.probe_candidate(ref, cl["harness"], codes[i]) for i in tied]
            evid = {("clause", k): v for k, v in divergences(outs).items()}
        if not evid:
            hr = load_json(D + f"_harness/{safe_filename(t)}.json")
            if hr and hr.get("ok"):
                outs = [fmv_probe.probe_candidate(ref, hr["harness"], codes[i]) for i in tied]
                evid = {("rand", k): v for k, v in divergences(outs).items()}
        if not evid:
            out["path"] = "no_divergence"
            return out
        rng = random.Random(h(t, str(p)))
        order = list(range(len(tied)))
        rng.shuffle(order)
        labels = [chr(ord("A") + i) for i in range(len(tied))]
        lab_of = {tied[order[j]]: labels[j] for j in range(len(tied))}
        code_txt = "\n".join(f"##实现 {labels[j]}\n```c\n{codes[tied[order[j]]]}\n```" for j in range(len(tied)))
        ev_lines = []
        for (kind, case), fields in list(evid.items())[:MAX_CASES]:
            if kind == "clause":
                desc, assigns = clause_desc(spec, cmap, case)
                nominal = spec.get("nominal") or []
                inp = "; ".join(f"{a.get('lvalue')}={a.get('value')}" for a in (nominal + assigns)[-24:]
                                if isinstance(a, dict))
                ev_lines.append(f"###用例 {case}：{desc}\n输入（名义输入 + 改写）：{inp}")
            else:
                ev_lines.append(f"###用例 r{case}：取值范围内的随机合法输入（输入值未列出）")
            for f, vals in list(fields.items())[:MAX_FIELDS]:
                cells = ", ".join(f"{lab_of[tied[i]]}={'无输出' if vals[i] is None else format(vals[i], '.6g')}"
                                  for i in range(len(tied)))
                ev_lines.append(f"- {f}: {cells}")
        prompt = PROMPT.format(n=len(tied), labels="、".join(labels), requirement=extract_requirement(ref["prompt"]),
                               codes=code_txt, evidence="\n".join(ev_lines))
        key = h(prompt)
        with _lock:
            hit = cache.get(key)
        if hit is None:
            content, usage = call_llm(args.model, [{"role": "user", "content": prompt}])
            with _lock:
                _tok["judge"] += (usage or {}).get("total_tokens", 0)
            best = None
            try:
                best = json.loads(content[content.rfind('{"cases"'):content.rfind("}") + 1]).get("best")
            except (ValueError, AttributeError):
                m = re.search(r'"best"\s*:\s*"([A-Z])"', content)
                best = m.group(1) if m else None
            hit = {"best": best}
            with _lock:
                cache[key] = hit
        best = hit.get("best")
        inv = {v: k for k, v in lab_of.items()}
        out["path"] = "evidence_judge"
        out["n_cases"] = len(evid)
        out["pick"] = inv.get(best, ours)
        return out

    global EV
    EV = {p: load_json(D + f"pool{p}/_eval.json") for p in args.pools}
    jobs = [(p, t) for p in args.pools for t in tids]
    with ThreadPoolExecutor(args.workers) as ex:
        R = [r for r in ex.map(run, jobs) if r]
    json.dump(cache, open(cache_p, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(R, open(D + f"_evjudge_results{'_lim' if args.limit else ''}.json", "w"), ensure_ascii=False, indent=1)
    paths = Counter(r["path"] for r in R)
    print(f"tokens={dict(_tok)} paths={dict(paths)}")
    for name, key in (("ours", "ours"), ("evidence_judge", "pick")):
        pv = [sum(r["P"][r[key]] for r in R if r["p"] == p) / max(1, sum(1 for r in R if r["p"] == p)) for p in args.pools]
        sv = [sum(r["S"][r[key]] for r in R if r["p"] == p) / max(1, sum(1 for r in R if r["p"] == p)) for p in args.pools]
        print(f"  {name:15} pass {100 * statistics.mean(pv):.1f}  safe {100 * statistics.mean(sv):.1f}  {[round(100 * x, 1) for x in sv]}")
    ch = [r for r in R if r["path"] == "evidence_judge"]
    better = sum(r["S"][r["pick"]] and not r["S"][r["ours"]] for r in ch)
    worse = sum(r["S"][r["ours"]] and not r["S"][r["pick"]] for r in ch)
    mixed = [r for r in ch if any(r["S"]) and not all(r["S"])]
    print(f"  invoked {len(ch)}; changed-to-correct {better}; changed-to-wrong {worse}; "
          f"mixed pools judged {len(mixed)}, correct pick in mixed {sum(r['S'][r['pick']] for r in mixed)}")


EV = {}
if __name__ == "__main__":
    main()
