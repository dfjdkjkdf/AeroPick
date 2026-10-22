"""外部基线：思考模型直接读代码挑选（LLM-as-a-judge，CodeJudge / Crupi et al. 一类）。
与我们、Fan 共用候选（编译通过→违规最少的并列候选）；内容相同的代码合为一组；只剩 1 组时不调用。
提示 = 需求 + C 代码上下文 + 各实现代码（A/B/C），思考模式，不出测试、不执行、无项目约定词表；只输出 {"pick":..}。
解析失败或调用失败时在并列候选上取期望。用法：code_judge.py <名> <jsonl>；环境变量 MODEL_DIR、POOLS、LLM_MODEL 同 llm_judge；
CJ_DRY=1 只统计不调用；CJ_WORKERS 并发。缓存 <ds>/_codejudge/，结果 _bprobe/_codejudge_result.json。"""
import json, sys, os, re, hashlib, statistics, threading
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import llm_tests as L  # noqa: E402
from llm_tests import ej, pc, refs, D, OUT, call_llm, extract_requirement  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

DRY = os.environ.get("CJ_DRY") == "1"
THINK = os.environ.get("CJ_THINK", "1") == "1"  # 2026-10-10：CJ_THINK=0 为不思考版（缓存与结果文件另存）
JD = D + ("_codejudge/" if THINK else "_codejudge_nothink/")
os.makedirs(JD, exist_ok=True)
POOLS = [int(x) for x in os.environ.get("POOLS", "1 2 3").split()]
_lock = threading.Lock()
TOK = {"prompt": 0, "completion": 0, "calls": 0, "est_chars": 0}
PROMPT = """你是航天 GNC 软件评审专家。下面是同一个 C 函数的 {k} 种不同实现（记为 {labels}）。
请对照需求，判断哪一种实现最符合需求（功能正确）。

#需求
{requirement}

#C代码上下文（类型、宏、常量）
```c
{context}
```

{impls}

只输出 JSON：{{"pick":"{label_opts}","reason":"一句话"}}
"""


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m: return None
    try: return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
    except Exception: return None


def run(job):
    t, p = job
    ref = refs[t]
    codes, S, comp, tied = L.load_pool(t, p)
    groups = {}
    for i in tied: groups.setdefault(codes[i].strip(), []).append(i)
    G = list(groups.values())
    base = statistics.mean(S[i] for i in tied)
    res = {"t": t, "p": p, "final": base, "path": "single"}
    if len(G) < 2: return res
    labels = [chr(65 + k) for k in range(len(G))]
    impls = "\n\n".join(f"#实现{lab}\n```c\n{codes[g[0]]}\n```" for lab, g in zip(labels, G))
    prompt = PROMPT.format(k=len(G), labels="、".join(labels), label_opts="|".join(labels),
                           requirement=extract_requirement(ref["prompt"]), context=pc.build_context(ref["prompt"], "full"), impls=impls)
    if DRY:
        with _lock: TOK["calls"] += 1; TOK["est_chars"] += len(prompt)
        res["path"] = "dry"; return res
    cp = JD + hashlib.sha1(prompt.encode()).hexdigest() + ".json"
    hit = ej.load_json(cp)
    if hit is None:
        try:
            content, usage = call_llm(L.MODEL, [{"role": "user", "content": prompt}], thinking=THINK, max_tokens=32768 if THINK else None)
            if THINK and not parse(content):  # 思考过长输出为空：加大上限重试一次
                with _lock:
                    TOK["prompt"] += (usage or {}).get("prompt_tokens", 0); TOK["completion"] += (usage or {}).get("completion_tokens", 0); TOK["calls"] += 1
                content, usage = call_llm(L.MODEL, [{"role": "user", "content": prompt}], thinking=True, max_tokens=65536)
        except Exception as e:
            print(f"[llm_error] {t} p{p}: {str(e)[:200]}", flush=True)
            res["path"] = "llm_error"; return res
        with _lock:
            TOK["prompt"] += (usage or {}).get("prompt_tokens", 0); TOK["completion"] += (usage or {}).get("completion_tokens", 0); TOK["calls"] += 1
        hit = parse(content) or {"raw": (content or "")[:4000]}
        hit["_usage"] = {"prompt": (usage or {}).get("prompt_tokens"), "completion": (usage or {}).get("completion_tokens")}
        json.dump(hit, open(cp, "w"), ensure_ascii=False, indent=1)
    pick = str(hit.get("pick", "")).strip()[:1]
    if pick in labels:
        g = G[labels.index(pick)]
        res["final"] = statistics.mean(S[i] for i in g); res["path"] = "picked"
    else:
        res["path"] = "no_pick"
    res["usage"] = hit.get("_usage")
    return res


if __name__ == "__main__":
    jobs = [(t, p) for t in sorted(refs) for p in POOLS]
    with ThreadPoolExecutor(int(os.environ.get("CJ_WORKERS", "4"))) as ex:
        R = list(ex.map(run, jobs))
    from collections import Counter
    print(f"== {L.bp.n if hasattr(L, 'bp') else ''} 代码裁判 {len(jobs)} 个池  路径 {dict(Counter(r['path'] for r in R))}  tokens {TOK}")
    if DRY: sys.exit(0)
    json.dump(R, open(OUT + (("_codejudge_result" if THINK else "_codejudge_nothink_result") + (".json" if POOLS == [1, 2, 3] else f"_p{''.join(map(str, POOLS))}.json")), "w"), ensure_ascii=False, indent=1)  # 分池运行时按池号分文件，避免覆盖
    runs = [100 * statistics.mean(r["final"] for r in R if r["p"] == p) for p in POOLS]
    print(f"   代码裁判 safe-pass@1 {statistics.mean(runs):.1f} ± {statistics.stdev(runs) if len(runs) > 1 else 0:.1f} {[round(x, 1) for x in runs]}")
