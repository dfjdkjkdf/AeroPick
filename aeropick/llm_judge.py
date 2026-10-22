"""第四步精判：对粗筛后仍有分歧的池，把模型出的题上各组输出的具体差异交给裁判（被测模型），对照需求判断哪组对。
裁判不看代码、不看组的人数。用法：llm_judge.py <名> <jsonl> [池数上限] [随机种子]
只处理"模型出题"分组 ≥2 组的池；没选出/解析失败时沿用分层结果。"""
import json, sys, os, re, hashlib, random, statistics, threading
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import llm_tests as L  # noqa: E402  （按 argv 设置数据集）
from llm_tests import bp, fp, ej, pc, rp, refs, D, OUT, safe_filename, call_llm, extract_requirement  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 20261004
THINK = os.environ.get("JUDGE_THINK") == "1"  # 思考模式裁判（方案 A）
GLOSS = os.environ.get("JUDGE_GLOSS") == "1"  # 方案①：裁判提示加项目约定词表 + 输出数值标注宏名
EV2 = os.environ.get("JUDGE_EV2") == "1"  # 方案②：按覆盖面挑分歧测试，多拍展示所有分歧步，关键字段优先
STRUCT = os.environ.get("JUDGE_STRUCT") == "1"
BUDGET = int(os.environ.get("JUDGE_BUDGET", "0"))  # 提示中要求的思考字数上限（0 表示不限）  # 方案 C：非思考 + 由思考过程提炼的固定答题格式
JD = D + (f"_judge_think_gloss_b{BUDGET}/" if BUDGET else "_judge_struct/" if STRUCT else "_judge_gloss/" if GLOSS and not THINK else "_judge_think_gloss_ev2/" if GLOSS and EV2 else "_judge_think_gloss/" if GLOSS else "_judge_think/" if THINK else "_judge/")
GLOSS_TEXT = open("utils/conventions_zh.txt", encoding="utf-8").read().strip()
_TLM = threading.local()
# Optional project-wide header whose integer macros (e.g. TRUE32) are used to annotate outputs; not distributed.
STD_HDR = open("eval/includes/std_utils.h", encoding="utf-8", errors="ignore").read() if os.path.exists("eval/includes/std_utils.h") else ""


def macro_map(prompt):
    """上下文中 #define 的整数宏（|值| >= 256 的"魔数"），用于把输出数值标注为宏名。"""
    m = {}
    prompt = prompt + "\n" + STD_HDR  # 公共头文件 std_utils.h 中的宏（如 TRUE32）所有题共用
    for name, val in re.findall(r"#define\s+(\w+)\s+\(?\s*(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\s*\)?\s*(?:$|/)", prompt, re.M):
        v = int(val, 16) if val.lower().startswith("0x") else int(val)
        if abs(v) >= 256:
            m.setdefault(v, name)
    return m
os.makedirs(JD, exist_ok=True)
MAX_TESTS, MAX_FIELDS = 6, 8
_lock = threading.Lock()
TOK = {"prompt": 0, "completion": 0, "calls": 0}

PROMPT = """你是航天 GNC 软件评审专家。同一个 C 函数有几种不同实现（记为 {labels}），它们在下面这些测试上的输出不同。
请**只依据需求**判断：在每个测试上，哪一种实现的输出符合需求。你看不到代码，也不知道各实现的数量。

#需求
{requirement}

#C代码上下文（类型、宏、常量）
```c
{context}
```

#有分歧的测试
{evidence}

#判断要求
- 按需求逐条推理这个输入应走的分支和应得到的结果（数值可近似估算，看符号、量级、是否清零/置默认值、状态/模式、返回码）。
- 若需求对这个输入的行为没有规定，记为"未规定"，不要据此判断。
- "__crash__: 1" 表示该实现在此测试上崩溃或超时。
- 最后综合所有测试，选出最符合需求的一种实现。

只输出 JSON：{{"tests":[{{"id":测试编号,"correct":"{label_opts}|未规定|无法判断","reason":"一句话"}}],"pick":"{label_opts}"}}
"""


PROMPT_STRUCT_TAIL = """#答题格式（必须按顺序填写；只输出一个 JSON，不要输出其他文字）
{{"conventions": ["用到的约定/常量及其具体数值，如 TRUE32=3952088175、失败时输出的默认值、阈值及比较方向；与词表冲突时以词表为准"],
 "tests": [
  {{"id": 测试编号,
   "actual": "只抄各实现在区分字段上的实际值，如 A: mode=2, ret=0; B: mode=0, ret=1（崩溃/无输出照写）",
   "branch": "这组输入适用的需求条款原文要点及应走的分支；先检查失效/错误/保护条件（NULL、非有限、退化、越界、阈值边界），再看正常路径；多拍测试逐拍写状态",
   "expected": "区分字段的期望值（只需精确到能区分各实现：符号、量级、是否为0、模式值、返回码）；需求对该字段确无规定才写'未规定'",
   "ok": ["符合需求的实现标签，可多个；都不符合写空列表；未规定写['未规定']"]}}
 ],
 "violations": {{"A": 不符合需求的测试数, "B": 不符合需求的测试数}},
 "pick": "违反最少的单个实现标签，必须与上面逐测试结论一致"}}
要求：不要分析各实现输出相同的字段；不要只写"按公式计算"而不给数值；pick 只能是一个标签。
"""


def pretty(k):
    return k.replace(rp.PREFIX + "obj_", "").replace(rp.PREFIX, "")


def fmt(v):
    if v is None:
        return "（无输出）"
    s = f"{v:.6g}"
    mm = getattr(_TLM, "macros", None)
    if mm and v == v and abs(v) < 1e19 and float(v).is_integer() and int(v) in mm:
        s = f"{int(v)}（={mm[int(v)]}）"
    elif mm and v == v and abs(v) < 1e19 and float(v).is_integer():
        s = f"{int(v)}" if abs(v) >= 1e6 else s
    return s


def evidence(t, tests, groups, outs):
    reps = [g[0] for g in groups]
    labels = [chr(ord("A") + i) for i in range(len(groups))]
    cases = sorted({k.split("::")[0] for r in reps for k in (outs[r] or {})}, key=lambda c: int(c) if c.isdigit() else 10 ** 9)
    div = []
    for c in cases:
        fields = sorted({k.split("::", 1)[1] for r in reps for k in (outs[r] or {}) if k.split("::")[0] == c})
        diff = []
        for f in fields:
            vals = [(outs[r] or {}).get(f"{c}::{f}") for r in reps]
            if any(v is None for v in vals) and not all(v is None for v in vals) or \
                    any(not pc.match(vals[0], v) for v in vals[1:] if v is not None and vals[0] is not None):
                diff.append((f, vals))
        if diff:
            div.append((c, diff))
    # 每个测试只取第一个出现分歧的步
    seen, picked = set(), []
    for c, diff in div:
        ti = int(c) // 10 if c.isdigit() else -1
        if ti in seen or ti < 0 or ti >= len(tests):
            continue
        seen.add(ti)
        picked.append((ti, int(c) % 10, diff))
        if len(picked) >= MAX_TESTS:
            break
    blocks = []
    for n, (ti, si, diff) in enumerate(picked, 1):
        steps = [s for s in (tests[ti].get("steps") or []) if isinstance(s, str)][: si + 1]
        st = "\n".join(f"  第{j + 1}次调用前赋值：\n" + "\n".join("    " + x.strip() for x in s.split("\n") if x.strip())
                       for j, s in enumerate(steps))
        rows = "\n".join(f"  {pretty(f)}：" + "；".join(f"{lab}={fmt(v)}" for lab, v in zip(labels, vals))
                         for f, vals in diff[:MAX_FIELDS])
        more = f"\n  （另有 {len(diff) - MAX_FIELDS} 个字段也不同，略）" if len(diff) > MAX_FIELDS else ""
        blocks.append(f"测试 {n}（类型 {tests[ti].get('kind', '')}；未赋值的量为 0；共调用 {si + 1} 次，比较最后一次调用后的输出）\n{st}\n"
                      f"  输出差异：\n{rows}{more}")
    return labels, "\n\n".join(blocks), len(picked)


_KEY_FIELD = re.compile(r"__ret__|mode|state|status|flag|valid|fault|err|latch|cnt|count|accept|_b[A-Z]|\.b[A-Z]|\bb[A-Z]|F_|Flag|bool", re.I)
MAX_TESTS2, MAX_STEPS2, MAX_FIELDS2 = 10, 3, 8


def _base(f):
    return re.sub(r"\[\d+\]", "[]", f)


def evidence_v2(t, tests, groups, outs):
    """按覆盖面挑分歧测试：贪心选覆盖最多"新分歧字段"的测试（关键字段计双倍），至多 10 个；
    每个测试展示至多 3 个分歧步（分歧字段多者优先），字段按关键性排序、数组元素合并为一行。"""
    reps = [g[0] for g in groups]
    labels = [chr(ord("A") + i) for i in range(len(groups))]
    cases = sorted({k.split("::")[0] for r in reps for k in (outs[r] or {})}, key=lambda c: int(c) if c.isdigit() else 10 ** 9)
    per_test = {}
    for c in cases:
        if not c.isdigit():
            continue
        ti, si = int(c) // 10, int(c) % 10
        if ti >= len(tests):
            continue
        fields = sorted({k.split("::", 1)[1] for r in reps for k in (outs[r] or {}) if k.split("::")[0] == c})
        diff = []
        for f in fields:
            vals = [(outs[r] or {}).get(f"{c}::{f}") for r in reps]
            if any(v is None for v in vals) and not all(v is None for v in vals) or \
                    any(not pc.match(vals[0], v) for v in vals[1:] if v is not None and vals[0] is not None):
                diff.append((f, vals))
        if diff:
            per_test.setdefault(ti, []).append((si, diff))
    w = lambda b: 2 if _KEY_FIELD.search(b) else 1
    covered, chosen = set(), []
    while len(chosen) < MAX_TESTS2:
        best, gain = None, 0
        for ti, steps in per_test.items():
            if ti in chosen:
                continue
            bases = {_base(f) for _si, d in steps for f, _v in d}
            g = sum(w(b) for b in bases - covered)
            if g > gain:
                best, gain = ti, g
        if best is None:
            break
        chosen.append(best)
        covered |= {_base(f) for _si, d in per_test[best] for f, _v in d}
    # 剩余名额：按测试类型轮换、关键字段分歧多者优先补满
    rest = [ti for ti in per_test if ti not in chosen]
    keyn = lambda ti: sum(1 for _si, d in per_test[ti] for f, _v in d if _KEY_FIELD.search(f))
    rest.sort(key=lambda ti: (-keyn(ti), ti))
    kinds_used = {tests[ti].get("kind") for ti in chosen}
    while rest and len(chosen) < MAX_TESTS2:
        nxt = next((ti for ti in rest if tests[ti].get("kind") not in kinds_used), rest[0])
        rest.remove(nxt)
        chosen.append(nxt)
        kinds_used.add(tests[nxt].get("kind"))
        if len(kinds_used) >= len({tests[ti].get("kind") for ti in per_test}):
            kinds_used = set()
    blocks = []
    for n, ti in enumerate(sorted(chosen), 1):
        steps_div = sorted(per_test[ti], key=lambda x: -len(x[1]))[:MAX_STEPS2]
        steps_div.sort()
        last = steps_div[-1][0]
        steps = [s for s in (tests[ti].get("steps") or []) if isinstance(s, str)][: last + 1]
        st = "\n".join(f"  第{j + 1}次调用前赋值：\n" + "\n".join("    " + x.strip() for x in s.split("\n") if x.strip())
                       for j, s in enumerate(steps))
        parts = []
        for si, diff in steps_div:
            dead = [k for k in range(len(labels)) if all(vals[k] is None for _f, vals in diff)]
            note = "".join(f"\n    {labels[k]}：无输出（崩溃或超时）" for k in dead)
            live = [k for k in range(len(labels)) if k not in dead]
            diff = [(f, vals) for f, vals in diff if len(live) >= 2 and (
                any(vals[k] is None for k in live) or any(not pc.match(vals[live[0]], vals[k]) for k in live[1:]))]
            merged = {}
            for f, vals in diff:
                merged.setdefault(_base(f), []).append((f, vals))
            order = sorted(merged, key=lambda b: (0 if _KEY_FIELD.search(b) else 1, b))
            rows = []
            for b in order[:MAX_FIELDS2]:
                items = merged[b]
                if len(items) == 1:
                    f, vals = items[0]
                    rows.append(f"    {pretty(f)}：" + "；".join(f"{lab}={fmt(v)}" for lab, v in zip(labels, vals)))
                else:
                    rows.append(f"    {pretty(b)}（{len(items)} 个元素不同，列前 3 个）：" + "；".join(
                        f"{lab}=(" + ", ".join(f"{pretty(f).split('[')[-1].rstrip(']') if False else ''}{fmt(vals[k])}" for f, vals in items[:3]) + ")"
                        for k, lab in enumerate(labels)))
            more = f"\n    （另有 {len(order) - MAX_FIELDS2} 个字段也不同，略）" if len(order) > MAX_FIELDS2 else ""
            parts.append(f"  第{si + 1}次调用后的输出差异：{note}\n" + "\n".join(rows) + more)
        blocks.append(f"测试 {n}（类型 {tests[ti].get('kind', '')}；未赋值的量为 0；共调用 {last + 1} 次）\n{st}\n" + "\n".join(parts))
    return labels, "\n\n".join(blocks), len(chosen)


def parse_json(text):
    """宽松解析：去掉 ```、尾随逗号。"""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    s = re.sub(r",\s*([}\]])", r"\1", m.group(0))
    try:
        return json.loads(s)
    except Exception:
        return None


def judge(job):
    t, p = job
    ref = refs[t]
    codes, S, comp, tied = L.load_pool(t, p)
    hl = ej.load_json(OUT + f"{safe_filename(t)}.p{p}.llmtests.json")
    tests = ej.load_json(L.TD + safe_filename(t) + ".json")["tests"]
    outs = {i: fp.probe_candidate(ref, hl["harness"], codes[i], timeout=120.0) for i in tied}
    valid = [i for i in tied if outs.get(i)]
    groups = bp.clusters(outs, valid) if len(valid) >= 2 else []
    res = {"t": t, "p": p, "S": S, "groups": groups}
    if len(groups) < 2:
        res["path"] = "no_div"
        return res
    _TLM.macros = macro_map(ref["prompt"]) if GLOSS else None
    labels, ev, n_ev = (evidence_v2 if EV2 else evidence)(t, tests, groups, outs)
    _TLM.macros = None
    if not n_ev:
        res["path"] = "no_evidence"
        return res
    tmpl = PROMPT.replace("#有分歧的测试", GLOSS_TEXT.replace("{", "{{").replace("}", "}}") + "\n\n#有分歧的测试") if GLOSS else PROMPT
    if STRUCT:
        tmpl = tmpl[:tmpl.index("#判断要求")] + PROMPT_STRUCT_TAIL
    if BUDGET:
        tmpl = tmpl.replace("#判断要求\n", f"#判断要求\n- 思考请控制在约 {BUDGET} 字以内：只对能区分各实现的字段计算，算到足以区分即停止，不要反复验算或分析各实现相同的输出。\n", 1)
    prompt = tmpl.format(labels="、".join(labels), label_opts="|".join(labels), requirement=extract_requirement(ref["prompt"]),
                         context=pc.build_context(ref["prompt"], "full"), evidence=ev)
    key = hashlib.sha1(prompt.encode()).hexdigest()
    cp = JD + key + ".json"
    hit = ej.load_json(cp)
    if hit is None:
        try:
            if THINK:
                content, usage = call_llm(L.MODEL, [{"role": "user", "content": prompt}], thinking=True, max_tokens=32768)
                if GLOSS and not parse_json(content):  # 思考过长输出为空：加大上限重试一次
                    with _lock:
                        TOK["prompt"] += (usage or {}).get("prompt_tokens", 0)
                        TOK["completion"] += (usage or {}).get("completion_tokens", 0)
                        TOK["calls"] += 1
                    content, usage = call_llm(L.MODEL, [{"role": "user", "content": prompt}], thinking=True, max_tokens=65536)
            else:
                content, usage = call_llm(L.MODEL, [{"role": "user", "content": prompt}])
        except Exception as e:
            print(f"[llm_error] {t} p{p}: {str(e)[:200]}", flush=True)
            res["path"] = "llm_error"
            return res
        with _lock:
            TOK["prompt"] += (usage or {}).get("prompt_tokens", 0)
            TOK["completion"] += (usage or {}).get("completion_tokens", 0)
            TOK["calls"] += 1
        hit = parse_json(content) or {"raw": (content or "")[:4000]}
        if isinstance(hit, dict):
            hit["_usage"] = {"prompt": (usage or {}).get("prompt_tokens"), "completion": (usage or {}).get("completion_tokens")}
        json.dump(hit, open(cp, "w"), ensure_ascii=False, indent=1)
        if (usage or {}).get("reasoning"):  # 保存思考原文，供分析
            open(JD + key + ".reasoning.txt", "w", encoding="utf-8").write(usage["reasoning"])
    if isinstance(hit, dict) and "raw" in hit and "pick" not in hit:
        hit = parse_json(hit["raw"]) or hit
    pick = str(hit.get("pick", "")).strip()[:1] if isinstance(hit, dict) else ""
    if STRUCT and pick not in labels and isinstance(hit, dict) and isinstance(hit.get("violations"), dict):
        vio = {k2: v for k2, v in hit["violations"].items() if k2 in labels and isinstance(v, (int, float))}
        if vio and list(vio.values()).count(min(vio.values())) == 1:
            pick = min(vio, key=vio.get)
    res["prompt_key"] = key
    if pick in labels:
        g = groups[labels.index(pick)]
        res["judge"] = sum(S[i] for i in g) / len(g)
        res["path"] = "judged"
    else:
        res["path"] = "no_pick"
    return res


if __name__ == "__main__":
    R0 = json.load(open(OUT + f"_result_llmtests_pools{len(refs)}.json"))
    base = {(r["t"], r["p"]): r for r in R0}
    jobs = [(r["t"], r["p"]) for r in R0 if r.get("llm_cl") and len(r["llm_cl"]) > 1]
    random.Random(SEED).shuffle(jobs)
    jobs = jobs[:LIMIT]
    with ThreadPoolExecutor(int(os.environ.get("JUDGE_WORKERS", "6"))) as ex:
        J = list(ex.map(judge, jobs))
    for j in J:
        b = base[(j["t"], j["p"])]
        j["hier"] = b["hier"]
        j["final"] = j.get("judge", b["hier"])
        j["any_correct"] = any(b["S"][i] for i in b["tied"])
    json.dump(J, open(OUT + f"_judge_pilot{len(jobs)}{'_think' if THINK else ''}{'_gloss' if GLOSS else ''}{'_ev2' if EV2 else ''}{'_struct' if STRUCT else ''}{f'_b{BUDGET}' if BUDGET else ''}.json", "w"), ensure_ascii=False, indent=1)
    from collections import Counter
    print(f"== {bp.n} 精判 {len(jobs)} 个池  tokens {TOK}  路径 {dict(Counter(j['path'] for j in J))}")
    jd = [j for j in J if j["path"] == "judged"]
    print(f"   这些池：分层 {100 * statistics.mean(j['hier'] for j in J):.1f} → 精判 {100 * statistics.mean(j['final'] for j in J):.1f}")
    up = sum(1 for j in jd if j["judge"] > j["hier"] + 1e-9)
    down = sum(1 for j in jd if j["judge"] < j["hier"] - 1e-9)
    print(f"   精判比分层更好 {up} 个池，更差 {down} 个池，持平 {len(jd) - up - down}；其中池内有正确候选的 {sum(j['any_correct'] for j in J)}")
