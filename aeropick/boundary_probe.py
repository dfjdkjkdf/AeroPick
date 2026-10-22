"""方案 A：关键分支探针（不调用 LLM、不看参考实现/测试）。用法：boundary_probe.py <名> <jsonl> [K]
从 9 个候选代码的比较语句里收集 (变量名, 常数)，构造探针时每个输入有 1/2 概率取匹配常数（等于/略大/略小），
否则照原随机取值。对照：原 harness（K=10）、纯随机 K、关键分支 K。各自做 9 选 1（MISRA 最少后多数投票）。"""
import json, sys, hashlib, random, os, re, collections, threading
sys.path[:0] = ["utils", "eval"]
import fmv_probe as fp, random_probe as rp, point_compute as pc, evidence_judge as ej
from eval import load_jsonl
from record import safe_filename
from concurrent.futures import ThreadPoolExecutor

n, ds = (sys.argv[1], sys.argv[2]) if __name__ == "__main__" else (os.environ["BP_N"], os.environ["BP_DS"])
K = int(sys.argv[3]) if __name__ == "__main__" and len(sys.argv) > 3 else 100
MODEL_DIR = os.environ.get("MODEL_DIR", "deepseek-v4-flash")  # 换模型时用环境变量指定（默认不变）
POOLS = tuple(int(x) for x in os.environ.get("POOLS", "1 2 3").split())
D = f"logs/cache/{MODEL_DIR}/{n}/"
OUT = D + "_bprobe/"
os.makedirs(OUT, exist_ok=True)
refs = {r["task_id"]: r for r in load_jsonl("dataset/" + ds)}
EV = {p: json.load(open(D + f"pool{p}/_eval.json")) for p in POOLS}
h = lambda t, c: hashlib.sha1((t + "\0" + c).encode()).hexdigest()

NUM = r"-?(?:0[xX][0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?)[uUlLfF]*"
CONST = rf"(?:\(\s*\w+\s*\)\s*)?(?:{NUM}|[A-Z_][A-Z0-9_]*)"
LHS = r"((?:[A-Za-z_]\w*(?:\s*(?:->|\.)\s*[A-Za-z_]\w*|\s*\[[^\]\[]*\])*))"
OP = r"(==|!=|<=|>=|<|>)"
CMP1 = re.compile(rf"{LHS}\s*{OP}\s*({CONST})(?![\w(])")
CMP2 = re.compile(rf"(?<![\w)\]])({CONST})\s*{OP}\s*{LHS}")


def last_name(expr):
    names = re.findall(r"[A-Za-z_]\w*", re.sub(r"\[[^\]]*\]", "", expr))
    return names[-1] if names else None


def mine(codes):
    """返回 {变量末级名: set(常数文本)}。"""
    m = collections.defaultdict(set)
    for code in codes:
        code = re.sub(r"/\*.*?\*/|//[^\n]*", "", code, flags=re.S)
        for a, _op, c in CMP1.findall(code):
            if last_name(a) and c.strip() != "NULL":
                m[last_name(a)].add(c.strip())
        for c, _op, a in CMP2.findall(code):
            if last_name(a) and c.strip() != "NULL":
                m[last_name(a)].add(c.strip())
    return m


def variants(c):
    body = re.sub(r"^\(\s*\w+\s*\)\s*", "", c)
    if re.fullmatch(r"-?(?:0[xX][0-9a-fA-F]+|\d+)[uUlL]*", body):
        return [f"({c})", f"({c})+1", f"({c})-1"]
    if re.fullmatch(NUM, body):
        v = float(re.sub(r"[uUlLfF]+$", "", body))
        d = 1e-6 * (abs(v) + 1.0)
        return [repr(v), repr(v + d), repr(v - d)]
    # 宏：类型未知，整数与浮点两种扰动都给
    return [f"({c})", f"({c})+1", f"({c})-1", f"({c})*(1.0+1e-6)+1e-9", f"({c})*(1.0-1e-6)-1e-9"]


_TL = threading.local()
_orig_disc, _orig_cont = rp._emit_discrete_assign, rp._emit_continuous_assign


def _mix(lvalue, dims, orig_lines, indent):
    vals = []
    for c in sorted((getattr(_TL, "mined", None) or {}).get(last_name(lvalue)) or []):
        vals += variants(c)
    if not vals:
        return orig_lines
    out = []
    for idx, line in zip(rp._indices(dims), orig_lines.split("\n") if dims is None else _split_elems(orig_lines, dims)):
        target = f"{lvalue}{rp._subscript(idx)}"
        pv = f"{rp.PREFIX}bp_{rp._unique_suffix(lvalue, idx, 'b')}"
        chain = " else ".join(f"if ({pv} == {i}u) {{ {target} = {v}; }}" for i, v in enumerate(vals))
        out.append(f"{indent}if ({rp.PREFIX}rand_u32() % 2u == 0u) {{ unsigned int {pv} = {rp.PREFIX}rand_u32() % {len(vals)}u; "
                   f"{chain} }} else {{ {line.strip()} }}")
    return "\n".join(out)


def _split_elems(text, dims):
    """原赋值按元素生成：连续量每元素一行；离散量每元素一个 {...} 块（多行）。按元素切开。"""
    lines = text.split("\n")
    if all(not ln.strip().startswith("{") for ln in lines):
        return lines
    blocks, cur = [], []
    for ln in lines:
        cur.append(ln.strip())
        if ln.rstrip().endswith("} }"):
            blocks.append(" ".join(cur))
            cur = []
    return blocks


def emit_disc(lvalue, dims, values, indent="        "):
    o = _orig_disc(lvalue, dims, values, indent)
    if dims is None:
        o = " ".join(x.strip() for x in o.split("\n"))
    return _mix(lvalue, dims, o, indent)


def emit_cont(lvalue, dims, lo, hi, indent="        "):
    return _mix(lvalue, dims, _orig_cont(lvalue, dims, lo, hi, indent), indent)


def _resolve(table, roots, path):
    parts = re.split(r"\.|->", re.sub(r"\[[^\]]*\]", "", path))
    tname = roots.get(parts[0])
    dims = None
    for f in parts[1:]:
        hit = [x for x in table.get(tname, []) if x[1] == f]
        if not hit:
            return None, None
        tname, dims = hit[0][0], hit[0][2]
    return tname, dims


def _expand(table, path, tname, depth, out, budget):
    for ftype, fname, dims in table.get(tname, []):
        p = f"{path}.{fname}"
        if ftype in table:
            if dims is None and depth < 2:
                _expand(table, p, ftype, depth + 1, out, budget)
            elif dims is not None and len(dims) == 1 and depth < 2:
                for i in range(min(dims[0], 4)):
                    _expand(table, f"{p}[{i}]", ftype, depth + 1, out, budget)
            continue
        n = rp._element_count(dims)
        if budget[0] >= n:
            out.append((p, dims, None))
            budget[0] -= n


def observe_outputs(ref, outputs, codes, max_elements=300):
    """可观测性补全：① 规格里的输出若实际是结构体，展开为数值叶子字段；② 候选代码里被赋值的全局结构体/指针参数，
    整体展开其数值字段。只看上下文与候选代码。"""
    prompt = ref.get("prompt") or ""
    table = rp._struct_table(prompt)
    ext = rp.scan_extern_globals(prompt)
    roots = {k: v[0] for k, v in ext.items()}
    roots.update({n: t for n, _p, t in rp.parse_params_typed(ref.get("declaration") or "")})
    out, budget = [], [max_elements]
    for path, dims, rng in outputs:
        tname, _d = _resolve(table, roots, path)
        if tname in table and "[" not in path:
            _expand(table, path, tname, 0, out, budget)
        else:
            out.append((path, dims, rng))
            budget[0] -= rp._element_count(dims)
    written = set()
    for code in codes:
        for m in re.finditer(r"(?m)^\s*([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*(?:\.|->)[^=;]*?[^=!<>]=(?!=)", code):
            written.add(m.group(1))
    for r in sorted(written):
        if r in roots and roots[r] in table:
            _expand(table, r, roots[r], 0, out, budget)
    seen, res = set(), []
    for o in out:
        if o[0] not in seen:
            seen.add(o[0])
            res.append(o)
    return res


def fix_inputs(ref, discrete, continuous):
    """输入路径若实际是结构体（如 VECTOR3），展开为其数值叶子字段，沿用原取值范围/候选值。"""
    prompt = ref.get("prompt") or ""
    table = rp._struct_table(prompt)
    roots = {k: v[0] for k, v in rp.scan_extern_globals(prompt).items()}
    roots.update({n: t for n, _p, t in rp.parse_params_typed(ref.get("declaration") or "")})

    def leaves(path):
        tname, _d = _resolve(table, roots, path)
        if tname in table and "[" not in path:
            out = []
            _expand(table, path, tname, 0, out, [200])
            return [(p, d) for p, d, _ in out]
        return None

    disc, cont = [], []
    for path, dims, vals in discrete:
        lv = leaves(path)
        disc += [(p, d, vals) for p, d in lv] if lv else [(path, dims, vals)]
    for path, dims, lo, hi in continuous:
        lv = leaves(path)
        cont += [(p, d, lo, hi) for p, d in lv] if lv else [(path, dims, lo, hi)]
    return disc, cont


rp._emit_discrete_assign, rp._emit_continuous_assign = emit_disc, emit_cont  # 线程本地无常数时等同原函数


def load_task(t):
    codes, S = [], []
    for p in (1, 2, 3):
        for c in json.load(open(D + f"pool{p}/{safe_filename(t)}.json"))["candidates"]:
            e = EV[p][h(t, c["misra"])]
            codes.append(c["misra"])
            S.append(e["outcome"] == "passed" and e["safe"] in (True, "True"))
    comp = [i for i in range(9) if EV[1 + i // 3][h(t, codes[i])]["outcome"] != "compile_error"] or list(range(9))
    viol = {i: len(EV[1 + i // 3][h(t, codes[i])]["violations"]) for i in comp}
    tied = [i for i in comp if viol[i] == min(viol.values())]
    return codes, S, comp, tied


def build(t, mode):
    """mode: rand / bound / rand_obs / bound_obs（_obs：补全可观测输出）。缓存到 _bprobe/<题>.<mode>K.json"""
    p = OUT + f"{safe_filename(t)}.{mode}{K}.json"
    if os.path.exists(p):
        return json.load(open(p))
    spec = ej.load_json(D + f"_harness/{safe_filename(t)}.spec.json")
    if not spec:
        return None
    codes, _S, comp, _ = load_task(t)
    _TL.mined = mine(codes) if mode.startswith("bound") else {}
    outs = observe_outputs(refs[t], spec["outputs"], codes) if mode.endswith("_obs") else spec["outputs"]
    disc, cont = fix_inputs(refs[t], spec["discrete"], spec["continuous"]) if mode.endswith("_obs") else (
        spec["discrete"], spec["continuous"])
    try:
        harness, ok, info = fp.build_harness_pruned(refs[t], [codes[i] for i in comp], disc, cont, outs, K)
    finally:
        mined, _TL.mined = _TL.mined, {}
    r = {"harness": harness, "ok": ok, "info": info, "mined": {k: sorted(v) for k, v in mined.items()}}
    json.dump(r, open(p, "w"), ensure_ascii=False)
    return r


def same(a, b):
    return set(a) == set(b) and all(pc.match(a[k], b[k]) for k in a)


def clusters(outs, idx):
    cl = []
    for i in idx:
        for c in cl:
            if same(outs[i], outs[c[0]]):
                c.append(i)
                break
        else:
            cl.append([i])
    return sorted(cl, key=len, reverse=True)


def pick_rate(outs, idx, S):
    """多数投票的期望正确率（平局簇间、簇内均匀随机）；无有效输出返回 None。"""
    valid = [i for i in idx if outs.get(i)]
    if len(valid) < 2:
        return None, None
    cl = clusters(outs, valid)
    top = [c for c in cl if len(c) == len(cl[0])]
    return sum(sum(S[i] for i in c) / len(c) for c in top) / len(top), cl


def task(t):
    ref = refs[t]
    codes, S, comp, tied = load_task(t)
    res = {"t": t, "S": S}
    base = sum(S[i] for i in tied) / len(tied)
    old = ej.load_json(D + f"_harness/{safe_filename(t)}.json")
    for name, hr in (("old10", old), ("rand", build(t, "rand")), ("bound", build(t, "bound")),
                     ("rand_obs", build(t, "rand_obs")), ("bound_obs", build(t, "bound_obs"))):
        if not (hr and hr.get("ok")):
            res[name] = base
            res[name + "_cl"] = None
            continue
        outs = {i: fp.probe_candidate(ref, hr["harness"], codes[i]) for i in tied}
        r, cl = pick_rate(outs, tied, S)
        res[name] = base if r is None else r
        res[name + "_cl"] = cl
        if name == "bound_obs":
            res["n_mined"] = sum(len(v) for v in hr["mined"].values())
    return res


if __name__ == "__main__":
    with ThreadPoolExecutor(6) as ex:
        R = list(ex.map(task, sorted(refs)))
    json.dump(R, open(OUT + f"_result{K}.json", "w"), indent=1)
    N = len(R)
    step1 = {r["t"]: r["kind"] for r in json.load(open(
        os.path.dirname(os.path.abspath(__file__)) + f"/misread/{n}/_step1.json"))}
    print(f"== {n}  9 选 1 safe-pass@1（MISRA 最少后多数投票，平局取期望）  K={K}")
    for k, name in (("old10", "原随机探针 K=10"), ("rand", f"纯随机 K={K}"), ("bound", f"关键分支 K={K}"),
                    ("rand_obs", f"纯随机+补全输出 K={K}"), ("bound_obs", f"关键分支+补全输出 K={K}")):
        print(f"   {name:16} {100 * sum(r[k] for r in R) / N:.1f}")
    b3 = [r for r in R if step1.get(r["t"]) == "B3"]
    for k in ("old10", "rand", "bound", "rand_obs", "bound_obs"):
        sep = sum(1 for r in b3 if r[k + "_cl"] and not any(
            any(r["S"][i] for i in c) and not all(r["S"][i] for i in c) for c in r[k + "_cl"]))
        print(f"   B3 {len(b3)} 题中对错被完全分开（{k}）: {sep}，这些题平均挑对率 {100 * sum(r[k] for r in b3) / max(1, len(b3)):.1f}")
    print("   有收集到常数的题", sum(1 for r in R if r.get("n_mined")), "/", N)
