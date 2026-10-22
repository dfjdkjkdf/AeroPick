"""被测模型直接出题（只给输入、不给答案），9 个候选作答，按输出分组投票。
用法：llm_tests.py <名> <jsonl> [题目数上限] [题名前缀过滤]
模型输出 JSON：{"tests":[{"kind":..., "steps":["C 语句…", ...]}]}，每个 step 执行完调用一次被测函数。
harness：每个测试 fork 一个子进程（全局/静态状态天然是初始值）；函数参数声明为同名变量，
指针参数指向零初始化对象（可赋 NULL）；打印的输出字段沿用"补全"后的输出列表（通用，不针对题目）。"""
import json, sys, os, re, hashlib, threading
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["BP_N"], os.environ["BP_DS"] = sys.argv[1], sys.argv[2]
sys.path.insert(0, HERE)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import boundary_probe as bp  # noqa: E402
from boundary_probe import rp, fp, ej, safe_filename, refs, D, OUT  # noqa: E402
import point_compute as pc  # noqa: E402
from llm_generate import call_llm, extract_requirement  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
PREFIX_F = sys.argv[4] if len(sys.argv) > 4 else ""
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
TD = D + "_llmtests/"
os.makedirs(TD, exist_ok=True)
P = rp.PREFIX
_lock = threading.Lock()
TOK = {"prompt": 0, "completion": 0, "calls": 0}

PROMPT = """你是航天 GNC 软件测试工程师。下面是一个 C 函数的需求和上下文。请为它设计约 20 个测试输入，
用来区分不同实现是否正确。**只写输入，不要写期望输出。**

#需求
{requirement}

#函数声明
{decl}

#C代码上下文（类型、宏、全局变量）
```c
{context}
```

#可赋值的输入字段（程序从类型定义自动列出；优先使用这些左值，写法照抄，数组用下标；清单之外的全局变量按上下文中的声明使用）
{fields}

#测试代码的写法
- 函数参数已声明为同名变量：{params_desc}
- 指针参数已指向一个全部清零的有效对象，用 `名字->字段 = 值;` 赋值；要测试空指针就写 `名字 = NULL;`。
- 全局变量直接赋值。未赋值的量都是 0（全局变量为其初始值）。
- 结构体里的指针字段已指向一块可写的数组（至少 16 个元素），用 `x.p->字段`、`x.p[i].字段` 赋值；要测试空指针就写 `x.p = NULL;`。
- 每个测试由若干"步"组成：每一步是一段 C 赋值语句（每行一条语句），执行完这一步后调用一次被测函数。
  单次调用的测试只有一步；需要连续多拍调用的（状态机、计数、滤波、锁存等）写多步，后面的步只写需要改变的量。
- 只能用赋值语句和常量/宏，不要写循环、函数调用或声明。字段清单里 `a[0..3]` 表示下标可取 0 到 3，写测试时用具体下标。

#覆盖要求
正常工况、边界值（阈值上/下、恰好相等）、出错处理（空指针、NaN/无穷、非法参数、越界）、多拍状态转移（按需求描述的条件连续调用到达各个状态/模式）。重点覆盖需求里每个条件分支。

只输出 JSON：{{"tests":[{{"kind":"normal|boundary|error|sequence","steps":["语句1\\n语句2", "..."]}}]}}
"""


def params_full(ref):
    """[(名, 是否指针, 类型, 数组维度 tuple 或 None)]；数组参数如 double w[3]、adcs_vec3_t axes[4]。"""
    decl = ref.get("declaration") or ""
    m = re.search(r"\(([^)]*)\)", decl)
    defines = {k: int(v) for k, v in rp._DEFINE_NUM_RE.findall(ref.get("prompt") or "")}
    out = []
    for n, isp, ty in rp.parse_params_typed(decl):
        dims = None
        if m:
            dm = re.search(rf"\b{re.escape(n)}\s*((?:\[[^\]]*\]\s*)+)", m.group(1))
            if dm:
                ds = [x.strip() for x in re.findall(r"\[([^\]]*)\]", dm.group(1))]
                ds = [int(x) if x.isdigit() else defines.get(x, 16) if x else 16 for x in ds]
                dims = tuple(ds)
        out.append((n, isp, ty, dims))
    return out


def params_desc(ref):
    ps = params_full(ref)
    return "；".join(f"{t} {'*' if isp else ''}{n}{''.join(f'[{d}]' for d in dims) if dims else ''}"
                     for n, isp, t, dims in ps) or "（无参数）"


def global_dims(ref, name):
    text = rp._strip_c_comments(ref.get("prompt") or "")
    m = re.search(rf"(?m)^[^\n(]*\b\w+\s+{re.escape(name)}\s*\[\s*(\w+)\s*\]\s*(?:\[\s*(\w+)\s*\])?\s*(?:=|;)", text)
    if not m:
        return None
    defines = {k: int(v) for k, v in rp._DEFINE_NUM_RE.findall(text)}
    ds = [g for g in m.groups() if g]
    ds = [int(x) if x.isdigit() else defines.get(x) for x in ds]
    return None if any(x is None for x in ds) else tuple(ds)


MAX_FIELD_LINES = 250
_NUM_T = re.compile(r"(double|float|int|unsigned|long|short|char|u?int\d+_t|\w*32|\w*16|\w*8|\w*BOOL\w*|bool)$")


def field_list(ref, codes=None):
    """通用部分（程序生成）：可赋值的输入字段清单。参数全部列出；全局变量只列需求/声明里提到的
    （变量名或其字段名出现在需求里），一个都没提到时列出全部。不看候选代码（各候选池共用同一套题）。"""
    prompt = ref.get("prompt") or ""
    table = rp._struct_table(prompt)
    pf = pointer_fields(ref)
    lines = []

    def leaf(path, ftype, dims, note=""):
        d = "".join(f"[{x}]" for x in dims) if dims else ""
        lines.append(f"{path}{d}  ({ftype}{note})")

    def expand(prefix, tname, depth):
        if len(lines) >= MAX_FIELD_LINES or depth > 3:
            return
        for ftype, fname, dims in table.get(tname, []):
            if ftype in table:
                if dims is None:
                    expand(f"{prefix}{fname}.", ftype, depth + 1)
                elif len(dims) == 1:
                    expand(f"{prefix}{fname}[0..{dims[0] - 1}].", ftype, depth + 1)
            else:
                leaf(prefix + fname, ftype, dims)
        for pt, fname in pf.get(tname, []):
            if pt in table:
                lines.append(f"{prefix}{fname}  (指针字段，指向 {pt} 数组，可设为 NULL)")
                expand(f"{prefix}{fname}->", pt, depth + 1)
            elif _NUM_T.match(pt):
                lines.append(f"{prefix}{fname}[i]  ({pt} 指针字段，指向数组，可设为 NULL)")

    for n, isp, ty, dims in params_full(ref):
        if dims:
            if ty in table and len(dims) == 1:
                expand(f"{n}[0..{dims[0] - 1}].", ty, 0)
            else:
                leaf(n, ty, dims, " 数组参数")
        elif ty in table:
            if isp:
                lines.append(f"{n}  (指针参数，可设为 NULL)")
            expand(f"{n}->" if isp else f"{n}.", ty, 0)
        else:
            lines.append(f"{'*' if isp else ''}{n}  ({ty}{' 指针参数，可设为 NULL' if isp else ''})")
    req = extract_requirement(prompt) + "\n" + (ref.get("declaration") or "")
    gl = sorted(all_globals(ref).items())

    def mentioned(g, ty):
        names = [g] + [f for _t, f, _d in table.get(ty, [])] + [f for _t, f in pf.get(ty, [])]
        return any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(x)}(?![A-Za-z0-9_])", req) for x in names if len(x) >= 2)
    picked = [(g, v) for g, v in gl if mentioned(g, v[0])] if codes is None else \
        [(g, v) for g, v in gl if re.search(rf"\b{re.escape(g)}\b", "\n".join(codes))]
    if codes is None:  # 全部全局变量列得下（≤ MAX_FIELD_LINES 行）就全列，否则只列需求提到的
        n_all = sum(max(1, len(table.get(v[0], [])) + len(pf.get(v[0], []))) for _g, v in gl)
        picked = gl if n_all + len(lines) <= MAX_FIELD_LINES else picked
    for g, (ty, isp) in (picked or gl):
        gd = None if isp else global_dims(ref, g)
        if ty in table:
            if gd and len(gd) == 1:
                expand(f"{g}[0..{gd[0] - 1}].", ty, 0)
            else:
                expand(f"{g}->" if isp else f"{g}.", ty, 0)
        elif _NUM_T.match(ty) and not isp:
            leaf(g, ty, gd)
    if len(lines) > MAX_FIELD_LINES:
        lines = lines[:MAX_FIELD_LINES] + ["…（其余字段从略）"]
    return "\n".join(lines) or "（无）"


def written_global_outputs(ref, codes, budget=300):
    """候选代码里被赋值的全局变量（含结构体数组，前 4 个元素）展开为数值叶子输出。"""
    table = rp._struct_table(ref.get("prompt") or "")
    code_text = "\n".join(codes)
    out = []

    def expand(prefix, tname, depth):
        for ftype, fname, dims in table.get(tname, []):
            if len(out) >= budget:
                return
            if ftype in table:
                if dims is None and depth < 3:
                    expand(f"{prefix}{fname}.", ftype, depth + 1)
                elif dims and len(dims) == 1 and depth < 3:
                    for i in range(min(dims[0], 4)):
                        expand(f"{prefix}{fname}[{i}].", ftype, depth + 1)
            elif _NUM_T.match(ftype):
                out.append((prefix + fname, dims, None))

    for g, (ty, isp) in sorted(all_globals(ref).items()):
        if not re.search(rf"(?m)^\s*{re.escape(g)}\b[^;]*?[^=!<>]=(?!=)", code_text):
            continue
        gd = None if isp else global_dims(ref, g)
        if ty in table:
            if gd and len(gd) == 1:
                for i in range(min(gd[0], 4)):
                    expand(f"{g}[{i}].", ty, 0)
            else:
                expand(f"{g}->" if isp else f"{g}.", ty, 0)
        elif _NUM_T.match(ty) and not isp:
            out.append((g, gd, None))
    return out


def make_prompt(t, codes=None):
    ref = refs[t]
    return PROMPT.format(requirement=extract_requirement(ref["prompt"]), decl=(ref.get("declaration") or "").strip(),
                         context=pc.build_context(ref["prompt"], "full"), params_desc=params_desc(ref),
                         fields=field_list(ref))


def get_tests(t, codes=None):
    """返回测试列表（每题一套，各池共用）；API 出错返回 None（不写缓存，重跑时补）。"""
    prompt = make_prompt(t)
    p = TD + safe_filename(t) + ".json"
    key = hashlib.sha1(prompt.encode()).hexdigest()
    c = ej.load_json(p)
    if c and c.get("key") == key:
        return c["tests"]
    tests = None
    for _ in range(2):
        try:
            content, usage = call_llm(MODEL, [{"role": "user", "content": prompt}])
        except Exception as e:  # 余额不足、网络等：不缓存
            print(f"[llm_error] {t}: {str(e)[:200]}", flush=True)
            return None
        with _lock:
            TOK["prompt"] += (usage or {}).get("prompt_tokens", 0)
            TOK["completion"] += (usage or {}).get("completion_tokens", 0)
            TOK["calls"] += 1
        m = re.search(r"\{.*\}", content or "", re.S)
        try:
            tests = json.loads(m.group(0))["tests"]
            break
        except Exception:
            tests = None
    json.dump({"key": key, "tests": tests or []}, open(p, "w"), ensure_ascii=False, indent=1)
    return tests or []


_PF_RE = re.compile(r"(?:const\s+)?(\w+)\s*\*\s*(?:const\s+)?(\w+)\s*;")


def pointer_fields(ref):
    """{结构体类型: [(指向类型, 字段名)]}：结构体里的指针字段（_struct_table 不含指针字段）。"""
    text = rp._strip_c_comments(ref.get("prompt") or "")
    out = {}
    for tag, body, names in rp._STRUCT_DEF_RE.findall(text):
        f = []
        for decl in body.split(";"):
            m = re.match(r"\s*(?:const\s+|volatile\s+)*(\w+)\s+(.*)$", decl, re.S)
            if not m or m.group(1) == "char":
                continue
            for item in m.group(2).split(","):
                pm = re.fullmatch(r"\s*\*\s*(?:const\s+)?(\w+)\s*", item)
                if pm:
                    f.append((m.group(1), pm.group(1)))
        for nm in [tag] + [x.strip().lstrip("*").strip() for x in names.split(",")]:
            if nm and f:
                out[nm] = f
    return out


def all_globals(ref):
    """extern 与文件作用域 static/普通 结构体变量：{名: (类型, 是否指针)}。"""
    text = rp._strip_c_comments(ref.get("prompt") or "")
    table = rp._struct_table(text)
    g = dict(_orig_scan(ref.get("prompt") or ""))
    top, depth = [], 0  # 只看文件作用域（花括号外）的声明，排除函数体内的局部变量
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            top.append(" ")
        elif depth == 0:
            top.append(ch)
    text = "".join(top)
    for m in re.finditer(r"(?m)(?:^|(?<=;))\s*(?:static\s+|extern\s+)?(?:volatile\s+)?(\w+)\s+(\*?)\s*(\w+)\s*(?:\[[^\]]*\]\s*)*(?:=[^;]*)?;", text):
        if m.group(1) in table and m.group(3) not in g:
            g[m.group(3)] = (m.group(1), bool(m.group(2)))
    return g


_orig_scan = rp.scan_extern_globals
rp.scan_extern_globals = lambda prompt: all_globals({"prompt": prompt})  # 本脚本内：static 全局也算（用于补全输出）


def pointer_backing(ref):
    """返回 [(模型写法前缀 'root.F' 或 'p->F', 指针左值, 指向类型, 后备对象名)]，覆盖参数与全局结构体的一级指针字段。"""
    table = rp._struct_table(ref.get("prompt") or "")
    pf = pointer_fields(ref)
    res = []
    for n, isp, ty in rp.parse_params_typed(ref.get("declaration") or ""):
        for pt, f in pf.get(ty, []):
            if pt in table or re.fullmatch(r"(double|float|int|unsigned|long|short|u?int\d+_t|\w+32|\w+16|\w+8)", pt):
                acc = f"{n}->{f}" if isp else f"{n}.{f}"
                res.append((acc, f"{P}obj_{n}.{f}" if isp else f"{n}.{f}", pt, f"{P}pf_{n}_{f}"))
    for g, (ty, isp) in all_globals(ref).items():
        if isp:
            continue
        for pt, f in pf.get(ty, []):
            if pt in table or re.fullmatch(r"(double|float|int|unsigned|long|short|u?int\d+_t|\w+32|\w+16|\w+8)", pt):
                res.append((f"{g}.{f}", f"{g}.{f}", pt, f"{P}pf_{g}_{f}"))
    return res


def rewrite_stmt(stmt, backing):
    """'X.F->a = v' / 'X.F[0] = v' 改写为对后备对象赋值；'X.F = NULL' 保持（测试空指针）。"""
    for acc, _lv, _pt, obj in backing:
        a = re.escape(acc).replace(r"\.", r"\s*\.\s*").replace(r"\-\>", r"\s*->\s*").replace("->", r"\s*->\s*")
        stmt = re.sub(a + r"\s*->\s*", obj + "[0].", stmt)
        stmt = re.sub(a + r"\s*\[", obj + "[", stmt)
        stmt = re.sub(r"\*\s*\(?\s*" + a + r"\s*\)?(?=\s*=)", obj + "[0]", stmt)
    return stmt


_ASSIGN_RE = re.compile(r"^(.+?)\s*(?<![=!<>+\-*/%&|^])=(?!=)\s*(.+?);?$", re.S)


def byte_assign(stmt):
    """'LHS = RHS;' -> 按字节写入（结构体字段声明为 const 时也能赋值）；其他语句原样返回。"""
    m = _ASSIGN_RE.match(stmt.strip())
    if not m:
        return stmt
    lhs, rhs = m.group(1).strip(), m.group(2).strip()
    return f"{{ __typeof__({lhs}) {P}v = ({rhs}); memcpy((void *)&({lhs}), &{P}v, sizeof({P}v)); }}"


def build_source(ref, tests, outs):
    params = rp.parse_params_typed(ref.get("declaration") or "")
    pnames = {n for n, _, _ in params}
    ext_ptr = {n for n, (_t, isp) in rp.scan_extern_globals(ref.get("prompt") or "").items() if isp}
    func_m = re.search(r"(\w+)\s*\(", ref.get("declaration") or "")
    if not func_m:
        raise ValueError("函数声明解析失败")
    func = func_m.group(1)
    pdims = {n: d for n, _i, _t, d in params_full(ref)}
    ret_tokens = [x for x in re.findall(r"[A-Za-z_]\w*", (ref.get("declaration") or "")[:func_m.start()])
                  if x not in ("static", "inline", "extern", "const", "volatile")]
    ret_type = ret_tokens[-1] if ret_tokens else "void"
    has_ret = ret_type != "void" and ret_type not in rp._struct_table(ref.get("prompt") or "")

    def lv(path):  # 输出路径 -> 打印用左值（指针参数通过其对象访问，参数为 NULL 时仍可打印）
        root, _, rest = path.partition(".")
        if root in pnames:
            isp = next(i for n, i, _ in params if n == root)
            base = f"{P}obj_{root}" if isp else root
            return base + (("." + rest) if rest else "")
        if rest and root in ext_ptr:
            return f"{root}->{rest}"
        return path

    decls = []
    for n, isp, ty in params:
        if pdims.get(n):
            decls.append(f"            static {ty} {n}{''.join(f'[{d}]' for d in pdims[n])}; memset({n}, 0, sizeof({n}));")
        elif isp:
            decls.append(f"            static {ty} {P}obj_{n}; {ty} *{n} = &{P}obj_{n};")
        else:
            decls.append(f"            {ty} {n}; memset(&{n}, 0, sizeof({n}));")
    backing = pointer_backing(ref)
    for _acc, lvp, pt, obj in backing:
        decls.append(f"            static {pt} {obj}[16]; {lvp} = {obj};")  # 指针字段可能指向数组
    prints = []
    for path, dims, _r in outs:
        for idx in rp._indices(dims):
            tgt = f"{lv(path)}{rp._subscript(idx)}"
            label = f"{path}{''.join(f'[{i}]' for i in idx)}"
            prints.append(f'                printf("{label}: %.10g\\n", (double)({tgt})); /*@P*/')
    call_args = ", ".join(n for n, _, _ in params)
    call = f"{P}ret = {func}({call_args});" if has_ret else f"{func}({call_args});"
    body = []
    for ti, test in enumerate(tests[:30]):
        steps = [s for s in (test.get("steps") or []) if isinstance(s, str)][:10]
        if not steps:
            continue
        body.append(f"    {{ pid_t {P}pid = fork(); if ({P}pid == 0) {{ alarm(2);")
        body += decls
        if has_ret:
            body.append(f"            {ret_type} {P}ret; memset(&{P}ret, 0, sizeof({P}ret));")
        for si, st in enumerate(steps):
            body.append(f'                printf("## %d\\n", {ti * 10 + si});')
            for stmt in re.split(r"\n|(?<=;)\s+(?=\S)", st):
                stmt = stmt.strip()
                if stmt and not stmt.startswith("//"):
                    stmt = byte_assign(rewrite_stmt(stmt, backing))
                    body.append(f"                {stmt if stmt.endswith(';') else stmt + ';'} /*@A*/")
            body.append(f"                {call}")
            body += prints
            if has_ret:
                body.append(f'                printf("__ret__: %.10g\\n", (double)({P}ret)); /*@P*/')
        body.append(f"            fflush(stdout); _exit(0); }}")
        body.append(f"        int {P}st = 0; waitpid({P}pid, &{P}st, 0);")
        body.append(f'        if (!WIFEXITED({P}st)) printf("__crash__: 1\\n"); }}')
    src = "\n".join([
        "#include <string.h>", "#include <stdio.h>", "#include <unistd.h>", "#include <sys/wait.h>",
        "int main(void) {", "    setvbuf(stdout, NULL, _IONBF, 0);", *body, "    return 0;", "}"])
    test_has_own_close = (ref.get("test") or "").lstrip().startswith("}")
    return ("}\n" + src) if test_has_own_close else ("\n" + src)


def load_pool(t, p):
    codes, S = [], []
    for c in json.load(open(D + f"pool{p}/{safe_filename(t)}.json"))["candidates"]:
        e = bp.EV[p][bp.h(t, c["misra"])]
        codes.append(c["misra"])
        S.append(e["outcome"] == "passed" and e["safe"] in (True, "True"))
    comp = [i for i in range(len(codes)) if bp.EV[p][bp.h(t, codes[i])]["outcome"] != "compile_error"] or list(range(len(codes)))
    viol = {i: len(bp.EV[p][bp.h(t, codes[i])]["violations"]) for i in comp}
    tied = [i for i in comp if viol[i] == min(viol.values())]
    return codes, S, comp, tied


def prune_loop(ref, cands, harness, max_rounds=15):
    pruned = 0
    for cand in cands:
        for _ in range(max_rounds):
            r = fp.run_probe_plain(ref, cand, harness, timeout=60.0)
            if r["outcome"] in ("ran", "runtime_error"):
                return harness, True, pruned
            if r["outcome"] != "compile_error":
                break
            _g, h_start = rp.region_line_ranges(ref, cand, harness)
            lines = harness.split("\n")
            bad = {int(m.group(1)) - h_start for m in fp._GCC_ERR_LINE_RE.finditer(r.get("stderr", ""))}
            bad = {i for i in bad if 0 <= i < len(lines) and lines[i].rstrip().endswith(fp._PRUNABLE)}
            if not bad:
                break
            pruned += len(bad)
            harness = "\n".join(ln for i, ln in enumerate(lines) if i not in bad)
    return harness, False, pruned


def pool_outputs(t, codes):
    spec = ej.load_json(D + f"_harness/{safe_filename(t)}.spec.json") or {"outputs": [], "discrete": [], "continuous": []}
    outs = bp.observe_outputs(refs[t], spec["outputs"], codes)
    outs += [o for o in rp.auto_output_specs(refs[t]) if o[0] not in {x[0] for x in outs}]
    outs += [o for o in written_global_outputs(refs[t], codes) if o[0] not in {x[0] for x in outs}]
    return spec, outs


def build_rand(t, p, codes, comp):
    """随机+补全探针（K=100），只用本池候选。"""
    path = OUT + f"{safe_filename(t)}.p{p}.rand_obs100.json"
    c = ej.load_json(path)
    if c:
        return c
    spec, outs = pool_outputs(t, codes)
    if not spec.get("prompt_sha1"):
        c = {"harness": "", "ok": False}
    else:
        disc, cont = bp.fix_inputs(refs[t], spec["discrete"], spec["continuous"])
        harness, ok, _info = fp.build_harness_pruned(refs[t], [codes[i] for i in comp], disc, cont, outs, 100)
        c = {"harness": harness, "ok": ok}
    json.dump(c, open(path, "w"), ensure_ascii=False)
    return c


def build_llm(t, p, codes, comp):
    path = OUT + f"{safe_filename(t)}.p{p}.llmtests.json"
    tests = get_tests(t)
    if tests is None:
        return {"harness": "", "ok": False, "n_tests": 0, "pruned": 0, "llm_error": True}
    key = hashlib.sha1(json.dumps(tests, ensure_ascii=False).encode()).hexdigest()
    c = ej.load_json(path)
    if c and c.get("key") == key:
        return c
    _spec, outs = pool_outputs(t, codes)
    try:
        src = build_source(refs[t], tests, outs) if tests else None
    except ValueError:
        src = None
    harness, ok, pruned = "", False, 0
    if src:
        harness, ok, pruned = prune_loop(refs[t], [codes[i] for i in comp],
                                         fp._splice_preamble(src, fp._test_preamble(refs[t])))
    c = {"key": key, "harness": harness, "ok": ok, "n_tests": len(tests), "pruned": pruned}
    json.dump(c, open(path, "w"), ensure_ascii=False)
    return c


def pool_task(job):
    t, p = job
    ref = refs[t]
    codes, S, comp, tied = load_pool(t, p)
    res = {"t": t, "p": p, "S": S, "tied": tied}
    base = sum(S[i] for i in tied) / len(tied)
    res["misra_random"] = base
    hr, hl = build_rand(t, p, codes, comp), build_llm(t, p, codes, comp)
    o_r = {i: fp.probe_candidate(ref, hr["harness"], codes[i]) for i in tied} if hr["ok"] else {}
    o_l = {i: fp.probe_candidate(ref, hl["harness"], codes[i], timeout=120.0) for i in tied} if hl["ok"] else {}
    res.update(ok=hl["ok"], n_tests=hl["n_tests"], pruned=hl["pruned"], llm_error=hl.get("llm_error", False))
    for name, outs in (("rand_obs", o_r), ("llm", o_l)):
        r, cl = bp.pick_rate(outs, tied, S)
        res[name] = base if r is None else r
        res[name + "_cl"] = cl
    if res["rand_obs_cl"] and res["llm_cl"]:  # 分层：先随机+补全取最大簇，再按模型出的题细分
        top = [c for c in res["rand_obs_cl"] if len(c) == len(res["rand_obs_cl"][0])]
        vals = []
        for c in top:
            subs = [s for s in ([i for i in u if i in c] for u in res["llm_cl"]) if s]
            m = max(len(s) for s in subs)
            best = [s for s in subs if len(s) == m]
            vals.append(sum(sum(S[i] for i in s) / len(s) for s in best) / len(best))
        res["hier"] = sum(vals) / len(vals)
    else:
        res["hier"] = res["rand_obs"]
    # 是否仍有分歧（需精判）：本池 tied 内有 ≥2 组
    res["divergent"] = bool(res["llm_cl"] and len(res["llm_cl"]) > 1) or bool(res["rand_obs_cl"] and len(res["rand_obs_cl"]) > 1)
    return res


if __name__ == "__main__":
    import statistics
    ts = [t for t in sorted(refs) if t.startswith(PREFIX_F)][:LIMIT]
    with ThreadPoolExecutor(int(os.environ.get("TEST_WORKERS", "6"))) as ex:
        list(ex.map(get_tests, ts))  # 先出题（每题一次）
        R = list(ex.map(pool_task, [(t, p) for t in ts for p in bp.POOLS]))
    json.dump(R, open(OUT + f"_result_llmtests_pools{'_' + PREFIX_F if PREFIX_F else ''}{len(ts)}.json", "w"), indent=1)
    print(f"== {bp.n}  {len(ts)} 题 × 3 池  tokens {TOK}")
    for k, name in (("misra_random", "MISRA最少+随机"), ("rand_obs", "随机+补全 K=100"), ("llm", "模型出题"),
                    ("hier", "分层(随机→模型出题)")):
        v = [100 * statistics.mean(r[k] for r in R if r["p"] == p) for p in bp.POOLS]
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"   {name:18} {statistics.mean(v):.1f} ± {sd:.1f}   {[round(x, 1) for x in v]}")
    print("   API 出错", sum(r["llm_error"] for r in R) // 3, " 模型出题 harness 可用", sum(r["ok"] for r in R), "/", len(R),
          " 平均题数", round(sum(r["n_tests"] for r in R) / max(1, len(R)), 1), " 删除语句", sum(r["pruned"] for r in R))
    print("   仍有分歧(需精判)的池", sum(r["divergent"] for r in R), "/", len(R))
