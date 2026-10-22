#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FMV（Functional Majority Voting，见 arXiv:2604.15618）选拔机制用到的探针
harness构建 + 软投票——不依赖隐藏测试：K个测试输入由 random_probe.extract_probe_spec
从需求文本推出的输入取值范围随机采样生成（跟隐藏测试完全独立），候选之间互相比对
在这K个输入上算出的输出是否一致来投票，不需要任何标准答案。

跟 baselines/llm_oracle.py 的Tier1差分测试不是一回事：Tier1是"候选 vs 一份离线用
隐藏测试验证过的参考实现"做差分，本质上需要提前有一个已知正确的答案存在；这里是
"候选互相投票"，谁都不需要预先假定对，只要没编译/没跑起来的候选被剔除、剩下的
候选里占多数的那派更可能是对的。
"""
import os
import re
import subprocess
import tempfile

import random_probe
from llm_generate import call_llm, extract_c_code


# ---------------------------------------------------------------------------
# harness 构建 + 现场生成偶尔出现的bug自动修复
# ---------------------------------------------------------------------------

def _test_preamble(ref):
    """有些题的候选会调用"跨任务共享函数"（比如某些p1题调用AttiDifference，
    那其实是另一道题自己的测试目标，为避免多题各自生成候选时重复定义冲突，没打进
    项目静态库），真正的隐藏测试(ref['test'])会在"int main"之前内嵌一份该函数的
    参考实现来单独解决自己的链接问题。我们自己现场拼的探针harness不知道这些，
    直接编译会在链接期报undefined reference——不是候选代码错，是探针harness缺了
    这段内嵌桩函数。这里把ref['test']里"int main"之前的部分摘出来，拼接进
    我们自己的harness里（fmv机制本身完全不用ref['test']里"int main"之后的部分，
    那才是真正的隐藏测试断言）。"""
    test = ref.get("test") or ""
    idx = test.find("int main")
    if idx == -1:
        return ""
    preamble = test[:idx]
    if preamble.startswith("}"):
        preamble = preamble[1:]
    return preamble


def _splice_preamble(harness, preamble):
    if not preamble.strip():
        return harness
    if harness.startswith("}"):
        return "}" + preamble + harness[1:]
    return preamble + harness


_PTR_DOT_ERR_RE = re.compile(r"'\([^']*?\)&(\w+)'\s+is a pointer; did you mean to use '->'\?")


def _autofix_pointer_dot(harness, stderr):
    """extract_probe_spec/build_harness_source 现场生成的harness，偶尔会把实际是
    指针类型的extern全局变量按值类型写成"var.field"（应该是"var->field"）——这是
    harness生成阶段对指针类型识别不准的bug，不是候选代码的问题。gcc对这种写法会报
    "'(T *)&var' is a pointer; did you mean to use '->'?"，从报错里抠出变量名，
    把harness里这个变量的"var."全部替换成"var->"再重试编译。"""
    bad_vars = set(_PTR_DOT_ERR_RE.findall(stderr))
    if not bad_vars:
        return harness, False
    fixed = harness
    for v in bad_vars:
        fixed = re.sub(rf"\b{re.escape(v)}\.", f"{v}->", fixed)
    return fixed, True


_CONST_LOCATION_ERR_RE = re.compile(r"assignment of read-only location '([^']+)'")
_CONST_MEMBER_ERR_RE = re.compile(r"assignment of read-only member '([^']+)'")


def _autofix_const_assign(harness, stderr):
    """有些函数的输入参数类型本身在项目头文件里带了const修饰，harness生成时
    没识别到这层，直接写"X.field = 随机值;"，这在C里是给只读左值赋值，编译期
    直接报错。gcc对数组元素和标量成员报的错文案不一样：数组元素给的是完整表达式
    （"read-only location 'X.field[0]'"），标量成员只给裸成员名（"read-only
    member 'field'"，不带前缀），后者要在harness里搜"任意前缀.field ="把完整
    表达式找回来。找到完整表达式后统一把"EXPR = "改写成
    "*(typeof(EXPR + 0)*)&EXPR = "：typeof(EXPR)本身还是带const的，直接cast没用，
    但EXPR + 0是算术表达式产生的右值，类型限定符会被丢掉，typeof(EXPR + 0)拿到的
    才是去掉const之后的裸类型。用(?<!&)防止对已经修过的那一份重复再套一层。"""
    bad_exprs = set(_CONST_LOCATION_ERR_RE.findall(stderr))
    for member in _CONST_MEMBER_ERR_RE.findall(stderr):
        for m in re.finditer(r"[\w.]+\." + re.escape(member) + r"(?:\[\d+\])?", harness):
            bad_exprs.add(m.group(0))
    if not bad_exprs:
        return harness, False
    fixed = harness
    changed = False
    for expr in bad_exprs:
        pattern = r"(?<!&)" + re.escape(expr) + r"(\s*)="
        new_fixed, n = re.subn(pattern, f"*(typeof({expr} + 0)*)&{expr}\\1=", fixed)
        if n:
            fixed = new_fixed
            changed = True
    return fixed, changed


def run_probe_plain(ref, generation, harness, timeout=15.0):
    """跟 random_probe.run_probe_ubsan 逻辑一致，但不加 -fsanitize=undefined——
    有些题（比如调用跨任务共享函数的题）加了UBSan之后反而对静态库里的符号报
    undefined reference，同一份代码不加sanitizer编译链接正常，是这个环境下UBSan
    编译path本身的问题，跟候选代码对错无关。FMV只需要每个候选在K个探针用例上的
    输出值做投票，不需要UBSan的缺陷信号，换成不带sanitizer的普通编译更省事也
    更稳。"""
    source = random_probe.build_probe_source(ref, generation, harness)
    with tempfile.TemporaryDirectory(prefix="fmv_probe_") as workdir:
        with open(os.path.join(workdir, "case.c"), "w", encoding="utf-8") as f:
            f.write(source)
        try:
            compiled = random_probe.docker_run(workdir, random_probe._compile_cmd(ref.get("project")), timeout)
        except subprocess.TimeoutExpired:
            return {"outcome": "timeout", "stdout": "", "stderr": ""}
        # 其他异常（如 docker 守护进程不可用的 RuntimeError）必须抛出：曾因这里吞异常，
        # docker 宕机期间所有探针都被记成"跑不起来"，进而触发 LLM 反复修 harness、裁判全量调用。
        if compiled.returncode != 0:
            return {"outcome": "compile_error", "stdout": "", "stderr": compiled.stderr or compiled.stdout}
        try:
            ran = random_probe.docker_run(workdir, "./case.out", timeout)
        except subprocess.TimeoutExpired:
            return {"outcome": "timeout", "stdout": "", "stderr": ""}
        if ran.returncode != 0:
            return {"outcome": "runtime_error", "stdout": ran.stdout, "stderr": ran.stderr}
        return {"outcome": "ran", "stdout": ran.stdout, "stderr": ran.stderr}


_HARNESS_FIX_PROMPT = """下面这段C代码是自动生成的"探针harness"，用来在函数外部构造测试输入、
调用目标函数、打印输出字段，给多份候选代码做行为比对用（不是隐藏测试，没有标准答案）。
编译这段代码报了下面的错误，请只修复这个报错本身对应的那一行/那几行，不要改动其余部分的
逻辑结构（不要删掉"打印输出"或"赋随机值"这些语句所在的循环结构，只把类型不对的赋值方式
改对，比如整体赋值给结构体类型的地方要改成逐个成员赋值，或者改成memset+按需成员赋值）。

#编译报错
{stderr}

#当前harness代码（这段代码会直接接在候选函数体后面，所以从"}}"开头是正常的，不要删掉）
```c
{harness}
```

请输出修好之后的完整harness代码，用```c代码块包裹，不要输出任何解释文字，不要省略任何一行。
"""


def _llm_fix_harness(ref, harness, model, task_id, log=None, max_rounds=3, tokens_acc=None):
    """regex自动修复搞不定的编译错误，交给LLM像改patch一样看着gcc报错原文去改——
    跟候选代码的_compile_repair是同一个思路，只是这次改的是我们自己现场生成的
    探针harness，不是候选代码。regex修复只覆盖了"指针写成点号""const字段赋值"
    这两类我们提前遇到过的错误，碰到没见过的错误类型（比如结构体数组元素被当
    标量赋值）就没辙；LLM能读懂任意报错文本去对症下药，覆盖面不受限于我们
    提前想到过哪几类错误。返回 (harness, ok)。tokens_acc不为None时把这里每次
    call_llm花的token累加进去（key="harness"），方便按阶段统计消耗。"""
    canonical = ref.get("canonical_solution") or ""
    current = harness
    for round_i in range(max_rounds):
        r = run_probe_plain(ref, canonical, current, timeout=15.0)
        if r["outcome"] in ("ran", "runtime_error"):
            return current, True
        stderr = (r.get("stderr") or r.get("stdout") or "")[:2000]
        prompt = _HARNESS_FIX_PROMPT.format(stderr=stderr, harness=current)
        content, usage = call_llm(model, [{"role": "user", "content": prompt}])
        if tokens_acc is not None:
            tokens_acc["harness"] += usage["total_tokens"]
        fixed = extract_c_code(content)
        if not fixed.strip():
            break
        if log:
            log(f"[{task_id}] LLM修复探针harness（第{round_i + 1}轮）")
        current = fixed
    r = run_probe_plain(ref, canonical, current, timeout=15.0)
    return current, r["outcome"] in ("ran", "runtime_error")


def _safe_fallback_harness(ref, outputs, k, log=None, task_id=None):
    """终极兜底：放弃"给每个输入字段各自赋合理范围内的随机值"这个精细目标，
    只保留"清零输出字段避免残留陈旧值"（build_harness_source原本就有的行为，
    见其docstring里某内部任务那个真实翻车案例）+ 调用目标函数 + 打印输出字段——不
    传discrete/continuous（即不生成任何"往具体字段赋值"的代码），这类赋值正是
    前面两层修复想解决的错误来源。没有任何按类型赋值的代码就不可能有类型不
    匹配的编译错误，唯一的代价是K个探针用例之间输入没有差异（全是默认/清零
    状态），投票信号比正常版本弱，但保证这道题的FMV机制至少能跑起来参与
    投票，不会因为harness生成失败就完全退化成"矇眼随便选一个"。"""
    harness, _output_ranges = random_probe.build_harness_source(
        ref, [], [], outputs, k=k, seed=random_probe.DEFAULT_SEED
    )
    preamble = _test_preamble(ref)
    harness = _splice_preamble(harness, preamble)
    if log:
        log(f"[{task_id}] 前两层修复都失败，启用兜底harness（不随机赋值输入字段，"
            f"只清零输出字段+调用+打印，K个用例之间输入没有差异，投票信号会比正常情况弱）")
    return harness


def _validate_and_fix_harness(ref, harness, task_id, model, k, log=None, max_tries=4, tokens_acc=None):
    """三层保险，从省钱到兜底依次尝试，只要还没通过就往下一层退：
    1) 正则自动修复我们已知的两类bug（指针写成点号、const字段直接赋值）；
    2) 交给LLM看gcc报错原文去改，覆盖面不受限于我们提前想到过哪几类错误；
    3) 兜底harness——不做任何按类型的字段赋值，保证编译一定能过。
    正常情况下用canonical_solution（一定是结构正确的C代码）当一次性的编译
    探针来验证；返回 (harness, ok, tier)，tier标记最终是哪一层过的，ok目前
    恒为True（第3层理论上不可能编译失败），保留这个返回值是为了防御性——
    万一第3层因为某种没预见到的原因也失败了，调用方仍然能拿到明确信号退化
    成只用MISRA选拔，而不是把一份跑不起来的harness当成能用的。tokens_acc
    不为None时把这一路（第2层LLM修复+第3层兜底harness需要重新提取spec）
    花的token都累加进去（key="harness"）。"""
    canonical = ref.get("canonical_solution") or ""
    for attempt in range(max_tries):
        r = run_probe_plain(ref, canonical, harness, timeout=15.0)
        if r["outcome"] in ("ran", "runtime_error"):
            return harness, True, 1
        stderr = r.get("stderr", "")
        fixed, changed = _autofix_pointer_dot(harness, stderr)
        if not changed:
            fixed, changed = _autofix_const_assign(harness, stderr)
        if not changed:
            break
        if log:
            log(f"[{task_id}] 探针harness正则自动修复（第{attempt + 1}次）")
        harness = fixed

    fixed, ok = _llm_fix_harness(ref, harness, model, task_id, log=log, tokens_acc=tokens_acc)
    if ok:
        return fixed, True, 2

    if log:
        log(f"[{task_id}] 正则修复+LLM修复都没能让探针harness编译通过，改用兜底harness")
    _discrete, _continuous, outputs, usage = random_probe.extract_probe_spec(model, ref)
    if tokens_acc is not None:
        tokens_acc["harness"] += usage["total_tokens"]
    fallback = _safe_fallback_harness(ref, outputs, k, log=log, task_id=task_id)
    r = run_probe_plain(ref, canonical, fallback, timeout=15.0)
    if r["outcome"] in ("ran", "runtime_error"):
        return fallback, True, 3
    if log:
        log(f"[{task_id}] 兜底harness理论上不该失败，但确实失败了，放弃FMV投票，退化成只用MISRA信号："
            f"{(r.get('stderr') or '')[:300]}")
    return fallback, False, 3


def get_harness(task_id, ref, oracle_lib, model, k, log=None, tokens_acc=None):
    """返回 (harness, ok)。优先复用oracle_lib里离线预先构建、验证过的harness
    （K可能跟调用方要的不一样，没关系——fmv_select只看每个候选输出里前K个探针
    case的键，不要求harness恰好按这个K构建）；没有缓存时现场调用
    extract_probe_spec+build_harness_source构建，并走三层修复兜底。tokens_acc
    不为None时把这里所有LLM调用（extract_probe_spec + 后续修复用到的）花的
    token都累加进去（key="harness"）——覆盖"harness阶段消耗token"的完整口径，
    不只是修复那一步。"""
    preamble = _test_preamble(ref)
    entry = (oracle_lib or {}).get(task_id)
    if entry and entry.get("tier1_harness"):
        harness = _splice_preamble(entry["tier1_harness"], preamble)
    else:
        discrete, continuous, outputs, usage = random_probe.extract_probe_spec(model, ref)
        if tokens_acc is not None:
            tokens_acc["harness"] += usage["total_tokens"]
        harness, _output_ranges = random_probe.build_harness_source(
            ref, discrete, continuous, outputs, k=k, seed=random_probe.DEFAULT_SEED
        )
        harness = _splice_preamble(harness, preamble)
    harness, ok, _tier = _validate_and_fix_harness(ref, harness, task_id, model, k, log=log, tokens_acc=tokens_acc)
    return harness, ok


def probe_candidate(ref, harness, code, timeout=15.0):
    """返回探针输出 {"case_idx::字段名": 值} 的字典；候选编译失败/探针跑不起来
    时返回 None（FMV投票里直接排除，不能当0分——那是满分，会让"根本没测出来"
    的候选反而显得最完美）。"""
    r = run_probe_plain(ref, code, harness, timeout=timeout)
    if r["outcome"] not in ("ran", "runtime_error"):
        return None
    return random_probe.parse_probe_output(r["stdout"])


# ---------------------------------------------------------------------------
# 软功能投票（Functional Majority Voting）
# ---------------------------------------------------------------------------

def case_idx(key):
    case_str, _, _field = key.partition("::")
    try:
        return int(case_str)
    except ValueError:
        return -1


def fmv_select(parsed_list, valid_mask, k_cases, rng):
    """软功能投票：候选i的得分 = sum_{j != i} sum_{k in 前k_cases个探针用例}
    I(候选i和候选j在这个用例这个字段上的值一致)——不需要标准答案，纯粹是候选
    互相比对，谁跟"大多数"一致谁得分高。

    无效候选（探针没跑起来，parsed为None/{}）直接排除，不参与投票也不会被选中。
    出现打平（多个候选并列最高分，最常见于"这几个候选其实独立算出了同一个正确
    答案"）时，不能固定选"下标最小"那个——真实验证过这样做在某道题上恰好选中
    了打平里唯一错的那个候选（打平的4个候选里3个真值正确、1个错误，固定选
    最小下标的兜底规则蒙对/蒙错纯属运气）。这里改成在打平的候选里用传入的rng
    随机选一个：如果打平的大多数确实是对的，随机选中对的的概率也更高；如果
    打平的候选之间其实完全等价（比如只是写法不同、行为处处一致），随机选哪个
    都无所谓。

    返回 (被选中的候选下标或None, 各候选得分列表, 打平的候选下标列表, 诊断信息字典)。
    诊断信息字典包含：
      n_requested — 传进来的候选总数（含无效的）
      n_valid — 探针真正跑起来、参与投票的候选数（可能小于n_requested，比如
                有候选虽然编译过了但探针运行崩溃/超时）
      k_requested — 调用方要求的探针用例数k_cases
      k_effective_min/k_effective_max — 每份有效候选实际输出了多少个不同的
                case编号（可能小于k_requested，比如候选跑到第3个用例就自己
                崩溃了，第4~9个用例根本没打印出来）；min是"所有有效候选都
                覆盖到的用例数"（投票真正能公平比较到的范围），max是"跑得
                最远的那份候选覆盖到的用例数"
    全部候选都无效时返回 (None, scores全-1, [], 诊断信息字典)。"""
    n = len(parsed_list)
    scores = [-1] * n
    valid_idx = [i for i in range(n) if valid_mask[i]]

    def _case_counts():
        # 跟下面真正参与打分的filtered用同一个口径——只数case_idx在
        # [0, k_cases)范围内的（哪怕候选/缓存的harness实际输出了更多，比如
        # 某内部任务复用的离线harness是按k=500构建的，调用方这次只要求k_cases=10，
        # 就该按10算，不能把没参与投票的那490个也算进"实际K"里）。
        counts = []
        for i in valid_idx:
            keys = (parsed_list[i] or {}).keys()
            counts.append(len(set(case_idx(k) for k in keys if 0 <= case_idx(k) < k_cases)))
        return counts

    if not valid_idx:
        diag = {"n_requested": n, "n_valid": 0, "k_requested": k_cases,
                "k_effective_min": 0, "k_effective_max": 0}
        return None, scores, [], diag

    case_counts = _case_counts()
    diag = {
        "n_requested": n, "n_valid": len(valid_idx), "k_requested": k_cases,
        "k_effective_min": min(case_counts) if case_counts else 0,
        "k_effective_max": max(case_counts) if case_counts else 0,
    }

    filtered = []
    for i in range(n):
        if not valid_mask[i]:
            filtered.append({})
            continue
        filtered.append({k: v for k, v in (parsed_list[i] or {}).items() if case_idx(k) < k_cases})

    for i in valid_idx:
        s = 0
        fi = filtered[i]
        for j in valid_idx:
            if j == i:
                continue
            fj = filtered[j]
            for key, vi in fi.items():
                if key not in fj:
                    continue
                vj = fj[key]
                vi_nan, vj_nan = (vi != vi), (vj != vj)
                if vi_nan and vj_nan:
                    s += 1
                elif not vi_nan and not vj_nan and random_probe.values_close(vi, vj):
                    s += 1
        scores[i] = s

    top_score = max(scores[i] for i in valid_idx)
    tied = [i for i in valid_idx if scores[i] == top_score]
    best = rng.choice(tied)
    return best, scores, tied, diag


# ---------------------------------------------------------------------------
# 不用 LLM 修、不用 canonical 的 harness 构建（2026-10-02）
# ---------------------------------------------------------------------------

_GCC_ERR_LINE_RE = re.compile(r"^case\.c:(\d+):\d+: error:", re.M)
_PRUNABLE = ("/*@A*/", "/*@P*/")


def build_harness_pruned(ref, candidates, discrete, continuous, outputs, k, max_rounds=8):
    """用候选代码（而不是标准答案）验证 harness 能否编译；编译失败时按 gcc 报错行号删掉
    出错的赋值/打印行（build_harness_source(one_line=True) 生成，每条语句一行并带标记），
    重新编译，直到通过。删不动时退回不给输入赋值的兜底 harness。全程不调用 LLM。
    返回 (harness, ok, info)，info = {"tier": "spec"|"fallback"|"failed", "pruned_assign": n, "pruned_print": n}。"""
    preamble = _test_preamble(ref)
    info = {"tier": "spec", "pruned_assign": 0, "pruned_print": 0}
    # 输出字段 = LLM 给出的 outputs ∪ 从指针参数结构体自动推导的数值字段（LLM 常漏给或给空）
    seen = {o[0] for o in outputs}
    outputs = list(outputs) + [o for o in random_probe.auto_output_specs(ref) if o[0] not in seen]
    info["n_outputs"] = len(outputs)
    attempts = [("spec", discrete, continuous), ("fallback", [], [])]
    for tier, disc, cont in attempts:
        info["tier"] = tier
        harness, _ranges = random_probe.build_harness_source(ref, disc, cont, outputs, k=k,
                                                             seed=random_probe.DEFAULT_SEED, one_line=True)
        harness = _splice_preamble(harness, preamble)
        for cand in candidates:
            for _round in range(max_rounds):
                r = run_probe_plain(ref, cand, harness, timeout=30.0)
                if r["outcome"] in ("ran", "runtime_error"):
                    return harness, True, info
                if r["outcome"] != "compile_error":
                    break
                _gen, h_start = random_probe.region_line_ranges(ref, cand, harness)
                lines = harness.split("\n")
                bad = set()
                for m in _GCC_ERR_LINE_RE.finditer(r.get("stderr", "")):
                    idx = int(m.group(1)) - h_start
                    if 0 <= idx < len(lines) and lines[idx].rstrip().endswith(_PRUNABLE):
                        bad.add(idx)
                if not bad:
                    break  # 错误不在可删行上（比如候选自身/preamble 冲突），换下一个候选验证
                for idx in bad:
                    info["pruned_print" if lines[idx].rstrip().endswith("/*@P*/") else "pruned_assign"] += 1
                harness = "\n".join(ln for i, ln in enumerate(lines) if i not in bad)
    info["tier"] = "failed"
    return harness, False, info
