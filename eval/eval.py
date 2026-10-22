#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""编译并运行生成的 C 语言解答，对照 AeroRHF 的黄金测试给出 pass@1 / safe@1。

复刻自 ICSE-2027-revision 的 eval_generate.py / eval_all_pass1.py，重写为单文件、
只依赖标准库的版本（不需要 numpy / tree_sitter，MISRA 安全检查直接调用 docker 镜像
里现成的 cppcheck，无需额外 Python 依赖）：

  1. build_source()：把 prompt + 生成的函数体 + 黄金测试拼成一份可编译的 C 文件，
     和原来的 process_humaneval_test() 完全一致。
  2. compile_and_run()：在 `aerorhf-eval` docker 镜像里用 gcc 编译并执行
     （绕开 includes/ 下 x86_64/arm64-Linux 静态库和本机 macOS gcc/clang 之间的架构不兼容）。
     只看进程退出码判定通过与否——黄金测试就是手写的 assert() 断言，不是比对 stdout。
  3. check_safety()：同一份源码额外跑一遍 cppcheck 的 MISRA-C 2012 检查，只统计生成
     函数体那个行区间内、命中"航天安全关键"规则子集（内存安全、抗辐射状态防御、
     数值安全、接口鲁棒性四类共 24 条）的违规——对应用户所说的"20多条安全约束"。
  4. estimate_pass_at_k() / aggregate()：标准的无偏 pass@k 估计量（Chen et al., 2021），
     safe@1 / safe-pass@1 用简单的通过率统计（默认每题 1 个样本，@1 场景下两者等价）。
  5. print_table()：按数据集来源（function/repo）和难度（easy/medium/hard）分组，
     汇总输出 pass@1、safe@1、safe-pass@1 表格。

用法：
    python3 eval.py --dataset ../dataset/gnc-adcs-100.jsonl \\
        --generations my_model_outputs.jsonl --output results.jsonl

    python3 eval.py --dataset ../dataset/test-20.jsonl --self-test \\
        --output smoke_test.jsonl
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import unicodedata
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDES_DIR = os.path.join(HERE, "includes")
DOCKER_IMAGE = "aerorhf-eval:latest"

# 每一次编译检查/MISRA检查/K=500探测运行/隐藏测试评分，都会各自开一个全新的
# `docker run --rm` 容器——baseline跑多题时靠 --workers 做任务级并发，任务内部
# 又要经过"生成→编译→MISRA→探测"层层反馈，短时间内产生的容器创建/销毁次数
# 非常密集。真实撞见过：--workers 5 跑 diffrandom 这种反馈轮次多的方法时，
# Docker Desktop 后端积压到连 `docker info` 都排不上号、daemon失联，且不会
# 自愈（得手动重启 Docker Desktop 才能恢复）。这里加一道进程级信号量，把
# **同时在跑的 docker 进程数**单独限流，跟任务级并发（--workers，影响的是
# LLM API调用并发）彻底解耦——任务线程数不用降，只是排队等一个"docker名额"，
# 从根上控制容器创建速率，不需要用户自己记得调小 --workers。
# 默认4：Docker Desktop这台机器的VM配了8个vCPU，给docker并发流量留出一半
# 余量，不需要跟task级worker数强行绑定。可以用环境变量覆盖，不用改代码。
_DOCKER_CONCURRENCY = int(os.environ.get("AERORHF_DOCKER_CONCURRENCY", "4"))
_docker_semaphore = threading.Semaphore(_DOCKER_CONCURRENCY)

# MISRA-C 2012 里被定义为"航天安全关键"的规则子集（共 24 条，四个类别），
# 对应 ICSE-2027-revision 的 eval_generate.py::static_check() 里的 aerosafetycritical_rules。
AEROSAFETY_RULE_GROUPS = {
    "内存安全与确定性": ["4.12", "21.3", "17.2", "18.8", "18.6", "18.1", "19.1"],
    "抗辐射与状态防御": ["16.4", "15.7", "2.1", "14.2", "15.1", "17.4"],
    "数值安全与异常保护": ["4.1", "14.1", "10.3", "9.1", "12.2"],
    "接口与编译器鲁棒性": ["13.2", "13.5", "17.7", "4.7", "8.4", "17.3"],
}
AEROSAFETY_RULES = {rule for rules in AEROSAFETY_RULE_GROUPS.values() for rule in rules}

# cppcheck 的 --addon=misra 默认只报规则编号，正文说"use --rule-texts=<file> to get
# proper output"——MISRA-C:2012 官方规则原文有版权，cppcheck 不内置。这里给这26条
# 自己写一句大白话描述（不是官方原文的转述/翻译，只是给模型一个"这条规则大概在说
# 什么"的方向提示），不给的话反馈只有裸编号，模型经常连着好几轮都不知道该往哪改
# （实测过 某内部任务 卡在同一条规则6轮没修好，就是这个原因）。
AEROSAFETY_RULE_HINTS = {
    "4.12": "不要用malloc/free/calloc/realloc这类动态内存分配",
    "21.3": "不要调用stdlib.h里的动态内存分配/释放函数",
    "17.2": "函数不能直接或间接递归调用自己",
    "18.8": "不要用变长数组（数组长度不能是运行时才确定的变量）",
    "18.6": "不要返回/存储指向已经出了作用域的局部变量的指针",
    "18.1": "指针运算/下标访问不能越出它所指向对象的边界",
    "19.1": "不要对有重叠的内存区域做拷贝（比如memcpy的源和目的地址范围重叠）",
    "16.4": "switch语句要有default分支",
    "15.7": "if/else if链的最后要有一个无条件的else兜底分支，不能有条件覆盖不到的情况",
    "2.1": "代码里不能有执行不到的死代码（比如return之后还有语句）",
    "14.2": "for循环的循环控制变量不能在循环体内被意外修改",
    "15.1": "不要使用goto语句",
    "17.4": "有返回值的函数，所有可能的执行路径都必须显式return一个值",
    "4.1": "八进制/十六进制转义序列后紧跟的字符不能产生歧义（比如\\x后面又接了个十六进制数字字符）",
    "14.1": "循环控制变量不能声明成浮点类型",
    "10.3": "不能把一个表达式的值隐式赋给类型更窄或不兼容的对象（比如把double直接赋给一个更小范围的int变量，没有显式转换）",
    "9.1": "自动变量（局部变量）在第一次被读取之前必须先显式赋值，不能读未初始化的值",
    "12.2": "位移操作符右边的位移量不能超过左边操作数类型的位宽",
    "13.2": "同一个表达式里，多个子表达式的求值结果不能依赖具体的求值顺序（比如同一条语句里对同一个变量既读又写）",
    "13.5": "&&或||右边的操作数不能有副作用（比如函数调用改了全局状态），因为短路求值时右边不一定会执行",
    "17.7": "调用有返回值的函数时，返回值必须被使用，或者显式(void)转换后丢弃，不能直接忽略",
    "4.7": "如果调用的库函数会返回错误码，调用处必须检查这个返回值",
    "8.4": "有外部链接的函数/变量，在使用之前必须有匹配的显式声明（extern声明要跟定义一致）",
    "17.3": "调用一个函数之前，必须先有它的函数原型声明，不能隐式声明",
}

MISRA_VIOLATION_RE = re.compile(r":(\d+):\d+: style: misra violation.*\[misra-c2012-(\d+\.\d+)\]")


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_source(ref, generation):
    """拼出完整可编译的 C 文件。

    ref['prompt'] 以函数签名和左花括号结尾；ref['test'] 以对应的右花括号开头，
    后面跟着手写的、里面全是 assert() 的 main()。
    """
    return "\r\n" + ref["prompt"] + generation + "\r\n" + ref["test"]


def generation_line_range(ref, generation):
    """算出 generation 在拼接后源码里对应的起止行号（1-indexed，右开区间），
    用来把 MISRA 检查限制在"生成的这段代码"里，而不是 prompt/test 的模板代码。"""
    prefix = "\r\n" + ref["prompt"]
    start = prefix.count("\n") + 1
    end = start + generation.count("\n") + 1
    return start, end


# ---------------------------------------------------------------------------
# docker 调用：每个进程一个常驻容器 + docker exec（2026-10-02）
# Docker Desktop(macOS) 对每次 `docker run -v <新临时目录>` 打开的目录句柄不释放，
# 跑几千个容器后后台进程句柄耗尽（实测 6.4 万个，DIR 类型，指向 /var/folders/.../T），
# 守护进程失去响应。改为每进程只挂载一次系统临时目录，之后用 docker exec 执行。
# 超时由容器内 `timeout -s KILL` 负责，杀掉的是容器内进程，不会遗留空转进程。
# 工作目录不在系统临时目录下时退回旧的一次性 `docker run`。
# ---------------------------------------------------------------------------
_TMP_ROOT = os.path.realpath(tempfile.gettempdir())
_persist_lock = threading.Lock()
_persist_name = None


def _docker_failed(result):
    err = result.stderr or ""
    return (result.returncode == 125 or "Cannot connect to the Docker daemon" in err
            or "No such container" in err or "is not running" in err)


def _ensure_persistent_container(force_new=False):
    global _persist_name
    with _persist_lock:
        if _persist_name and not force_new:
            return _persist_name
        if _persist_name:
            subprocess.run(["docker", "rm", "-f", _persist_name], capture_output=True, timeout=60)
        name = f"aerorhf-eval-persist-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        r = subprocess.run(
            ["docker", "run", "-d", "--rm", "--name", name,
             "-v", f"{INCLUDES_DIR}:/includes:ro", "-v", f"{_TMP_ROOT}:/hosttmp",
             DOCKER_IMAGE, "sleep", "infinity"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"docker 运行失败（不是被测代码的问题）：无法启动常驻容器：{(r.stderr or '').strip()[:300]}")
        _persist_name = name
        import atexit
        atexit.register(lambda n=name: subprocess.run(["docker", "rm", "-f", n], capture_output=True, timeout=60))
        return name


def _docker_run_once(workdir, shell_cmd, timeout):
    # 旧路径：一次性容器（会触发 Docker Desktop 的句柄泄漏，只在工作目录不在临时目录下时使用）
    name = f"aerorhf-eval-{uuid.uuid4().hex[:12]}"
    try:
        return subprocess.run(
            ["docker", "run", "--rm", "--name", name,
             "-v", f"{INCLUDES_DIR}:/includes:ro", "-v", f"{workdir}:/work", "-w", "/work",
             DOCKER_IMAGE, "bash", "-c", shell_cmd],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
        raise


def docker_run(workdir, shell_cmd, timeout):
    real = os.path.realpath(workdir)
    with _docker_semaphore:
        if not real.startswith(_TMP_ROOT + os.sep):
            result = _docker_run_once(workdir, shell_cmd, timeout)
        else:
            inner = "/hosttmp/" + os.path.relpath(real, _TMP_ROOT)
            secs = max(1, int(round(timeout)))
            for attempt in range(2):
                name = _ensure_persistent_container(force_new=attempt > 0)
                result = subprocess.run(
                    ["docker", "exec", "-w", inner, name,
                     "timeout", "-s", "KILL", f"{secs}s", "bash", "-c", shell_cmd],
                    capture_output=True, text=True, timeout=timeout + 60)
                if result.returncode == 137 and "timeout" not in (result.stderr or "").lower():
                    # 容器内 timeout 以 SIGKILL 结束命令 → 与旧实现一致，按超时处理
                    raise subprocess.TimeoutExpired(shell_cmd, timeout)
                if not _docker_failed(result):
                    break  # 常驻容器失效（docker 重启等）时重建一次再试
    # docker 自身起不来（守护进程挂了等）时，调用方会把它当成"编译失败"静默记账
    # （曾因此污染过一整批评测缓存），这里直接抛异常。
    if _docker_failed(result):
        raise RuntimeError(f"docker 运行失败（不是被测代码的问题）：{(result.stderr or '').strip()[:300]}")
    return result


def _compile_cmd(project):
    link_libs = (["-lstdutils7_V2_01"] if os.path.exists(os.path.join(INCLUDES_DIR, "libstdutils7_V2_01.a")) else []) + ([f"-l{project}"] if project else [])  # gnc100 needs no project library
    # 数据集是照着 gcc 7.5 校验通过的，那个版本里"把二维数组传给声明为一维指针的
    # 形参"只是警告；现代 gcc（14+）默认把它升级成报错，这里降级回警告，
    # 还原老工具链的行为，而不是因为编译器版本漂移就判题目失败。
    # --start-group/--end-group：部分工程静态库会反过来调用 libstdutils7_V2_01
    # （或者反之），链接器单趟从左到右扫描不够用，得允许它反复扫描直到收敛。
    return (
        "gcc case.c -I/includes -Wl,--start-group "
        f"-L/includes {' '.join(link_libs)} -Wl,--end-group "
        "-Wno-incompatible-pointer-types -o case.out -lm"
    )


def compile_and_run_detailed(project, timeout, workdir):
    """返回 (outcome, message)。outcome 同 compile_and_run()；message 是编译器/
    运行时输出的原始文本（outcome=="passed" 时为空字符串），给最终评分用。"""
    try:
        compiled = docker_run(workdir, _compile_cmd(project), timeout)
    except subprocess.TimeoutExpired:
        return "timeout", "编译超时"
    if compiled.returncode != 0:
        return "compile_error", compiled.stderr or compiled.stdout

    try:
        ran = docker_run(workdir, "./case.out", timeout)
    except subprocess.TimeoutExpired:
        return "timeout", "运行超时"
    if ran.returncode != 0:
        return "runtime_error", (ran.stderr or ran.stdout or f"进程以非零退出码 {ran.returncode} 结束")
    return "passed", ""


def compile_and_run(project, timeout, workdir):
    """返回 passed / compile_error / runtime_error / timeout 之一。"""
    outcome, _ = compile_and_run_detailed(project, timeout, workdir)
    return outcome


def compile_only(project, timeout, workdir):
    """只编译+链接，不执行 case.out。给生成阶段的反馈循环用——那里的收敛/
    早停判断绝不能依赖隐藏测试（case.c 里 ref['test'] 那段 assert()）的运行
    结果，那相当于把标准答案泄露进生成过程；只有 gcc 编译器本身的诊断（或
    额外的静态分析，见 feedback_misra.py）是生成时就能拿到的、不含答案的
    信号。最终评分（本文件的 evaluate()/run_case()）不受影响，仍然用上面
    完整跑一遍的 compile_and_run_detailed()。

    返回 (outcome, message)，outcome 只有 compiled / compile_error / timeout。
    """
    try:
        compiled = docker_run(workdir, _compile_cmd(project), timeout)
    except subprocess.TimeoutExpired:
        return "timeout", "编译超时"
    if compiled.returncode != 0:
        return "compile_error", compiled.stderr or compiled.stdout
    return "compiled", ""


def check_safety(line_range, timeout, workdir):
    """对 workdir/case.c 跑 cppcheck 的 MISRA 检查，只看生成代码那段行区间内、
    命中"航天安全关键"规则子集的违规。返回 (是否安全, 命中的规则号列表, 明细列表)。
    明细列表每项是 {"rule":..., "line": 相对生成代码起始行的行号（1-indexed）}——
    cppcheck本身不给行号以外的规则描述（见 AEROSAFETY_RULE_HINTS 的注释），行号
    是唯一能帮模型定位到具体哪一行的信息，之前这里直接把它丢掉了，导致反馈只有
    裸规则编号，模型经常不知道该改哪里。violations列表为了兼容旧调用方保留，
    跟明细列表的规则集合一致，只是没有行号。"""
    start, end = line_range
    cmd = "cppcheck --addon=misra --enable=all --suppress=missingIncludeSystem -I/includes case.c"
    # cppcheck 超时是机器负载问题，不是代码不安全：并发评测时曾有约 1/4 的
    # "通过但不安全"其实是这里超时被记成违规。超时后放宽到 6 倍重试，仍超时就抛异常。
    try:
        result = docker_run(workdir, cmd, timeout)
    except subprocess.TimeoutExpired:
        try:
            result = docker_run(workdir, cmd, timeout * 6)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"cppcheck 超时（{timeout * 6:.0f}s），属评测环境问题，不能记为不安全")

    violations = []
    details = []
    for match in MISRA_VIOLATION_RE.finditer(result.stderr):
        line_no, rule = int(match.group(1)), match.group(2)
        if start <= line_no < end and rule in AEROSAFETY_RULES:
            violations.append(rule)
            details.append({"rule": rule, "line": line_no - start + 1})
    return len(violations) == 0, violations, details


def run_case(ref, generation, timeout, workdir):
    source = build_source(ref, generation)
    with open(os.path.join(workdir, "case.c"), "w", encoding="utf-8") as f:
        f.write(source)

    outcome = compile_and_run(ref.get("project"), timeout, workdir)
    safe, violations, _details = check_safety(generation_line_range(ref, generation), timeout, workdir)
    return {"outcome": outcome, "safe": safe, "violations": violations}


def estimate_pass_at_k(n, c, k):
    """无偏 pass@k 估计量（Chen et al., 2021）：1 - C(n-c, k) / C(n, k)。"""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def rate_at_1(flags):
    """@1 场景下的简单通过率（每题只有 1 个样本时，等价于无偏估计量）。"""
    return sum(flags) / len(flags) if flags else None


def aggregate(results, k_values):
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)

    n_samples = [len(v) for v in by_task.values()]
    n_correct = [sum(x["outcome"] == "passed" for x in v) for v in by_task.values()]

    summary = {"num_tasks": len(by_task), "num_samples": sum(n_samples)}
    for k in k_values:
        if n_samples and all(n >= k for n in n_samples):
            scores = [estimate_pass_at_k(n, c, k) for n, c in zip(n_samples, n_correct)]
            summary[f"pass@{k}"] = sum(scores) / len(scores)
        else:
            summary[f"pass@{k}"] = None  # 不是每道题都有 >= k 个样本

    # 编译通过率：outcome 只要不是 compile_error 就算编译过了（passed/runtime_error 显然
    # 编译成功；timeout 绝大多数是运行时死循环超时，不是编译超时，10s 的编译超时预算
    # 对这些单文件小程序几乎不会触发，所以按"编译成功"计入，这里明确记一下这个简化）。
    summary["compile@1"] = rate_at_1([r["outcome"] != "compile_error" for r in results])
    summary["safe@1"] = rate_at_1([r["safe"] for r in results])
    summary["safe-pass@1"] = rate_at_1([r["outcome"] == "passed" and r["safe"] for r in results])
    return summary


def evaluate(refs, generations, timeout, workers):
    """并行编译+运行每一对 (ref, generation)，返回每次尝试的结果列表。"""
    tasks = []
    for gen in generations:
        ref = refs.get(gen["task_id"])
        if ref is None:
            print(f"warning: unknown task_id {gen['task_id']!r}, skipping", file=sys.stderr)
            continue
        tasks.append((ref, gen))

    results = []
    with tempfile.TemporaryDirectory(prefix="aerorhf_eval_") as base_tmp:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, (ref, gen) in enumerate(tasks):
                workdir = os.path.join(base_tmp, f"case_{i}")
                os.makedirs(workdir)
                future = pool.submit(run_case, ref, gen["generation"], timeout, workdir)
                futures[future] = ref["task_id"]

            done = 0
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    case_result = future.result()
                except Exception as e:
                    case_result = {"outcome": f"error: {e}", "safe": False, "violations": []}
                done += 1
                print(f"[{done}/{len(tasks)}] {task_id}: {case_result['outcome']}, "
                      f"safe={case_result['safe']}", file=sys.stderr)
                results.append({"task_id": task_id, **case_result})
    return results


def group_key(ref):
    """按数据集来源（function/repo）分组：repo 级题目带 project 字段，function 级没有。"""
    origin = "repo" if ref.get("project") else "function"
    difficulty = ref.get("difficulty", "unknown")
    return origin, difficulty


def build_breakdown(refs, results, k_values):
    """按 (来源, 难度) 分组统计，外加每个来源的小计、以及全部题目的总计。"""
    groups = defaultdict(list)
    for r in results:
        ref = refs.get(r["task_id"])
        if ref is None:
            continue
        groups[group_key(ref)].append(r)

    rows = []
    for origin in ["function", "repo"]:
        origin_results = []
        for difficulty in ["easy", "medium", "hard"]:
            key = (origin, difficulty)
            if key in groups:
                rows.append((origin, difficulty, aggregate(groups[key], k_values)))
                origin_results += groups[key]
        if origin_results:
            rows.append((origin, "总计", aggregate(origin_results, k_values)))

    all_results = [r for v in groups.values() for r in v]
    if all_results:
        rows.append(("全部", "总计", aggregate(all_results, k_values)))
    return rows


def display_width(s):
    """终端里的实际显示宽度：中文全角字符算 2 列，其余算 1 列。
    直接用 len() 对齐表格会因为中文字符导致列错位，所以单独算一个显示宽度。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def pad(s, width):
    return s + " " * max(0, width - display_width(s))


def format_table(rows, k_values):
    headers = ["数据集", "难度", "样本数", "compile@1"] + [f"pass@{k}" for k in k_values] + ["safe@1", "safe-pass@1"]

    def fmt(v):
        return "--" if v is None else f"{v:.3f}"

    table_rows = []
    for origin, difficulty, summary in rows:
        row = [origin, difficulty, str(summary["num_samples"]), fmt(summary.get("compile@1"))]
        row += [fmt(summary.get(f"pass@{k}")) for k in k_values]
        row += [fmt(summary.get("safe@1")), fmt(summary.get("safe-pass@1"))]
        table_rows.append(row)

    widths = [max(display_width(h), *(display_width(r[i]) for r in table_rows)) for i, h in enumerate(headers)]
    lines = []
    lines.append("  ".join(pad(h, w) for h, w in zip(headers, widths)))
    lines.append("  ".join("-" * w for w in widths))
    for row in table_rows:
        lines.append("  ".join(pad(c, w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def check_docker_ready():
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=True)
    except Exception:
        sys.exit(
            "error: docker 不可用或没有启动。请启动 Docker Desktop，"
            "或用下面的命令构建评测镜像：\n"
            "  docker build -t aerorhf-eval:latest exps/eval/docker"
        )
    images = subprocess.run(
        ["docker", "images", "-q", DOCKER_IMAGE], capture_output=True, text=True
    ).stdout.strip()
    if not images:
        sys.exit(
            f"error: 找不到 docker 镜像 '{DOCKER_IMAGE}'，请先构建：\n"
            "  docker build -t aerorhf-eval:latest exps/eval/docker"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="题目数据集 jsonl（task_id, prompt, canonical_solution, test, [project], [difficulty]）")
    parser.add_argument("--generations", help="待评测的生成结果 jsonl，每行 {task_id, generation}（每个样本一行）")
    parser.add_argument("--self-test", action="store_true", help="忽略 --generations，直接拿每道题自己的 canonical_solution 当生成结果做冒烟测试")
    parser.add_argument("--output", required=True, help="逐条结果输出路径（jsonl）；同时会写一份同名的 *.summary.json")
    parser.add_argument("--k", type=int, nargs="+", default=[1], help="要计算的 pass@k 的 k 值（默认 1）")
    parser.add_argument("--timeout", type=float, default=10.0, help="编译/运行/静态检查每一步的超时时间，单位秒（默认 10）")
    parser.add_argument("--workers", type=int, default=8, help="并行 docker 调用数（默认 8）")
    args = parser.parse_args()

    if not args.self_test and not args.generations:
        parser.error("必须提供 --generations 或者 --self-test 其中之一")

    check_docker_ready()

    refs = {r["task_id"]: r for r in load_jsonl(args.dataset)}
    if args.self_test:
        generations = [{"task_id": tid, "generation": r["canonical_solution"]} for tid, r in refs.items()]
    else:
        generations = load_jsonl(args.generations)

    results = evaluate(refs, generations, args.timeout, args.workers)
    summary = aggregate(results, args.k)
    breakdown = build_breakdown(refs, results, args.k)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = os.path.splitext(args.output)[0] + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"overall": summary, "breakdown": [
            {"数据集": o, "难度": d, **s} for o, d, s in breakdown
        ]}, f, ensure_ascii=False, indent=2)

    print(format_table(breakdown, args.k))


if __name__ == "__main__":
    main()
