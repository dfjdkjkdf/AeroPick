#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""随机差分探测：给同一道题的多份候选实现造一份不含标准答案的确定性 harness
（固定种子、K 组随机输入，纯 Python 拼接，不需要 LLM 转写 C 语句），配合
`-fsanitize=undefined` 编译，产出两类信号：

1. 绝对信号（单份候选自己就能判，不需要跟别的候选比较）：
   - 未定义行为（UBSan 运行时报错：数组越界、整数溢出、无效移位……）
   - NaN/Inf 输出
   - 声明了值域的输出越界
2. 相对信号（候选之间逐输入比较数值是否一致）：分歧分。

设计取舍见对话记录，这里只列代码里不直接体现、容易忘记原因的几条：
- harness 只在最开始做一次 LLM 调用（抽取输入变量/值域，窄范围信息抽取），
  具体的 K 组随机赋值语句由纯 Python 拼接，不再有 LLM 转写/分批/截断这类
  历史上反复翻车的问题。数组支持一维/二维（矩阵在这个领域很常见，比如
  3x3 姿态矩阵），更高维度不支持——真遇到时会编译不过，交给上层的编译
  反馈循环去修（这条路径本来就存在，不需要 harness 生成器自己完美）。
- harness 自己声明的辅助符号（PRNG 状态、循环变量）统一加 `__aeroprobe_`
  前缀，避免跟候选幻觉出的字段名（或题目真实的字段名）撞车。
- 函数参数按 declaration 解析出的指针/值语义正确处理（指针参数取地址传参，
  内部用局部结构体实例 + 点号访问，不是直接对指针用点号）；每次循环开头
  对每个参数结构体做 memset 清零，避免读到上一次迭代残留的未初始化内存
  （这不仅是正确性问题，UBSan 对"读未初始化内存"本身就会报错，不清零会
  把"我们探测代码没初始化全"误判成"候选有未定义行为"）。
- 全局变量（repo 级题目常见，函数不带参数、直接读写共享头文件里的
  `extern` 全局结构体）尽力从 `extern TYPE name;` 声明里解析出类型做同样的
  清零；解析不出类型的全局量放弃清零，只赋值抽取到的具体字段——这是已知
  的精度上限，不是全部路径都能做到跟参数一样干净。
- 是否要在 harness 最前面补收尾 "}"，取决于 ref["test"] 是不是自己提供
  收尾括号（跟 llm_generate.fix_closing_brace 依据同一条规则，见
  build_harness_source 内部说明）——少数题目（ref["test"] 不以 "}" 开头）
  候选自己已经带了收尾括号，harness 不能再补一个。
"""
import math
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from eval import docker_run  # noqa: E402

from llm_generate import call_llm, extract_requirement  # noqa: E402

DEFAULT_K = 500
DEFAULT_SEED = 2026080101
PREFIX = "__aeroprobe_"
BULK_BLOCK_BEGIN = "    // __AEROPROBE_BULK_BEGIN__ 批量查表/常量矩阵字段，只在这里设置一次\n"
BULK_BLOCK_END = "    // __AEROPROBE_BULK_END__\n"

VALUES_CLOSE_ATOL = 1e-9
VALUES_CLOSE_RTOL = 1e-6


def values_close(a, b, atol=VALUES_CLOSE_ATOL, rtol=VALUES_CLOSE_RTOL):
    """相对+绝对混合容差——纯绝对容差在这个领域会同时犯两个方向的错：
    小量级（四元数分量）上把真实分歧判成一致，大量级（位置米数）上把浮点
    舍入判成分歧。"""
    if a == b:  # 2026-10-04：inf==inf 原先因 inf-inf=nan 判为不一致
        return True
    if math.isinf(a) or math.isinf(b):  # 原先 inf 与大有限值、+inf 与 -inf 都会判为一致
        return False
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


# ---------------------------------------------------------------------------
# 第一步：抽取输入变量/值域 + 输出（唯一一次 LLM 调用）
# ---------------------------------------------------------------------------

EXTRACT_PROBE_SPEC_PROMPT = """请分析下面的需求和C代码上下文，列出被测函数用到的每一个【输入】变量
（函数参数、或者函数直接读写的全局变量/结构体字段），按类型分类；同时列出每一个【输出】。

- 离散型：只能取有限几个固定值的变量（布尔量、模式字/枚举）。多个独立的布尔标志位/模式字，
  每一个都要单独列一行，不能合并、不能只列一个代表。
- 连续型：可以在某个范围内取任意具体数值的变量（浮点量为主），给出取值范围[最小值,最大值]——
  需求里有明确值域就用需求的，没有就给一个物理上合理的估计范围。
- 如果变量是一维数组/向量，标注元素个数，格式 [ARRAY n]；如果是二维数组/矩阵
  （比如3x3姿态矩阵），标注两维，格式 [ARRAY n,m]；标量不用标注。
- 输出：列出函数每一个输出的访问路径。如果需求或表格里明确给出了这个输出的合法值域
  （比如单位向量分量在[-1,1]、角度在[0,2π)这类），额外标注 [RANGE 最小值,最大值]；
  给不出可靠值域就不要标注，不要瞎猜一个。数组/矩阵输出同样标注 [ARRAY ...]。

访问路径写法：函数参数直接写参数名（比如 pOut），参数下面的结构体字段用点号（比如
pOut.field，不管这个参数实际是不是指针，统一用点号，不用箭头）；全局变量同理，
访问路径就是它在代码里被引用的写法本身，不要带下标。

数值写法：取值/范围/数组维度里的每一个数，一律写成十进制数字（比如 3.14159265、
-1000000.0），不要写 π、∞、e、NMAX+1 这类数学符号或代码里的宏名/表达式——
正则解析只认十进制数字，写符号会导致整行解析失败、这个变量/输出被直接丢弃。
遇到理论上无界的范围（比如"大于0"、"任意实数"），换算成一个物理上合理的、
足够大/小的具体数字代表"无界"（比如 ±1000000.0），不要写 ∞。数组维度如果需求里
是用一个 #define 宏表示的（比如 NMAX+1），到C代码上下文里查出这个宏实际定义的
数值，直接写算出来的整数（比如 #define NMAX 13 时，NMAX+1 要写成 14）。

#需求
{requirement}

#C代码上下文（结构体定义、全局变量声明、函数签名）
```c
{context}
```

#输出格式（严格按下面的格式，每行一条，不要有多余说明文字）
DISCRETE: 访问路径 [ARRAY n或n,m] = 取值1, 取值2, ...
CONTINUOUS: 访问路径 [ARRAY n或n,m] = [最小值, 最大值]
OUTPUT: 访问路径 [ARRAY n或n,m] [RANGE 最小值, 最大值]

（[ARRAY ...] 和 [RANGE ...] 都是可选的，标量/无已知值域时省略这部分，不要写空括号）

#示例（跟本次任务无关，只演示格式）
DISCRETE: sCmd.Mode = 0, 1, 2
DISCRETE: sCmd.EnableA = TRUE, FALSE
CONTINUOUS: sGyro.w [ARRAY 3] = [-0.1, 0.1]
CONTINUOUS: sAtti.Cbo [ARRAY 3,3] = [-1.0, 1.0]
CONTINUOUS: pIn.dt = [0.0, 1.0]
OUTPUT: sOut.q [ARRAY 4] [RANGE -1.0, 1.0]
OUTPUT: sOut.flag
"""

_ARRAY_TAG_RE = r"(?:\[ARRAY\s+(\d+)(?:\s*,\s*(\d+))?\])?"
_DISCRETE_RE = re.compile(
    r"^DISCRETE:\s*(.+?)\s*" + _ARRAY_TAG_RE + r"\s*=\s*(.+)$"
)
_CONTINUOUS_RE = re.compile(
    r"^CONTINUOUS:\s*(.+?)\s*" + _ARRAY_TAG_RE + r"\s*=\s*\[\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*\]$"
)
_OUTPUT_RE = re.compile(
    r"^OUTPUT:\s*([A-Za-z_][A-Za-z0-9_.]*)\s*" + _ARRAY_TAG_RE
    + r"\s*(?:\[RANGE\s+(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*\])?\s*$"
)
# DISCRETE行的取值列表来自LLM输出的自由文本，逗号分割之后原样当成C字面量塞进
# harness里——之前完全没有校验过每个token长得像不像一个合法的C字面量。真实
# 翻车案例：某内部任务，LLM在取值列表里混进了一段中文说明"连续型"（大概率是想说
# "这个字段其实该走CONTINUOUS，不是DISCRETE"，但格式没按行分开），被原样
# 当成一个赋值目标塞进C代码，直接编译失败，导致这道题的Tier1差分探测从此
# 完全跑不起来（探测harness构造失败是"编译错误"，不是"随机运气不好"）。
# 只接受数值字面量（含十六进制/科学计数法/浮点后缀）或裸标识符（宏常量在
# 这之前已经被_substitute_macro_constants换成数值了，正常情况不会剩下裸
# 标识符；剩下的十有八九就是这类夹带的自然语言碎片）——校验不过的token
# 整个丢弃，不是让它硬闯进C代码里之后才在编译阶段报错。
_VALID_C_TOKEN_RE = re.compile(r"^-?(0[xX][0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?[fFuUlL]*|[A-Za-z_]\w*)$")


def _dims(d1, d2):
    """把正则抓到的 (n, m或None) 归一成 None（标量）/ (n,)（一维）/ (n,m)（二维）。"""
    if not d1:
        return None
    return (int(d1), int(d2)) if d2 else (int(d1),)


_MATH_SYMBOL_SUBS = [
    (re.compile(r"(?<![\w.])-\s*∞"), "-1000000.0"),
    (re.compile(r"∞"), "1000000.0"),
    (re.compile(r"π"), "3.14159265358979"),
]


def _substitute_math_symbols(text):
    """兜底：即使 prompt 已经明确要求只写十进制数字，模型偶尔还是会习惯性写
    π/∞ 这类数学符号——正则解析只认十进制数字，不替换的话这一整行会直接解析
    失败、这个变量/输出被静默丢弃（真实翻车过：某题4个OUTPUT全部因为
    "[RANGE -∞, ∞]"这种写法解析失败，harness最后一个输出字段都没有）。
    这里在正则匹配之前统一替换掉，双保险。"""
    for pattern, repl in _MATH_SYMBOL_SUBS:
        text = pattern.sub(repl, text)
    return text


_DEFINE_CONST_RE = re.compile(r"#define\s+([A-Za-z_]\w*)\s+(-?\d+(?:\.\d+)?)\b")
_SIMPLE_ARITH_RE = re.compile(r"\b(\d+)\s*([+\-])\s*(\d+)\b")


def _substitute_macro_constants(text, prompt_text):
    """兜底2：真实翻车过——模型把数组维度写成 `NMAX+1` 这种"宏名+算式"，
    不是纯数字，正则同样解析不出来。从C代码上下文里扫 `#define NAME NUMBER`
    形式的宏定义，把这些宏名在LLM回复里按整词替换成对应数值（比如
    `#define NMAX 13` 时 `NMAX+1` 先变成 `13+1`），再对剩下的"数字+数字"/
    "数字-数字"这类最简单算式做一次化简（`13+1` -> `14`）——只处理这种最简单
    的算式，复杂表达式化简不了就留给下面的"非法标签整行丢弃"兜底。"""
    consts = dict(_DEFINE_CONST_RE.findall(prompt_text or ""))
    if consts:
        for name, value in consts.items():
            text = re.sub(rf"\b{re.escape(name)}\b", value, text)
    # 反复化简，直到没有更多"数字+数字"模式可以合并（应对NMAX+1+1这种链式写法）
    for _ in range(5):
        new_text = _SIMPLE_ARITH_RE.sub(
            lambda m: str(int(m.group(1)) + int(m.group(3)) if m.group(2) == "+"
                          else int(m.group(1)) - int(m.group(3))),
            text,
        )
        if new_text == text:
            break
        text = new_text
    return text


_MALFORMED_ARRAY_TAG_RE = re.compile(r"\[ARRAY\s+(?!\d+(?:\s*,\s*\d+)?\s*\])")
_MALFORMED_RANGE_TAG_RE = re.compile(r"\[RANGE\s+(?!-?[\d.eE+-]+\s*,\s*-?[\d.eE+-]+\s*\])")


def _has_malformed_tag(line):
    """兜底3（最后一道防线）：上面两层兜底之后，如果[ARRAY ...]/[RANGE ...]
    标签里仍然不是纯数字（比如遇到解析不了的复杂表达式），不能让下面的正则
    非贪婪匹配把这段乱码悄悄吞进变量路径里生成出语法错误的C代码——整行丢弃，
    这个变量/输出这次就不测了，比生成垃圾C代码安全。"""
    return bool(_MALFORMED_ARRAY_TAG_RE.search(line) or _MALFORMED_RANGE_TAG_RE.search(line))


def extract_probe_spec(model_key, ref, thinking=None, max_tokens=None):
    """返回 (discrete, continuous, outputs, usage)。
    discrete:   [(path, dims_or_None, [值1, 值2, ...]), ...]
    continuous: [(path, dims_or_None, lo, hi), ...]
    outputs:    [(path, dims_or_None, range_or_None), ...]  range是(lo,hi)或None
    dims 是 None（标量）/ (n,)（一维数组）/ (n,m)（二维数组）。
    """
    requirement = extract_requirement(ref["prompt"])
    prompt = EXTRACT_PROBE_SPEC_PROMPT.format(requirement=requirement, context=ref["prompt"])
    reply, usage = call_llm(model_key, [{"role": "user", "content": prompt}],
                             thinking=thinking, max_tokens=max_tokens)
    reply = _substitute_math_symbols(reply)
    reply = _substitute_macro_constants(reply, ref.get("prompt") or "")

    discrete, continuous, outputs = [], [], []
    for line in reply.splitlines():
        line = line.strip()
        if _has_malformed_tag(line):
            continue
        m = _DISCRETE_RE.match(line)
        if m:
            path, d1, d2, raw_values = m.groups()
            values = [v.strip() for v in raw_values.split(",") if v.strip()]
            values = [v for v in values if _VALID_C_TOKEN_RE.match(v)]
            if values:
                discrete.append((path.strip(), _dims(d1, d2), values))
            continue
        m = _CONTINUOUS_RE.match(line)
        if m:
            path, d1, d2, lo, hi = m.groups()
            continuous.append((path.strip(), _dims(d1, d2), float(lo), float(hi)))
            continue
        m = _OUTPUT_RE.match(line)
        if m:
            path, d1, d2, lo, hi = m.groups()
            rng = (float(lo), float(hi)) if lo is not None else None
            outputs.append((path.strip(), _dims(d1, d2), rng))
    return discrete, continuous, outputs, usage


# ---------------------------------------------------------------------------
# 第二步：declaration 解析——参数指针/值语义、类型名；全局 extern 变量类型扫描
# ---------------------------------------------------------------------------

_PARAM_LIST_RE = re.compile(r"\(([^)]*)\)")


def parse_params_typed(declaration):
    """从函数签名解析每个参数 (名字, 是否指针, 类型名)。类型名取参数声明里
    去掉限定符/星号/参数名/数组维度之后剩下的最后一个标识符。"""
    m = _PARAM_LIST_RE.search(declaration or "")
    if not m:
        return []
    params = []
    for part in m.group(1).split(","):
        part = part.strip()
        if not part or part == "void":
            continue
        is_pointer = "*" in part
        name_match = re.search(r"(\w+)\s*(?:\[[^\]]*\])?$", part)
        if not name_match:
            continue
        name = name_match.group(1)
        type_tokens = [
            t for t in re.findall(r"[A-Za-z_]\w*", part[:name_match.start()])
            if t not in ("const", "volatile", "static")
        ]
        if type_tokens:
            params.append((name, is_pointer, type_tokens[-1]))
    return params


_EXTERN_VAR_RE = re.compile(
    r"extern\s+((?:const\s+|volatile\s+)*[A-Za-z_]\w*)\s*(\*)?\s*(\w+)\s*(\[\s*\d+\s*\])?\s*;"
)


def scan_extern_globals(prompt_text):
    """扫 ref["prompt"] 里 `extern TYPE name;` / `extern TYPE *name;` /
    `extern TYPE name[N];` 形式的全局变量声明，返回
    {变量名: (类型名, 是否指针)}。类型名用于给全局输入做 memset 清零；
    是否指针用于决定访问这个全局变量的字段时该用 `.` 还是 `->`——这个判断
    之前完全没做（一律用`.`），真实翻车过：某个全局量实际声明是指针类型，
    候选/harness 里对它取字段却一直用`.`，gcc 直接报"is a pointer; did you
    mean to use '->'"。这跟函数参数的指针判断走的是同一条路——参数那边
    早就是解析 declaration 里的`*`来确定性判断，全局变量这里只是把同一个
    思路补上，不是新发明一套，也不需要让LLM去猜。
    解析不出的全局量（比如更复杂的声明写法）不在这个表里，调用方会跳过
    清零/退回默认的`.`访问。"""
    result = {}
    for m in _EXTERN_VAR_RE.finditer(prompt_text):
        type_name, is_ptr, name = m.group(1), m.group(2), m.group(3)
        result[name] = (type_name.split()[-1], bool(is_ptr))
    return result


# ---------------------------------------------------------------------------
# 第三步：纯 Python 拼 harness（无 LLM 转写）
# ---------------------------------------------------------------------------

def _base_root(path):
    return path.split(".", 1)[0] if "." in path else path


def _rewrite_lvalue(path, param_names, extern_pointer_names=frozenset()):
    """函数参数的路径统一改写成局部变量名（带前缀，用点号访问——参数在
    harness里是局部结构体实例，不是原来的指针本身，指针语义已经在调用时
    通过取地址处理掉了，这里不需要再用箭头）。

    非参数（全局变量）路径原样使用第一段作为根，但如果这个根在
    extern_pointer_names 里（scan_extern_globals 解析出的、真实声明为指针的
    全局量），把根之后的第一个分隔符从`.`换成`->`——全局变量指针不像参数
    那样会被改写成局部实例，访问它自己的字段必须用箭头，不能跟不是指针的
    全局变量一样用点号。"""
    root, _, rest = path.partition(".")
    if root in param_names:
        return f"{PREFIX}arg_{root}" + (("." + rest) if rest else "")
    if rest and root in extern_pointer_names:
        return f"{root}->{rest}"
    return path


def _indices(dims):
    """dims=None -> [()]；dims=(n,) -> [(0,),(1,),...]；dims=(n,m) -> [(0,0),(0,1),...]。
    每个元素是这个位置需要的下标元组（空元组表示标量，不加下标）。"""
    if dims is None:
        return [()]
    if len(dims) == 1:
        return [(i,) for i in range(dims[0])]
    return [(i, j) for i in range(dims[0]) for j in range(dims[1])]


def _subscript(idx):
    return "".join(f"[{i}]" for i in idx)


def _unique_suffix(*parts):
    return abs(hash(parts)) % 1000000


def _emit_discrete_assign(lvalue, dims, values, indent="        "):
    """离散赋值用 if/else 挑选链，不声明数组——不需要知道 values 的真实 C 类型
    （可能是数字字面量、也可能是 TRUE/FALSE 宏），直接把候选值原样抄进分支体。
    数组/矩阵的每个元素独立随机挑选，不是整体重复同一个值。"""
    lines = []
    for idx in _indices(dims):
        target = f"{lvalue}{_subscript(idx)}"
        pick_var = f"{PREFIX}pick_{_unique_suffix(lvalue, idx)}"
        lines.append(f"{indent}{{ unsigned int {pick_var} = {PREFIX}rand_u32() % {len(values)}u;")
        for i, v in enumerate(values):
            branch = "if" if i == 0 else "else if"
            lines.append(f"{indent}  {branch} ({pick_var} == {i}u) {{ {target} = {v}; }}")
        lines.append(f"{indent}  else {{ {target} = {values[-1]}; }} }}")
    return "\n".join(lines)


def _emit_continuous_assign(lvalue, dims, lo, hi, indent="        "):
    lines = []
    for idx in _indices(dims):
        target = f"{lvalue}{_subscript(idx)}"
        lines.append(f"{indent}{target} = {PREFIX}rand_range({lo!r}, {hi!r});")
    return "\n".join(lines)


def _emit_output_print(path, dims, indent="        "):
    lines = []
    for idx in _indices(dims):
        target = f"{path}{_subscript(idx)}"
        label = f"{path}{''.join(f'[{i}]' for i in idx)}"
        lines.append(f'{indent}printf("{label}: %.10g\\n", (double)({target}));')
    return "\n".join(lines)


MAX_PER_ITERATION_ELEMENTS = 20  # 单个输入的元素总数（数组/矩阵展开后）超过这个
# 阈值，判定为"批量表"（真实翻车案例：14x14矩阵输入，若每次K循环都重新随机赋值
# 196个元素，K=500时这一个变量就要重复生成98000行赋值语句，harness体积炸到几十
# 万字符，单是这一个变量就让某道"hard"题的harness膨胀到36KB、耗时17分钟）。
# 批量表只在main()开头赋值一次（不放进K循环），不逐迭代重新随机——这类大矩阵
# 更像是"配置好的常量表"而不是需要逐点探测的自由变量，跟旧版 case_enum.py
# 处理同一类问题时的思路一致。


def _element_count(dims):
    if dims is None:
        return 1
    return dims[0] if len(dims) == 1 else dims[0] * dims[1]


def _emit_discrete_bulk_setup(lvalue, dims, values, indent="    "):
    return "\n".join(f"{indent}{lvalue}{_subscript(idx)} = {values[0]};" for idx in _indices(dims))


def _emit_continuous_bulk_setup(lvalue, dims, lo, hi, indent="    "):
    mid = (lo + hi) / 2.0
    return "\n".join(f"{indent}{lvalue}{_subscript(idx)} = {mid!r};" for idx in _indices(dims))


def build_harness_source(ref, discrete, continuous, outputs, k=DEFAULT_K, seed=DEFAULT_SEED, one_line=False):
    """返回 (harness_code, output_ranges)。harness_code 一般是 "}\\nint main(void){...}" 形式，直接接在生成代码
    后面，跟 build_probe_source 的拼接方式一致）。纯 Python 拼接，不经过 LLM。

    是否要在最前面补一个 "}"，取决于 ref["test"] 是不是自己提供收尾括号——
    跟 llm_generate.fix_closing_brace() 依据的是同一条规则：多数题目
    ref["test"] 自己以 "}" 开头，候选（不管是生成的还是 canonical_solution）
    不带收尾括号，harness 要替 ref["test"] 补上这个 "}"；少数题目（比如
    部分函数级题）ref["test"] 不提供收尾括号，候选自己已经带了收尾 "}"，
    harness 就不能再补一个，否则变成"函数体提前收尾+多余括号"。
    output_ranges 是 {harness里实际打印的变量名: (lo,hi)}——键用改写后（参数路径
    变成局部变量名）的名字，跟 parse_probe_output() 解析出来的 key 直接对得上。
    """
    test_has_own_close = (ref.get("test") or "").lstrip().startswith("}")
    params = parse_params_typed(ref.get("declaration") or "")
    param_names = {name for name, _, _ in params}
    extern_types = scan_extern_globals(ref.get("prompt") or "")

    setup_lines = []
    for name, is_pointer, type_name in params:
        var = f"{PREFIX}arg_{name}"
        setup_lines.append(f"static {type_name} {var};")

    extern_pointer_names = {name for name, (_type, is_ptr) in extern_types.items() if is_ptr}

    reset_lines = []
    for name, is_pointer, type_name in params:
        var = f"{PREFIX}arg_{name}"
        reset_lines.append(f"        memset(&{var}, 0, sizeof({type_name}));")
    zeroed_globals = set()
    all_input_paths = [(p, d) for p, d, _ in discrete] + [(p, d) for p, d, _, _ in continuous]
    for path, _dims_ in all_input_paths:
        root = _base_root(path)
        if root in param_names or root in zeroed_globals or root in extern_pointer_names:
            # 指针类型的全局量不做memset清零：memset(&root,...)清的是指针变量
            # 自己所在的内存（把指针本身清成全0，也就是清成NULL），不是它指向
            # 的内存——清成NULL之后任何一次`root->field`赋值都会立刻空指针
            # 崩溃，比"没清零、可能读到脏值"更糟。指针指向的内存本来就不是
            # 我们分配的，不属于我们负责清零的范围，这里直接跳过，只处理
            # 访问符号（.变`->`）这一半问题，这是已知的精度上限。
            continue
        if root in extern_types:
            type_name, _is_ptr = extern_types[root]
            reset_lines.append(f"        memset(&{root}, 0, sizeof({type_name}));")
            zeroed_globals.add(root)

    # 纯输出的全局量（不出现在discrete/continuous输入列表里，只出现在outputs里）
    # 之前完全没有被清零过——上一轮迭代写进去的值会原样残留成这一轮调用前的
    # "初始状态"，如果候选代码在某个分支下不写这个输出字段（比如按条件决定
    # 写不写），打印出来的其实是上一轮的陈旧值，不是这一轮真正算出来的东西。
    # 差分测试（拿一份候选和一份参考实现分别独立跑同一份harness）尤其怕这个：
    # 两边各自的"陈旧残留"完全独立、大概率不一样，会把纯粹的状态残留误判成
    # 两份实现在语义上不一致，真实翻车案例见某内部任务。
    for path, _dims_, _rng_ in outputs:
        root = _base_root(path)
        if root in param_names or root in zeroed_globals or root in extern_pointer_names:
            continue
        if root in extern_types:
            type_name, _is_ptr = extern_types[root]
            reset_lines.append(f"        memset(&{root}, 0, sizeof({type_name}));")
            zeroed_globals.add(root)

    assign_lines = []
    bulk_setup_lines = []
    for path, dims, values in discrete:
        lvalue = _rewrite_lvalue(path, param_names, extern_pointer_names)
        if _element_count(dims) > MAX_PER_ITERATION_ELEMENTS:
            bulk_setup_lines.append(_emit_discrete_bulk_setup(lvalue, dims, values))
        else:
            assign_lines.append(_emit_discrete_assign(lvalue, dims, values))
    for path, dims, lo, hi in continuous:
        lvalue = _rewrite_lvalue(path, param_names, extern_pointer_names)
        if _element_count(dims) > MAX_PER_ITERATION_ELEMENTS:
            bulk_setup_lines.append(_emit_continuous_bulk_setup(lvalue, dims, lo, hi))
        else:
            assign_lines.append(_emit_continuous_assign(lvalue, dims, lo, hi))

    call_args = []
    for name, is_pointer, _ in params:
        var = f"{PREFIX}arg_{name}"
        call_args.append(f"&{var}" if is_pointer else var)
    func_name_match = re.search(r"(\w+)\s*\(", ref.get("declaration") or "")
    func_name = func_name_match.group(1) if func_name_match else "TargetFunc"
    call_line = f"        {func_name}({', '.join(call_args)});"
    ret_print = ""
    if one_line and func_name_match:
        # 非 void、非结构体的返回值（如状态码）也是输出，一并打印
        ret_tokens = [t for t in re.findall(r"[A-Za-z_]\w*", (ref.get("declaration") or "")[:func_name_match.start()])
                      if t not in ("static", "inline", "extern", "const", "volatile")]
        ret_type = ret_tokens[-1] if ret_tokens else "void"
        if ret_type != "void" and ret_type not in _struct_table(ref.get("prompt") or ""):
            setup_lines.append(f"static {ret_type} {PREFIX}ret;")
            call_line = f"        {PREFIX}ret = {func_name}({', '.join(call_args)});"
            ret_print = f'        printf("__ret__: %.10g\\n", (double)({PREFIX}ret)); /*@P*/'

    print_lines = []
    rewritten_output_ranges = {}
    for path, dims, rng in outputs:
        lvalue = _rewrite_lvalue(path, param_names, extern_pointer_names)
        print_lines.append(_emit_output_print(lvalue, dims))
        if rng is not None:
            rewritten_output_ranges[lvalue] = rng

    if one_line:
        # 每条赋值/打印各压成一行并打标记，编译失败时可按 gcc 报错行号整行删掉
        # （fmv_probe.build_harness_pruned），不需要 LLM 修 harness。
        assign_lines = [ln.replace("\n", " ") + " /*@A*/" for ln in assign_lines]
        bulk_setup_lines = [ln.replace("\n", " ") + " /*@A*/" for ln in bulk_setup_lines]
        print_lines = [ln.replace("\n", " ") + " /*@P*/" for ln in print_lines]

    body = "\n".join(filter(None, [
        "#include <string.h>",
        "#include <stdio.h>",  # 部分题（如 GNC-ADCS-100）的上下文不带 stdio.h
        "",
        f"static unsigned int {PREFIX}rng_state = {seed}u;",
        f"static unsigned int {PREFIX}rand_u32(void) {{",
        f"    unsigned int x = {PREFIX}rng_state;",
        "    x ^= x << 13; x ^= x >> 17; x ^= x << 5;",
        f"    {PREFIX}rng_state = x;",
        "    return x;",
        "}",
        f"static double {PREFIX}rand_range(double lo, double hi) {{",
        f"    return lo + (hi - lo) * ((double){PREFIX}rand_u32() / 4294967296.0);",
        "}",
        "",
        "\n".join(setup_lines),
        "",
        "int main(void) {",
        (BULK_BLOCK_BEGIN + "\n".join(bulk_setup_lines) + "\n" + BULK_BLOCK_END) if bulk_setup_lines else "",
        f"    for (int {PREFIX}k = 0; {PREFIX}k < {k}; {PREFIX}k++) {{",
        f'        printf("## %d\\n", {PREFIX}k);',
        "\n".join(reset_lines) if reset_lines else "",
        "\n".join(assign_lines) if assign_lines else "",
        call_line,
        "\n".join(print_lines) if print_lines else "",
        ret_print,
        "    }",
        "    return 0;",
        "}",
    ]))
    harness_code = ("}\n" + body) if test_has_own_close else body
    return harness_code, rewritten_output_ranges


# ---------------------------------------------------------------------------
# 第四步：编译探测（compile-only / 带 UBSan 运行）+ 输出解析
# ---------------------------------------------------------------------------

BULK_BLOCK_PLACEHOLDER = "    // __AEROPROBE_BULK_OMITTED__ (批量查表/常量矩阵初始化，已省略，不需要改这部分)\n"


def extract_bulk_block(main_text):
    """把main()正文里 BULK_BLOCK_BEGIN/END 标记之间的批量查表初始化代码单独
    切出来。返回 (bulk_block原文或None, 用占位注释替换掉bulk块之后的main文本)。

    这段代码专门给harness的patch流程用：批量查表/常量矩阵的初始化是纯Python
    按数值拼出来的、可能长达几十上百行（真实翻车案例：JetAccelerationA这类
    大矩阵），如果把它也原样喂给LLM、要求LLM在patch时"原样保留、只改别处"，
    等于要求LLM一字不差地转写一大段机械重复文本——这正是当初把harness从
    "LLM转写"改成"纯Python生成"想要避免的风险，转写字符越长，中途截断的
    概率越高（真实翻车案例：截断点停在 "__aeroprobe_rand_range" 写到一半的
    "__aerop"，产生新的语法错误）。用占位注释替换之后，LLM看到的main()始终
    是可控大小的，之后把占位注释换回真实内容即可——deterministic字符串替换，
    不需要LLM参与这一步。"""
    start = main_text.find(BULK_BLOCK_BEGIN)
    end = main_text.find(BULK_BLOCK_END)
    if start == -1 or end == -1 or end < start:
        return None, main_text
    bulk_block = main_text[start:end + len(BULK_BLOCK_END)]
    shown = main_text[:start] + BULK_BLOCK_PLACEHOLDER + main_text[end + len(BULK_BLOCK_END):]
    return bulk_block, shown


def restore_bulk_block(patched_text, bulk_block):
    """把 extract_bulk_block 挖掉的批量查表代码换回来。bulk_block 为 None
    （这道题本来就没有批量查表字段）时原样返回，不做任何替换。"""
    if bulk_block is None:
        return patched_text
    return patched_text.replace(BULK_BLOCK_PLACEHOLDER, bulk_block)


def build_probe_source(ref, generation, harness):
    return "\r\n" + ref["prompt"] + generation + "\r\n" + harness


def region_line_ranges(ref, generation, harness):
    """返回 ((候选起始行,候选结束行), harness起始行) ——用于把 gcc 报错行号归因到
    候选自己还是 harness。直接在拼好的真实源码字符串上数换行数，不手算偏移。"""
    prefix = "\r\n" + ref["prompt"]
    source = prefix + generation + "\r\n" + harness
    gen_start_off = len(prefix)
    gen_end_off = gen_start_off + len(generation)
    harness_start_off = gen_end_off + len("\r\n")
    gen_start_line = source[:gen_start_off].count("\n") + 1
    gen_end_line = source[:gen_end_off].count("\n") + 1
    harness_start_line = source[:harness_start_off].count("\n") + 1
    return (gen_start_line, gen_end_line), harness_start_line


_GCC_LOC_RE = re.compile(r"case\.c:(\d+):\d+:")


def classify_compile_error(stderr_text, gen_range, harness_start):
    """返回 (候选自己有责任, harness有责任)。两者都可能同时为真（一次编译报了
    多处错误，分别落在两个区间）；两者都为假时（比如报错没有明确行号，或落在
    prompt 区间——数据本身的问题），默认算候选自己的责任，不留空。"""
    lines = [int(m.group(1)) for m in _GCC_LOC_RE.finditer(stderr_text)]
    if not lines:
        return True, False
    in_gen = any(gen_range[0] <= ln <= gen_range[1] for ln in lines)
    in_harness = any(ln >= harness_start for ln in lines)
    if not in_gen and not in_harness:
        return True, False
    return in_gen, in_harness


def _compile_cmd(project):
    link_libs = (["-lstdutils7_V2_01"] if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "includes", "libstdutils7_V2_01.a")) else []) + ([f"-l{project}"] if project else [])  # gnc100 needs no project library
    return (
        "gcc case.c -I/includes -Wl,--start-group "
        f"-L/includes {' '.join(link_libs)} -Wl,--end-group "
        "-Wno-incompatible-pointer-types -o case.out -lm"
    )


def _compile_cmd_ubsan(project):
    link_libs = (["-lstdutils7_V2_01"] if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "includes", "libstdutils7_V2_01.a")) else []) + ([f"-l{project}"] if project else [])  # gnc100 needs no project library
    return (
        "gcc case.c -I/includes -Wl,--start-group "
        f"-L/includes {' '.join(link_libs)} -Wl,--end-group "
        "-Wno-incompatible-pointer-types -fsanitize=undefined -o case.out -lm"
    )


def compile_probe_only(ref, generation, harness, timeout=10.0):
    """只编译候选+harness（不跑 UBSan、不运行），给编译反馈循环用。
    返回 {"outcome": compiled/compile_error/timeout, "stderr": str}。"""
    source = build_probe_source(ref, generation, harness)
    with tempfile.TemporaryDirectory(prefix="probe_compile_") as workdir:
        with open(os.path.join(workdir, "case.c"), "w", encoding="utf-8") as f:
            f.write(source)
        try:
            compiled = docker_run(workdir, _compile_cmd(ref.get("project")), timeout)
        except subprocess.TimeoutExpired:  # docker 不可用等其他异常必须抛出，不能记成超时
            return {"outcome": "timeout", "stderr": ""}
        if compiled.returncode != 0:
            return {"outcome": "compile_error", "stderr": compiled.stderr or compiled.stdout}
        return {"outcome": "compiled", "stderr": ""}


def run_probe_ubsan(ref, generation, harness, timeout=10.0):
    """编译（带 -fsanitize=undefined）+ 运行一次。返回
    {"outcome": compile_error/timeout/runtime_error/ran, "stdout": str, "stderr": str}。"""
    source = build_probe_source(ref, generation, harness)
    with tempfile.TemporaryDirectory(prefix="probe_ubsan_") as workdir:
        with open(os.path.join(workdir, "case.c"), "w", encoding="utf-8") as f:
            f.write(source)
        try:
            compiled = docker_run(workdir, _compile_cmd_ubsan(ref.get("project")), timeout)
        except subprocess.TimeoutExpired:  # docker 不可用等其他异常必须抛出，不能记成超时
            return {"outcome": "timeout", "stdout": "", "stderr": ""}
        if compiled.returncode != 0:
            return {"outcome": "compile_error", "stdout": "", "stderr": compiled.stderr or compiled.stdout}
        try:
            ran = docker_run(workdir, "./case.out", timeout)
        except subprocess.TimeoutExpired:  # docker 不可用等其他异常必须抛出，不能记成超时
            return {"outcome": "timeout", "stdout": "", "stderr": ""}
        # UBSan 报错默认不改变退出码（continue-on-error），所以 runtime_error 只保留给
        # 真的非零退出（除零崩溃、段错误这类）；UB 本身从 stderr 里判断，不依赖退出码。
        if ran.returncode != 0:
            return {"outcome": "runtime_error", "stdout": ran.stdout, "stderr": ran.stderr}
        return {"outcome": "ran", "stdout": ran.stdout, "stderr": ran.stderr}


# 允许下标出现在路径中间（如 arg_p[0].x，条款探针的指针参数按数组声明）；原先只允许末尾下标
_OUTPUT_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:\[\d+\][A-Za-z0-9_.]*)*)\s*:\s*(-?[\w.eE+-]+)\s*$")
_CASE_HEADER_RE = re.compile(r"^##\s*(\d+)\s*$")


def parse_probe_output(stdout_text):
    """把 "## k" 分隔的 K 组输出解析成 {"k::变量名": float}。nan/inf 也解析出来
    （不是丢弃，是绝对缺陷信号的一部分），Python 的 float() 本身就认得
    "nan"/"inf"/"-inf" 这几种 C 的 printf 输出形式。"""
    values = {}
    current_case = "_default"
    for line in stdout_text.splitlines():
        line = line.strip()
        header_m = _CASE_HEADER_RE.match(line)
        if header_m:
            current_case = header_m.group(1)
            continue
        m = _OUTPUT_LINE_RE.match(line)
        if not m:
            continue
        key, raw = m.groups()
        try:
            values[f"{current_case}::{key}"] = float(raw)
        except ValueError:
            continue
    return values


# ---------------------------------------------------------------------------
# 第五步：缺陷统计（绝对信号）+ 分歧分（相对信号）
# ---------------------------------------------------------------------------

_UB_LINE_RE = re.compile(r"runtime error:.*", re.MULTILINE)
MAX_DEFECT_DETAIL = 10


def count_defects(ref, generation, harness, timeout, output_ranges):
    """跑一次 UBSan 探测，返回 (缺陷数, 缺陷详情文本, run_result, 缺陷签名集合)。
    output_ranges: {输出路径(不带下标): (lo,hi) 或 None}
    缺陷数为 None 表示这次探测本身就没跑起来（编译/运行失败），调用方应当把
    这种情况当成"最差"处理（不是 0，也不是某个具体数字），此时签名集合也是
    None（不是空集——空集表示"探测了，确实没缺陷"，跟"没法探测"是两回事）。

    签名集合是"这个字段/这类问题有没有触发缺陷"这个层级的指纹（下标已归一化
    成[]），不含具体第几次随机迭代、具体数值——单点随机探测在同一份代码上
    重跑一次，因为随机种子/迭代次序不同，命中的具体下标/数值大概率不一样，
    但"哪个字段容易在这份实现下越界/出NaN"这个结构性事实是稳定的。调用方
    拿这个签名集合去跟同一个候选池里其它候选的签名集合比较，只有"整个池子
    里独此一份触发"的签名才有区分度——诊断依据是某内部任务：正确实现偶发的边界
    NaN（跟其余正确实现共享）如果按裸缺陷数直接比较，会跟真正有语义bug的
    候选混在一起分不出高低，但两者的签名集合一对比，前者是全池共有的噪声，
    后者不是（详见 llm_diffrandom.py 里的池内相对打折机制）。
    """
    result = run_probe_ubsan(ref, generation, harness, timeout=timeout)
    if result["outcome"] not in ("ran", "runtime_error"):
        return None, f"探测harness运行失败（{result['outcome']}），无法测量缺陷", result, None

    ub_msgs = _UB_LINE_RE.findall(result["stderr"])
    parsed = parse_probe_output(result["stdout"])

    nan_inf_keys = [k for k, v in parsed.items() if v != v or v in (float("inf"), float("-inf"))]

    range_violations = []
    for key, v in parsed.items():
        if v != v:  # NaN 已经计入上面一类，这里不重复计
            continue
        var = key.split("::", 1)[1]
        base = re.sub(r"\[\d+\]", "", var)
        rng = output_ranges.get(base)
        if rng and not (rng[0] - 1e-6 <= v <= rng[1] + 1e-6):
            range_violations.append((key, v, rng))

    defect_count = len(ub_msgs) + len(nan_inf_keys) + len(range_violations)

    parts = []
    if ub_msgs:
        shown = sorted(set(ub_msgs))[:MAX_DEFECT_DETAIL]
        parts.append("【未定义行为】\n" + "\n".join(shown))
    if nan_inf_keys:
        shown = nan_inf_keys[:MAX_DEFECT_DETAIL]
        parts.append("【NaN/Inf输出】\n" + "\n".join(f"{k}: {parsed[k]}" for k in shown))
    if range_violations:
        shown = range_violations[:MAX_DEFECT_DETAIL]
        parts.append("【值域越界】\n" + "\n".join(f"{k}: {v}（应在{rng}内）" for k, v, rng in shown))
    detail = "\n\n".join(parts)

    signature = set()
    for msg in ub_msgs:
        signature.add(f"未定义行为:{msg}")
    for k in nan_inf_keys:
        field = re.sub(r"\[\d+\]", "[]", k.split("::", 1)[1])
        signature.add(f"NaN/Inf:{field}")
    for key, _v, _rng in range_violations:
        field = re.sub(r"\[\d+\]", "[]", key.split("::", 1)[1])
        signature.add(f"值域越界:{field}")

    return defect_count, detail, result, frozenset(signature)


def divergence_scores(parsed_list):
    """parsed_list: [{"k::var": value}, ...]，每个元素对应一份存活候选的探测输出。
    返回 (scores, common_keys)。scores[i] = 候选i跟其余候选逐输入不一致的次数总和。
    """
    n = len(parsed_list)
    scores = [0] * n
    if n < 2:
        return scores, set()
    common_keys = set(parsed_list[0])
    for p in parsed_list[1:]:
        common_keys &= set(p)
    for key in common_keys:
        vals = [p[key] for p in parsed_list]
        for i in range(n):
            for j in range(n):
                if i != j and not values_close(vals[i], vals[j]):
                    scores[i] += 1
    return scores, common_keys


MAX_DIVERGENCE_FIELDS_SHOWN = 15
MAX_DIVERGENCE_EXAMPLES_PER_FIELD = 3


def build_divergence_table(target_idx, parsed_list, common_keys):
    """按"字段"分组（忽略具体是第几组随机输入，只看变量名本身），每个字段最多
    展示 MAX_DIVERGENCE_EXAMPLES_PER_FIELD 组具体输入的对比，控制反馈体量不随
    K 增大而增大。返回 (文本, 展示了这次分歧的字段名集合)。"""
    by_field = {}
    for key in common_keys:
        _, _, var = key.partition("::")
        vals = [p[key] for p in parsed_list]
        if not all(values_close(vals[target_idx], v) for v in vals):
            by_field.setdefault(var, []).append((key, vals))

    if not by_field:
        return "", set()

    field_scores = {var: len(examples) for var, examples in by_field.items()}
    sorted_fields = sorted(field_scores, key=lambda v: -field_scores[v])
    shown_fields = sorted_fields[:MAX_DIVERGENCE_FIELDS_SHOWN]
    omitted_fields = len(sorted_fields) - len(shown_fields)

    lines = []
    for var in shown_fields:
        examples = by_field[var][:MAX_DIVERGENCE_EXAMPLES_PER_FIELD]
        omitted_examples = len(by_field[var]) - len(examples)
        lines.append(f"字段 `{var}`：")
        for key, vals in examples:
            others = [v for i, v in enumerate(vals) if i != target_idx]
            lines.append(f"  这组输入下，你算出 {vals[target_idx]}，另一份独立实现算出 {others}")
        if omitted_examples > 0:
            lines.append(f"  （另有 {omitted_examples} 组类似分歧未展示）")
    if omitted_fields > 0:
        lines.append(f"（另有 {omitted_fields} 个分歧字段模式类似，未逐一列出）")
    return "\n".join(lines), set(shown_fields)


def build_judge_spec(parsed_list, common_keys, max_fields=MAX_DIVERGENCE_FIELDS_SHOWN):
    """给"存活候选之间有分歧，先问一次能不能免生成"这一步准备盲判数据：找出所有
    存在分歧的字段（跟 build_divergence_table 同样的分组逻辑——按变量名分组，
    忽略具体第几组随机输入），每个字段取一组具体分歧输入做代表，把这组输入下
    各候选的取值按数值去重合并成"选项"（同一个数值只出现一次，不暴露有几份
    候选支持它——这是避免盲判退化成多数投票的关键，不是可选的美化）。

    返回 [(变量名, 具体输入key, [(数值, {支持这个数值的候选下标,...}), ...]), ...]，
    按分歧候选数从大到小排，最多 max_fields 个字段。
    """
    by_field = {}
    for key in common_keys:
        _, _, var = key.partition("::")
        vals = [p.get(key) for p in parsed_list]
        if any(v is None for v in vals):
            continue
        if all(values_close(vals[0], v) for v in vals):
            continue
        by_field.setdefault(var, []).append((key, vals))

    fields = sorted(by_field, key=lambda v: -len(by_field[v]))[:max_fields]
    spec = []
    for var in fields:
        key, vals = by_field[var][0]
        options = []
        for i, v in enumerate(vals):
            for ov, idxs in options:
                if values_close(ov, v):
                    idxs.add(i)
                    break
            else:
                options.append((v, {i}))
        spec.append((var, key, options))
    return spec


# ---------------------------------------------------------------------------
# 输出字段自动推导（2026-10-02）：不依赖 LLM 给出的 outputs 列表
# ---------------------------------------------------------------------------

_STRUCT_DEF_RE = re.compile(r"typedef\s+struct\s*(\w*)\s*\{(.*?)\}\s*([^;]+);", re.S)
_DEFINE_NUM_RE = re.compile(r"#define\s+(\w+)\s+\(?\s*(\d+)\s*[uU]?\s*\)?")


def _strip_c_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _struct_table(prompt):
    """{类型名或 struct tag: [(字段类型, 字段名, dims)]}，dims 为 None/(n,)/(n,m)。"""
    text = _strip_c_comments(prompt)
    defines = {k: int(v) for k, v in _DEFINE_NUM_RE.findall(text)}
    table = {}
    for tag, body, names in _STRUCT_DEF_RE.findall(text):
        fields = []
        for decl in body.split(";"):
            decl = decl.strip()
            m = re.match(r"(?:const\s+|volatile\s+)*(\w+)\s+(.+)$", decl, re.S)
            if not m:
                continue
            ftype, rest = m.group(1), m.group(2)
            for item in rest.split(","):
                item = item.strip()
                if not item or "*" in item or "(" in item:
                    continue
                nm = re.match(r"(\w+)\s*((?:\[[^\]]+\]\s*)*)$", item)
                if not nm:
                    continue
                dims = []
                for d in re.findall(r"\[([^\]]+)\]", nm.group(2)):
                    d = d.strip()
                    dims.append(int(d) if d.isdigit() else defines.get(d))
                if any(x is None for x in dims) or len(dims) > 2:
                    continue
                fields.append((ftype, nm.group(1), tuple(dims) if dims else None))
        for n in [tag] + [x.strip().lstrip("*").strip() for x in names.split(",")]:
            if n:
                table[n] = fields
    return table


def auto_output_specs(ref, max_elements=200):
    """从函数签名的非 const 指针参数出发，列出其结构体（至多再嵌套一层）里的数值字段，
    作为探针要打印的输出。返回与 extract_probe_spec 的 outputs 同格式的列表。"""
    decl = ref.get("declaration") or ""
    table = _struct_table(ref.get("prompt") or "")
    outs, total = [], 0
    for name, is_pointer, type_name in parse_params_typed(decl):
        if not is_pointer or re.search(r"const\s+\w+\s*\*\s*" + re.escape(name) + r"\b", decl):
            continue
        if type_name not in table:  # 指向标量的输出指针，如 double *loss
            if total + 1 <= max_elements:
                outs.append((name, None, None))
                total += 1
            continue
        stack = [(f"{name}", type_name, 0)]
        while stack:
            path, tname, depth = stack.pop(0)
            for ftype, fname, dims in table.get(tname, []):
                if ftype in table:
                    if depth < 1 and dims is None:
                        stack.append((f"{path}.{fname}", ftype, depth + 1))
                    continue
                n = _element_count(dims)
                if total + n > max_elements:
                    continue
                outs.append((f"{path}.{fname}", dims, None))
                total += n
    return outs
