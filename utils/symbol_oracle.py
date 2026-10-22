#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""符号裁判：确定性地检查生成代码里用到的结构体字段名、函数调用，是不是真的
在题目给的上下文（ref["prompt"]、std_utils.h）里声明过。

动机：反复实测证实的一个根因——模型写错字段名/函数名，不是"信息缺失"，
declaration 就在 ref["prompt"] 里，是模型没有真的逐字核对，凭需求里的
数学符号或训练数据里的命名习惯直接下笔（典型形态：连续几轮写出的字段名
压根不在该结构体的声明列表里——不是选错了真实候选，是凭空编的）。

要模型自己每写一个标识符都回头核对一遍声明，代价很高（要么占用有限的注意力，
要么思考模式里多写一大段核对文字）；但反过来，用代码去核对声明是纯正则/
词法分析，几乎零成本。所以这里不指望模型自己更仔细，而是生成完之后用确定性
规则扫一遍，把"引用了不存在的字段/函数"这类硬伤挑出来，附上真实可用的候选
清单喂回去定向修——跟 gcc 编译报错比，这一步不用等真的编译一次，而且能带上
"真实可用的名字有这些"这个 gcc 报错通常不会给全的信息。
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
STD_UTILS_PATH = os.path.join(HERE, "..", "eval", "includes", "std_utils.h")

# std_utils.h 本身 #include <math.h>，这些是 math.h/常见 libc 里合法但不会出现
# 在 std_utils.h 自己 extern 声明里的函数，不能被当成"未声明的函数"误报。
# 踩过的坑：最初只收了双精度版本，漏了 C99 的单精度 f 后缀变体（sinf/cosf/
# sqrtf...），模型合法调用 sinf/cosf/sqrtf 被误判成"凭空编的函数"，触发了
# 一次没必要的 patch。
_LIBC_BASE_NAMES = {
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "sqrt", "exp", "log", "log10", "log2",
    "fabs", "floor", "ceil", "round",
}
_LIBC_WHITELIST = _LIBC_BASE_NAMES | {name + "f" for name in _LIBC_BASE_NAMES} | {
    "atan2", "atan2f", "pow", "powf", "fmod", "fmodf", "hypot", "hypotf",
    "memcpy", "memset", "memmove", "strlen", "strcpy", "strcmp",
    "printf", "assert",
}
# 看起来像函数调用（后面跟括号）但其实是关键字/类型转换的 token，不能当函数查。
_NOT_FUNCTION_TOKENS = {
    "if", "for", "while", "switch", "return", "sizeof", "do", "else",
}

_EXTERN_FUNC_RE = re.compile(r"extern\s+[\w\s\*]+?\b(\w+)\s*\(([^)]*)\)\s*;")
_TYPEDEF_SCALAR_RE = re.compile(r"typedef\s+(?:signed|unsigned|const)?\s*[\w ]+?\s+(\w+)\s*;")
_STDINT_TYPE_RE = re.compile(r"\b(u?int(?:8|16|32|64)_t)\b")
# 题目有时候会在 ref["prompt"] 里直接给一个辅助函数的完整实现（不是 extern
# 声明），形如"float64 Helper(float64 *fx, siint32 n)\n{...}"——要求
# 前面有一个独立的"类型"标识符再跟函数名，天然排除 if(...){ / for(...){
# 这种控制结构（它们前面没有另一个类型 token）。
_FUNC_DEF_RE = re.compile(r"\b[A-Za-z_]\w*[\s\*]+(\w+)\s*\(([^)]*)\)\s*\{")
_DEFINE_NAME_RE = re.compile(r"#define\s+(\w+)\b")
_STRUCT_OPEN_RE = re.compile(r"\bstruct\s*(\w*)\s*\{")
_TYPEDEF_NAME_AFTER_CLOSE_RE = re.compile(r"\A\s*(\w+)\s*;")
_MEMBER_ACCESS_RE = re.compile(r"(?:->|\.)\s*([A-Za-z_]\w*)")
_TYPED_MEMBER_ACCESS_RE = re.compile(r"\b(\w+)\s*(->|\.)\s*([A-Za-z_]\w*)")
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(([^()]*)\)")
_PARAM_LIST_RE = re.compile(r"\(([^)]*)\)")


def parse_param_types(declaration):
    """从函数签名里解析出每个参数的 {参数名: 类型名}（去掉 `*`/数组维度，
    只留类型本身的标识符），配合 parse_struct_fields_by_type 把
    `input.field` 精确核对到 `input` 真实声明的类型上，而不是笼统核对
    "这个字段名是不是在任意结构体里存在过"。"""
    m = _PARAM_LIST_RE.search(declaration or "")
    if not m:
        return {}
    result = {}
    for part in m.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        name_match = re.search(r"(\w+)\s*(?:\[[^\]]*\])?$", part)
        if not name_match:
            continue
        name = name_match.group(1)
        type_tokens = [
            t for t in re.findall(r"[A-Za-z_]\w*", part[:name_match.start()])
            if t not in ("const", "volatile", "static")
        ]
        if type_tokens:
            result[name] = type_tokens[-1]
    return result


def _find_struct_bodies(text):
    """找出所有 struct{...} 定义体的原文（花括号配对的深度扫描，不是简单
    正则）。repo 级别的大结构体经常在里面嵌套匿名子结构体（比如
    struct {...} *aep;），用"不允许嵌套花括号"的正则（[^{}]*）遇到嵌套
    就整个匹配失败——不只是嵌套那部分丢了，外层结构体的其它字段也全部
    跟着丢（实测过：某个结构体因为内嵌了一个匿名子结构体，它的外层字段
    全部没解析出来，官方标准答案因此被误报）。这里手动做花括号深度扫描，
    嵌套多少层都能正确找到
    最外层的配对收尾，嵌套子结构体的字段顺带也会被收进去（对咱们只关心
    "这个名字是不是压根不存在"的场景无害，多收几个真实字段名不会影响
    精度）。"""
    bodies = []
    for m in _STRUCT_OPEN_RE.finditer(text):
        start = m.end()  # 跳过匹配到的开头 { 本身
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            bodies.append(text[start:i - 1])
    return bodies


def _find_struct_bodies_with_names(text):
    """跟 _find_struct_bodies 一样做花括号深度扫描找结构体定义体，但同时
    带上这个结构体的"名字"（`struct TAG {...}` 的 TAG，和/或紧跟在收尾
    `}` 后面、分号前的 typedef 别名——两者可能只有一个，也可能都有，
    都收进同一个别名集合里，只要通过其中任意一个名字引用到了这个类型，
    都算命中这个结构体的字段集合。返回 [(别名集合, 结构体原文), ...]。"""
    result = []
    for m in _STRUCT_OPEN_RE.finditer(text):
        tag_name = m.group(1)
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            continue
        body = text[start:i - 1]
        names = set()
        if tag_name:
            names.add(tag_name)
        typedef_match = _TYPEDEF_NAME_AFTER_CLOSE_RE.match(text[i:])
        if typedef_match:
            names.add(typedef_match.group(1))
        if names:
            result.append((names, body))
    return result


def _strip_comments(text):
    """去掉 // 和 /* */ 注释——需求里常有 Qa(Ctm·Coi) 这类写在注释里的伪代码
    示例调用，字符编码问题还可能把符号变成乱码（比如 Qa(Ctm??Coi)），不strip
    掉会被误当成真实调用去核对参数个数，产生假警报。"""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text


def _arity(params_text):
    params_text = params_text.strip()
    if params_text in ("", "void"):
        return 0
    return len(params_text.split(","))


def _parse_extern_functions(text):
    """从一段 C 代码文本里解析出 extern 函数声明，返回 {函数名: 参数个数}。"""
    sigs = {}
    for m in _EXTERN_FUNC_RE.finditer(text):
        name, params = m.group(1), m.group(2)
        sigs[name] = _arity(params)
    return sigs


_STD_UTILS_FUNCS = None


def _std_utils_functions():
    """std_utils.h 里的函数签名 + 它自己的 #define 宏重定向（比如
    Sinx/Cosx/Fabsx/Sqrtx 分别是 sin/cos/fabs/sqrt 的宏别名，不是 extern
    声明，之前漏了这些，把官方标准答案里合法调用它们的写法误判成了
    "未声明的函数"——20 题里 13 题都因为这个被误报过），只解析一次、缓存住。
    """
    global _STD_UTILS_FUNCS
    if _STD_UTILS_FUNCS is None:
        with open(STD_UTILS_PATH, encoding="utf-8") as f:
            text = f.read()
        funcs = _parse_extern_functions(text)
        for m in _DEFINE_NAME_RE.finditer(text):
            funcs.setdefault(m.group(1), None)
        _STD_UTILS_FUNCS = funcs
    return _STD_UTILS_FUNCS


def _parse_fields_from_body(raw_body):
    """从一个结构体定义体原文里解析出字段名集合（单个结构体范围内）。"""
    fields = set()
    body = _strip_comments(raw_body)
    for raw_line in body.split(";"):
        line = raw_line.strip()
        if not line:
            continue
        # 按逗号切分整行（不是先按空白切出"类型"再处理声明符）——像
        # "const float64 delta0[6]" 这种带限定符的声明，"第一个词当类型"
        # 会把 const 当类型、float64 误判成字段名。改成不管前面有几个
        # 类型/限定符 token，每一段里"最后一个标识符"（去掉数组维度后）
        # 才是字段名，对单声明符和逗号并列多声明符（如
        # "float64 a, b, c"）都成立。
        for segment in line.split(","):
            segment = re.sub(r"\[[^\]]*\]", "", segment)
            tokens = re.findall(r"[A-Za-z_]\w*", segment)
            if tokens:
                fields.add(tokens[-1])
    return fields


def parse_struct_fields(prompt_text):
    """从 ref["prompt"] 里解析出所有结构体/共用体定义里声明的字段名。

    返回一个不区分"这个字段属于哪个具体类型"的扁平集合，只用于兜底——当
    我们没法确定某个变量具体是哪个结构体类型时（比如不是函数参数、类型
    解析失败），退回到"这个名字是不是压根没在任何声明里出现过"这条更
    宽松的判断。精确到具体类型的核对见 parse_struct_fields_by_type()。
    """
    fields = set()
    for raw_body in _find_struct_bodies(prompt_text):
        fields |= _parse_fields_from_body(raw_body)
    return fields


def parse_struct_fields_by_type(prompt_text):
    """跟 parse_struct_fields 一样解析字段，但按结构体类型名分开，返回
    {类型名: 字段集合}（每个结构体的 struct tag 名和 typedef 别名都会
    各自映射到同一份字段集合，用哪个名字查都行）。

    这是为了修一个实测到的真实漏报：符号裁判原来的扁平字段核对只看
    "这个字段名是不是在任意结构体里出现过"，遇到"字段名真实存在，但
    属于另一个结构体"这种张冠李戴（典型形态：模型访问 `input.someField`，
    而 `someField` 其实是同名前缀的 OUTPUT 结构体的字段，INPUT 结构体
    根本没有它）就会漏判——字段名本身能在别处找到，扁平核对就认为
    "存在"，注意不到查错了结构体。
    """
    result = {}
    for names, raw_body in _find_struct_bodies_with_names(prompt_text):
        fields = _parse_fields_from_body(raw_body)
        for name in names:
            result.setdefault(name, set()).update(fields)
    return result


def known_functions_for(ref):
    """合并 std_utils.h 的函数目录 + ref["prompt"] 里题目自己声明/宏重定向/
    直接给出完整实现的函数名，返回 {函数名: 参数个数或 None（宏重定向的
    拿不到参数个数，只校验"存在"，不校验参数个数）}。"""
    funcs = dict(_std_utils_functions())
    prompt_text = _strip_comments(ref.get("prompt") or "")
    funcs.update(_parse_extern_functions(prompt_text))
    for m in _FUNC_DEF_RE.finditer(prompt_text):
        name, params = m.group(1), m.group(2)
        if name not in _NOT_FUNCTION_TOKENS:
            funcs.setdefault(name, _arity(params))
    for m in _DEFINE_NAME_RE.finditer(prompt_text):
        funcs.setdefault(m.group(1), None)
    return funcs


_STD_UTILS_TYPES = None


def _std_utils_scalar_types():
    """std_utils.h 里定义的标量类型别名（siint32/unint32/float32/float64...），
    只解析一次、缓存住。"""
    global _STD_UTILS_TYPES
    if _STD_UTILS_TYPES is None:
        with open(STD_UTILS_PATH, encoding="utf-8") as f:
            text = f.read()
        _STD_UTILS_TYPES = set(_TYPEDEF_SCALAR_RE.findall(text))
    return _STD_UTILS_TYPES


CATALOG_PATH = os.path.join(HERE, "data", "code_rag_data.txt")
_EXTERN_FULL_RE = re.compile(r"extern\s+[^;{}]+\([^)]*\)\s*;")
_CATALOG_DESC = None
_CATALOG_SIGS = None


def _catalog_descriptions():
    """解析检索语料库 code_rag_data.txt（"中文描述 函数名"，空行分隔），
    返回 {函数名: 中文描述}。"""
    global _CATALOG_DESC
    if _CATALOG_DESC is None:
        desc = {}
        with open(CATALOG_PATH, encoding="utf-8") as f:
            for block in f.read().split("\n\n"):
                block = block.strip()
                if not block:
                    continue
                d, _, name = block.rpartition(" ")
                if d and name:
                    desc[name.strip()] = d.strip()
        _CATALOG_DESC = desc
    return _CATALOG_DESC


def _std_utils_signatures():
    """解析 std_utils.h 里每个 extern 函数声明的完整原文（含返回类型、
    参数类型），返回 {函数名: 声明原文}。"""
    global _CATALOG_SIGS
    if _CATALOG_SIGS is None:
        with open(STD_UTILS_PATH, encoding="utf-8") as f:
            text = f.read()
        sigs = {}
        for m in _EXTERN_FULL_RE.finditer(text):
            line = " ".join(m.group(0).split())
            name_m = re.search(r"\b(\w+)\s*\(", line)
            if name_m:
                sigs.setdefault(name_m.group(1), line)
        _CATALOG_SIGS = sigs
    return _CATALOG_SIGS


def describe_catalog_entry(name):
    """确定性兜底：检索到的候选函数名之前只拼了个光秃秃的名字进 prompt
    （连语料库自己有的中文描述都没带上，更不用说参数签名），模型不知道
    怎么调用只能瞎猜。这里把 code_rag_data.txt 的中文描述 + std_utils.h
    里已经声明好的真实签名（返回类型、参数类型个数）拼成一行，两者都是
    题目原本就有的确定性信息，只是之前没被查出来用上。查不到签名（比如
    这个名字其实是个宏定义，不在 std_utils.h 的 extern 声明里）就只带
    描述；两者都没有就退化成裸名字。"""
    desc = _catalog_descriptions().get(name, "")
    sig = _std_utils_signatures().get(name, "")
    if desc and sig:
        return f"- {name}：{desc}；签名：{sig}"
    if sig:
        return f"- {name}：签名：{sig}"
    if desc:
        return f"- {name}：{desc}"
    return f"- {name}"


def known_types_for(ref):
    """合并 std_utils.h + ref["prompt"] 里 typedef 出来的标量类型名。"""
    types = set(_std_utils_scalar_types())
    types.update(_TYPEDEF_SCALAR_RE.findall(ref.get("prompt") or ""))
    return types


_TYPEDEF_FULL_RE = re.compile(r"typedef\s+([A-Za-z_ ]+?)\s+(\w+)\s*;")
_DEFINE_CONST_RE = re.compile(r"^[ \t]*#define[ \t]+(\w+)[ \t]+(\S.*?)(?:[ \t]*/\*(.*?)\*/)?[ \t]*$", re.MULTILINE)
_PURE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")
_STD_UTILS_TYPEDEF_PAIRS = None
_STD_UTILS_CONSTANTS = None


def _std_utils_typedef_pairs():
    """std_utils.h 里的标量类型别名，(别名, 底层类型) 对，按声明顺序，只解析一次。"""
    global _STD_UTILS_TYPEDEF_PAIRS
    if _STD_UTILS_TYPEDEF_PAIRS is None:
        with open(STD_UTILS_PATH, encoding="utf-8") as f:
            text = f.read()
        _STD_UTILS_TYPEDEF_PAIRS = [
            (name, " ".join(base_type.split())) for base_type, name in _TYPEDEF_FULL_RE.findall(text)
        ]
    return _STD_UTILS_TYPEDEF_PAIRS


def _std_utils_constants():
    """std_utils.h 里的数值常量宏（PI/DEG2RAD/EARTH_R...），按声明顺序，
    只解析一次。排除两类不是"常量"的 #define：没有值的头文件保护宏
    （#define STD_UTILS_H）、值是裸标识符的函数重定义宏（#define Sinx sin
    这类，不是数值/表达式）。返回 [(名字, 值, 注释)]。"""
    global _STD_UTILS_CONSTANTS
    if _STD_UTILS_CONSTANTS is None:
        with open(STD_UTILS_PATH, encoding="utf-8") as f:
            text = f.read()
        consts = []
        for name, value, comment in _DEFINE_CONST_RE.findall(text):
            value = value.strip()
            if _PURE_IDENTIFIER_RE.match(value):
                continue
            consts.append((name, value, comment.strip()))
        _STD_UTILS_CONSTANTS = consts
    return _STD_UTILS_CONSTANTS


def project_type_and_constant_context():
    """确定性兜底：论文里说"数据类型约定"和"常用变量表"因为体量小，全量
    塞进检索上下文里"，但实际检索管线（retrivel_*.py）背后的语料库
    code_rag_data.txt 只有函数条目，从没真的把 std_utils.h 里的类型别名/
    数值常量宏喂给过模型——模型只能凭训练数据里的常见写法去猜（典型形态：
    猜 PI 存在但自己重新声明一份同名局部变量导致宏重定义编译报错，或者
    猜错宏名比如 DEG_TO_RAD 而不是项目实际用的 DEG2RAD）。这里补上这个
    缺口：把 std_utils.h 的类型别名和数值常量宏原样格式化成一段文本，
    不依赖检索、每次生成都固定拼进 prompt。"""
    lines = ["#项目通用类型约定（std_utils.h，直接使用，不要自己重新typedef同名类型）："]
    for name, base in _std_utils_typedef_pairs():
        lines.append(f"- {name}: {base}")
    lines.append("")
    lines.append("#项目通用常量宏（std_utils.h，直接引用，不要自己重新声明同名变量/宏）：")
    for name, value, comment in _std_utils_constants():
        lines.append(f"- {name} = {value}" + (f"  /* {comment} */" if comment else ""))
    return "\n".join(lines)
_BASE_TYPE_SIGNATURE = {
    "signed char": (8, True), "unsigned char": (8, False),
    "short": (16, True), "short int": (16, True),
    "unsigned short": (16, False), "unsigned short int": (16, False),
    "int": (32, True), "signed int": (32, True), "unsigned int": (32, False), "unsigned": (32, False),
    "long long": (64, True), "long long int": (64, True),
    "unsigned long long": (64, False), "unsigned long long int": (64, False),
}
_STDINT_NAME_BY_SIGNATURE = {
    (8, True): "int8_t", (8, False): "uint8_t",
    (16, True): "int16_t", (16, False): "uint16_t",
    (32, True): "int32_t", (32, False): "uint32_t",
    (64, True): "int64_t", (64, False): "uint64_t",
}
_STDINT_TO_PROJECT_TYPE = None


def _stdint_to_project_type():
    """从 std_utils.h 的 typedef 定义里，按位宽+符号推出"标准 C99 类型名 ->
    项目自己类型名"的确定性映射（比如 unsigned short -> unint16，对应
    uint16_t -> unint16）。这是项目类型系统跟 stdint.h 本来就有的对应
    关系，不是猜的——只解析一次、缓存住。"""
    global _STDINT_TO_PROJECT_TYPE
    if _STDINT_TO_PROJECT_TYPE is None:
        with open(STD_UTILS_PATH, encoding="utf-8") as f:
            text = f.read()
        mapping = {}
        for base_type, name in _TYPEDEF_FULL_RE.findall(text):
            signature = _BASE_TYPE_SIGNATURE.get(" ".join(base_type.split()))
            if signature is None:
                continue
            mapping.setdefault(_STDINT_NAME_BY_SIGNATURE[signature], name)
        _STDINT_TO_PROJECT_TYPE = mapping
    return _STDINT_TO_PROJECT_TYPE


def fix_stdint_types(generation):
    """确定性兜底：项目自己有一套等宽 typedef（siint32/unint32/float64...），
    这个数据集只 #include "std_utils.h"，没 #include <stdint.h>，模型写了
    标准 C99 的 uint16_t/int32_t 这类名字必然编译不过（实测过多题因为
    这个编译失败，之前的做法是查出来当违规喂回去让模型自己"从候选里选
    一个位宽匹配的"，多占一轮 LLM patch，还是要模型自己选对）。
    位宽+符号是完全确定的对应关系，
    不是猜候选——不用等 gcc 报错、也不用占用一轮 LLM patch，直接换成
    项目自己对应的类型名。
    """
    for stdint_name, project_name in _stdint_to_project_type().items():
        generation = re.sub(rf"\b{stdint_name}\b", project_name, generation)
    return generation


# 项目自己在 std_utils.h 里声明了标准库内存/字符串函数的等价替身（参数顺序
# 跟标准库完全一致，纯粹是名字不同：项目的命名习惯是首字母大写再后缀一个 x），
# 数据集不 #include <string.h>，模型写标准库名字必然是"隐式声明"编译错误。
# 映射不硬编码，而是按这条命名约定去 std_utils.h 的声明里查证——查得到才建立
# 映射，查不到就不动，避免凭空造出一个库里不存在的函数名。
_LIBC_MEM_FUNC_CANDIDATES = ("memcpy", "memset", "memmove", "strlen", "strcpy", "strcmp")
_LIBC_MEM_FUNC_ALIASES = None


def _libc_mem_func_aliases():
    """{标准库函数名: 项目里对应的替身名}，按命名约定从 std_utils.h 推出来，
    只解析一次、缓存住。"""
    global _LIBC_MEM_FUNC_ALIASES
    if _LIBC_MEM_FUNC_ALIASES is None:
        declared = _std_utils_functions()
        _LIBC_MEM_FUNC_ALIASES = {
            name: name.capitalize() + "x"
            for name in _LIBC_MEM_FUNC_CANDIDATES
            if name.capitalize() + "x" in declared
        }
    return _LIBC_MEM_FUNC_ALIASES


def fix_libc_mem_functions(generation):
    """确定性兜底：见 _libc_mem_func_aliases 说明——把标准库函数名换成项目
    自己在 std_utils.h 里声明的同语义替身（参数顺序不变）。
    """
    for libc_name, project_name in _libc_mem_func_aliases().items():
        generation = re.sub(rf"\b{libc_name}\b(?=\s*\()", project_name, generation)
    return generation


def find_violations(generation, ref):
    """扫描生成代码，返回违规描述列表（中文，可以直接拼进反馈文本）。
    没有违规时返回空列表。"""
    generation = _strip_comments(generation)
    known_fields = parse_struct_fields(ref.get("prompt") or "")
    fields_by_type = parse_struct_fields_by_type(ref.get("prompt") or "")
    param_types = parse_param_types(ref.get("declaration") or "")
    known_funcs = known_functions_for(ref)

    violations = []
    seen_fields = set()
    for m in _TYPED_MEMBER_ACCESS_RE.finditer(generation):
        receiver, name = m.group(1), m.group(3)
        key = (receiver, name)
        if key in seen_fields:
            continue

        # 优先按"这个变量真实声明的类型"精确核对——只查函数参数（input/
        # output 这类），能覆盖实测到的真实漏报：字段名本身在题目里某个
        # 别的结构体里确实存在，但不属于这个变量的类型（实测过：模型访问
        # `input.someField`，这个字段名真实存在，但它是同名前缀的 OUTPUT
        # 结构体的字段，`input` 的 INPUT 类型根本没有它——原来"只要字段名
        # 存在于任意结构体就算数"的扁平核对完全漏判了这种张冠李戴）。
        receiver_type = param_types.get(receiver)
        if receiver_type and receiver_type in fields_by_type:
            if name not in fields_by_type[receiver_type]:
                seen_fields.add(key)
                violations.append(
                    f"访问了 `{receiver}.{name}`，但 `{receiver}` 的类型是 `{receiver_type}`，"
                    f"这个类型定义里没有 `{name}` 这个字段——注意别跟别的结构体的同名/相似字段搞混了；"
                    f"`{receiver_type}` 真实存在的字段有：{sorted(fields_by_type[receiver_type])}"
                )
            continue

        # 退回扁平核对：不知道这个变量具体是哪个类型（不是函数参数，或者
        # 类型解析不出字段），只能查"这个名字是不是压根没在任何声明里
        # 出现过"，覆盖面比精确核对窄，但不会比原来的行为更差。
        if name not in known_fields:
            seen_fields.add(key)
            violations.append(
                f"访问了字段 `{name}`，但这个名字没有在题目给的结构体定义里出现过；"
                f"结构体定义里真实存在的字段有：{sorted(known_fields)}"
            )

    # repo 级别的题目（有 ref["project"]）可能合法调用同一个项目里其它文件
    # 定义的函数——这些函数不会出现在当前题目的 ref["prompt"] 里，链接时
    # 靠项目自己的静态库（libp1.a~libp7.a）补上，我们这里看不到全貌。实测
    # 过 212 题全量跑一遍：字段名检查在 function/repo 两类题目上都干净，
    # 但"函数压根没声明过"这条在 repo 题目上有 32 处误报（全是这类跨文件
    # 调用），所以"未声明函数"这条只在 function 级别的题目上查；"参数
    # 个数对不上"这条不受影响，两类题目都查——只要我们确实在 ref["prompt"]
    # 或 std_utils.h 里找到了这个函数的声明，参数个数就是可靠的真相，不
    # 依赖"是不是同一个项目里的其它文件"。
    is_repo_level = bool(ref.get("project"))

    seen_calls = set()
    for m in _CALL_RE.finditer(generation):
        name, params_text = m.group(1), m.group(2)
        if name in _NOT_FUNCTION_TOKENS or name in seen_calls:
            continue
        if name in known_funcs:
            expected = known_funcs[name]
            if expected is not None and _arity(params_text) != expected:
                seen_calls.add(name)
                violations.append(
                    f"调用 `{name}` 传了 {_arity(params_text)} 个参数，"
                    f"但它的声明要求 {expected} 个参数。"
                )
            continue
        if name in _LIBC_WHITELIST or is_repo_level:
            continue
        seen_calls.add(name)
        violations.append(
            f"调用了 `{name}`，但这个函数名既不在 std_utils.h 目录里，"
            f"也没有在题目上下文里声明过——它可能是凭需求文字编出来的名字，"
            f"不是真实存在的函数。"
        )

    # 数据类型也照项目自己的约定来，不要用标准 C99 的 uint8_t/int32_t 这类
    # 写法——这个数据集只 #include "std_utils.h"，没 #include <stdint.h>，
    # 项目自己的类型系统是 siint32/unint32/float32/float64 这套命名（实测过
    # 多题因为写了 uint8_t/int32_t 编译失败）。
    known_types = known_types_for(ref)
    seen_types = set()
    for m in _STDINT_TYPE_RE.finditer(generation):
        name = m.group(1)
        if name in known_types or name in seen_types:
            continue
        seen_types.add(name)
        violations.append(
            f"用了标准 C99 类型 `{name}`，但这个项目没有 #include <stdint.h>，"
            f"没有这个类型——这个项目自己的类型命名是：{sorted(known_types)}，"
            f"应该从里面选一个位宽匹配的。"
        )

    return violations
