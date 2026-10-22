#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调用 OpenRouter 上的 LLM 生成/修复 C 函数实现。

对外的核心入口是 generate_solution()：给定一道题（ref）、检索到的候选函数
说明、生成模式（new/patch），返回一段可以直接拼进 eval.py::build_source()
的函数体（不重复写函数签名、不带结尾的右花括号，和数据集里
canonical_solution 的约定完全一致）。
"""
import re
import time

import requests

import symbol_oracle
import openrouter_config
from openrouter_config import (
    DEFAULT_MAX_TOKENS, MAX_RETRIES, REQUEST_TIMEOUT,
    SAMPLING_PARAMS, resolve_model,
)


_LEADING_PP_RE = re.compile(r"\A(?:[ \t]*(?:#[^\n]*)?\n)*")


def _split_leading_requirement(prompt):
    """返回 (前导预处理行, 需求注释正文, 注释之后的部分)；没有需求注释时注释正文为 None。
    需求注释前允许有 #include 等预处理行和空行（GNC-ADCS-100 的 repo 题就是先
    #include 再写需求注释；只认"以 /* 开头"会把这 50 题的需求整段丢掉）。"""
    lead = _LEADING_PP_RE.match(prompt).group(0)
    text = prompt[len(lead):]
    if text.startswith("/*"):
        end = text.find("*/")
        if end != -1:
            return lead, text[2:end], text[end + 2:]
    return lead, None, text


def extract_requirement(prompt):
    """从题目的 prompt 字段里取出开头 /* 需求如下：... */ 的自然语言部分。"""
    _lead, comment, _rest = _split_leading_requirement(prompt)
    if comment is None:
        return ""
    return re.sub(r"^需求如下[：:]\s*\n?", "", comment.strip())


def strip_requirement_comment(prompt_text):
    """给"展示为C代码上下文"这个场景专用：把开头 /* 需求如下：...*/ 这段
    注释切掉。ref["prompt"] 本身不能动它——build_source() 拼真实编译源码、
    行号定位（generation_line_range 等）都依赖 ref["prompt"] 的原始长度不变，
    改了会牵连一大片。这里只是在"需求已经单独用 extract_requirement() 抽出来
    展示过一遍"的场景下，给"#C代码上下文"这一栏用一份不重复的版本——不然
    需求文本会在同一份 prompt 里连续出现两遍（真实测过：一道题的需求文本
    平均2000+字符，每次调用白白多花千把token，且是每次 generate/patch 调用
    都会发生，不是偶发的）。"""
    lead, comment, rest = _split_leading_requirement(prompt_text)
    if comment is None:
        return prompt_text
    return lead.strip("\n") + ("\n" if lead.strip() else "") + rest.lstrip("\n")


_FENCE_OPEN_RE = re.compile(r"```(?:c|C)?\s*\n")


def extract_c_code(text):
    """从 LLM 回复里取出 ```c ...``` 代码块。

    优先匹配闭合的代码块；如果回复被截断（比如撞到 max_tokens 上限），只有
    开头的 ``` 没有结尾，退化成"开头 ``` 之后的所有内容"，而不是把裸的
    ``` 标记也当成代码混进去编译——那样必然编译失败，还会掩盖"是不是被
    截断了"这个真正的问题。真的一个代码块标记都没有，才把整段回复原样
    当代码用。
    """
    match = re.search(r"```(?:c|C)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip("\n")
    open_match = _FENCE_OPEN_RE.search(text)
    if open_match:
        return text[open_match.end():].strip("\n")
    return text.strip()


def try_extract_c_code(text):
    """同 extract_c_code，但没有任何代码块标记时返回 None（不把整段回复
    当代码），给 ReAct 这种"Thought 里不一定带代码"的场景用。"""
    match = re.search(r"```(?:c|C)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip("\n")
    open_match = _FENCE_OPEN_RE.search(text)
    if open_match:
        return text[open_match.end():].strip("\n")
    return None


def call_llm(model_key, messages, thinking=None, temperature=None, max_tokens=None, **sampling_overrides):
    """调用 LLM chat/completions（OpenRouter 或 DeepSeek 自己的 API，看
    model_key 对应哪个 provider）。返回 (content, usage_dict)。

    usage_dict: {prompt_tokens, completion_tokens, total_tokens, elapsed_time}
    sampling_overrides：临时覆盖 SAMPLING_PARAMS 里的某个字段（比如 presence_penalty=0.0），
    不影响全局默认值，给"对比不同采样参数效果"这种一次性诊断实验用。这套采样参数是
    照 Qwen3.5 官方 best practice 调的，只对 OpenRouter 那几个 Qwen 模型生效；
    DeepSeek 走自己的默认采样行为，不套用这套参数。
    """
    config = resolve_model(model_key)
    provider = openrouter_config.PROVIDERS[config["provider"]]
    base_url = provider["base_url"]
    api_key = provider["load_api_key"]()

    payload = {
        "model": config["model_name"],
        "messages": messages,
        "max_tokens": DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
    }
    if config["provider"] == "openrouter":
        thinking_on = config["thinking"] if thinking is None else thinking
        # 这几个模型默认开思考模式（返回 reasoning 字段、content 为空），
        # 必须显式传 reasoning.enabled 才能关掉——已用真实请求验证过。
        payload.update(SAMPLING_PARAMS[thinking_on])
        payload["reasoning"] = {"enabled": thinking_on}
        # 2026-10-07：可用环境变量固定 OpenRouter 供应商（如 akashml），保证全程同一精度的部署；未设置时行为不变。
        pinned = __import__("os").environ.get("OPENROUTER_PROVIDER")
        if pinned:
            payload["provider"] = {"order": [pinned], "allow_fallbacks": False}
            # 2026-10-08：OPENROUTER_FALLBACK="deepinfra,parasail" 时，固定供应商限流可切到同精度（FP8）的其他供应商
            fb = __import__("os").environ.get("OPENROUTER_FALLBACK")
            if fb:
                payload["provider"] = {"order": [pinned] + [x for x in fb.split(",") if x], "allow_fallbacks": False,
                                       "quantizations": ["fp8"]}
    elif config["provider"] == "deepseek":
        thinking_on = config["thinking"] if thinking is None else thinking
        # DeepSeek 自己的思考开关格式跟 OpenRouter 不一样，是
        # {"type": "enabled"/"disabled"} 这种结构体，不是布尔值/reasoning.enabled——
        # 已用真实请求逐个试过 thinking=False/enable_thinking/reasoning.enabled 这几种
        # 常见写法，只有这个格式真的生效（其它写法要么报错、要么完全不起作用，
        # reasoning_content 照样非空）。默认关闭，跟 Qwen 那几个模型的默认行为一致。
        payload["thinking"] = {"type": "enabled" if thinking_on else "disabled"}
    payload.update(sampling_overrides)
    if temperature is not None:
        payload["temperature"] = temperature
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_error = None
    for attempt in range(MAX_RETRIES):
        start = time.monotonic()
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            last_error = e
            time.sleep(2 ** attempt)
            continue
        elapsed = time.monotonic() - start

        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 402:
            # 账户额度不足，重试没有意义，直接抛出让上层决定是否等待后重跑。
            raise RuntimeError(f"额度不足 (HTTP 402): {resp.text[:300]}")
        resp.raise_for_status()

        data = resp.json()
        if "choices" not in data or not data["choices"]:
            # HTTP 200 但上游模型/供应商出错时，错误会放在 body 里而不是状态码上
            # （比如 {"error": {...}}），当成可重试的瞬时错误处理。
            last_error = RuntimeError(f"响应里没有 choices 字段: {str(data)[:300]}")
            time.sleep(2 ** attempt)
            continue

        message = data["choices"][0]["message"]
        plog = __import__("os").environ.get("OPENROUTER_PROVIDER_LOG")
        if plog and data.get("provider"):  # 记录每次调用实际由哪家供应商提供（论文中报告比例）
            with open(plog, "a", encoding="utf-8") as _f:
                _f.write(f"{data.get('provider')}\n")
        content = message.get("content") or ""
        usage = data.get("usage", {})
        return content, {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "elapsed_time": elapsed,
            # 思考模式打开时，模型的推理过程在这个独立字段里，不会混进 content——
            # 留着给 record.py 记录，方便对比"思考内容"和最终代码。OpenRouter 的
            # Qwen 模型叫 reasoning，DeepSeek 自己的 API 叫 reasoning_content，
            # 两个字段名都试一下。
            "reasoning": message.get("reasoning") or message.get("reasoning_content") or "",
        }

    raise RuntimeError(f"LLM 调用重试 {MAX_RETRIES} 次仍然失败：{last_error}")


CATALOG_CONTEXT_TEMPLATE = """可能有用的基础函数库条目（仅供参考，不必全部用到）：
{catalog_lines}"""


def _format_catalog(retrieved):
    """检索到的候选函数名之前只拼了个裸名字，模型不知道怎么调用只能瞎猜
    参数个数/类型（典型形态：检索命中的库函数只显示成一个光秃秃的名字，
    模型看不到 std_utils.h 里其实早就声明好的完整签名）。这里把语料库
    自带的中文描述、std_utils.h 里已经
    声明好的真实签名都拼上——两者都是确定性信息，不需要模型自己猜。"""
    if not retrieved:
        return "（无检索结果）"
    return "\n".join(symbol_oracle.describe_catalog_entry(name) for name in retrieved)


NO_OVERINTERPRET_COMMENTS_INSTRUCTION = (
    "- 需求文本和C代码上下文里的注释、备注、插入说明（不管是C语言的 /* */ 注释，还是需求正文里"
    "夹带的补充解释/提醒），只是帮助你理解背景，本身不是需要你额外实现的逻辑步骤。只根据需求里"
    "明确给出的公式、伪代码、赋值关系逐条实现；某个符号在公式里已经写清楚怎么算/怎么引用了，就"
    "完全按公式来，不要因为某处注释的措辞看着还有别的含义，就自己发明一层换算或者额外分支——"
    "注释的表述和公式的直接用法对不上时，以公式的直接用法为准。"
)


def build_generate_prompt(ref, retrieved, requirement, understanding=None, extra_instruction=None, diagnosis=None):
    catalog_block = CATALOG_CONTEXT_TEMPLATE.format(catalog_lines=_format_catalog(retrieved))
    type_const_block = symbol_oracle.project_type_and_constant_context()
    understanding_block = (
        f"\n#需求理解（符号映射/辅助函数映射/公式翻译，这是提前做好的翻译准备工作，"
        f"写代码时作为主要参考，能省掉你自己重新翻译一遍符号的功夫）\n{understanding}\n"
        f"\n**但这张表本身也可能有错**（比如凭命名习惯联想出一个实际并不存在的字段名/"
        f"参数名）——如果发现表里某一条跟下面的"
        f"「C代码上下文」对不上（比如提到了一个上下文里根本没有声明过的标识符），"
        f"以C代码上下文本身为准，不要被这张表带偏；C代码上下文才是唯一确定真实存在"
        f"哪些字段/参数的依据。\n"
        if understanding else ""
    )
    # diagnosis：跟 understanding 不是一回事——understanding 是需求本身的符号/公式
    # 翻译准备工作；diagnosis 是"之前几份独立候选实际跑出来之后，发现彼此有分歧，
    # 逐项核对需求后定位到的具体错误点+正确算法"，是错误诊断，不是需求翻译，
    # 不能借用"需求理解...不要自己重新翻译"这个标题（这次不是让模型别重新翻译，
    # 是让模型别重蹈诊断里指出的那些具体错误）。
    diagnosis_block = (
        f"\n#之前几份独立候选的错误诊断（这些候选各自跑出来后发现互相有分歧，"
        f"已经对照需求逐项核对定位过具体错在哪一步、该怎么算——这次重新实现时"
        f"请避免下面指出的这些具体错误，不是要你重新翻译需求）\n{diagnosis}\n"
        if diagnosis else ""
    )
    extra_instruction_line = f"\n{extra_instruction}" if extra_instruction else ""
    return f"""你是一名优秀的航天嵌入式软件工程师。请根据下面的需求和 C 代码上下文，实现指定函数的函数体。

#需求
{requirement}

#C代码上下文（结构体定义、全局变量声明、函数签名）
```c
{strip_requirement_comment(ref["prompt"])}
```

{type_const_block}

{catalog_block}
{understanding_block}{diagnosis_block}
#输出要求
- 只输出函数体的内容（从函数签名后面的第一行开始写，到 return 为止），不要重复写函数签名，不要写函数最后的右花括号 "}}"。
- 用 ```c ...``` 代码块包裹你的答案，代码块里只放函数体，不要有多余说明文字。{extra_instruction_line}
"""


def build_patch_prompt(ref, prev_generation, feedback_text, requirement):
    return f"""你是一名优秀的航天嵌入式软件工程师。下面是你上一轮为该函数写的实现，以及它存在的问题，请只做**最小必要的修改**来解决这些问题，其余代码保持不变。

#需求
{requirement}

#C代码上下文
```c
{strip_requirement_comment(ref["prompt"])}
```

#上一轮实现
```c
{prev_generation}
```

#需要解决的问题
{feedback_text}

#输出要求
用一个或多个"查找/替换"块描述你的修改，不要输出整个函数体——你只需要精确定位
要改的地方，不需要把没改的部分再抄一遍。每个块格式如下：
<<<<<<< SEARCH
（从"上一轮实现"里逐字摘抄的一段原文，改动只涉及的最小范围就够，不要贪多）
=======
（替换成的新代码）
>>>>>>> REPLACE

- SEARCH 部分必须和"上一轮实现"逐字一致（包括缩进和空白），且在整份实现里唯一
  出现一次——写太短容易在别处重复出现，写太长容易打错字，找一段刚好能唯一
  定位、又抄得准的范围。
- 需要改多处就写多个块，块之间不需要额外说明。
- 不要输出除查找/替换块之外的任何说明文字，也不要用代码块包裹这些块本身。
"""


_SEARCH_REPLACE_BLOCK_RE = re.compile(
    r"<{3,}\s*SEARCH\s*\n(.*?)\n={3,}\s*\n(.*?)\n>{3,}\s*REPLACE",
    re.DOTALL,
)
_DIFF_MARKER_RE = re.compile(r"<{3,}|>{3,}")


def parse_search_replace_blocks(content):
    """解析 LLM 回复里的查找/替换块，返回 [(search_text, replace_text), ...]。"""
    return [(m.group(1), m.group(2)) for m in _SEARCH_REPLACE_BLOCK_RE.finditer(content)]


def apply_search_replace_blocks(prev_generation, blocks):
    """把查找/替换块依次应用到 prev_generation 上。

    每一块的 SEARCH 文本必须在当前内容里唯一出现一次才应用——找不到或者
    出现了不止一次（说明模型摘抄得不够精确，无法确定改哪一处）都判定整体
    失败，不做"猜一个位置改"这种不确定的事。返回 (结果, 是否全部应用成功)。
    """
    result = prev_generation
    for search, replace in blocks:
        if result.count(search) != 1:
            return prev_generation, False
        result = result.replace(search, replace, 1)
    return result, True


def generate_solution(model_key, ref, retrieved=None, mode="new",
                       prev_generation=None, feedback_text=None, thinking=None, max_tokens=None,
                       understanding=None, requirement_override=None, extra_instruction=None, diagnosis=None):
    """生成/修复一次函数体。

    mode="new"（默认）：无视上一轮结果，从需求重新完整实现。
    mode="patch"：基于 prev_generation + feedback_text 做定向最小修改。
    两种模式返回值格式一致，都是可以直接拼进 build_source() 的函数体字符串。
    thinking=None 时用模型配置里的默认值（默认关闭）；显式传 True/False 覆盖。
    max_tokens=None 时用 DEFAULT_MAX_TOKENS；思考模式下模型可能光推理就把预算
    用完、完全没有余量写代码，可以显式调大覆盖。
    understanding：llm_understand.understand() 产出的结构化"符号映射/公式翻译"
    文本，非 mode="patch" 时会拼进生成 prompt 里当权威上下文用。
    requirement_override：不为 None 时，用这份文本替代 extract_requirement(ref["prompt"])
    抽出来的需求——给"换措辞复述需求"这类实验用，C代码上下文（struct/签名）仍然
    照常来自 ref["prompt"]，只有自然语言需求这一段被替换。
    extra_instruction：不为 None 时，追加到 mode="new" 的输出要求末尾的一段额外指令
    文本（比如 NO_OVERINTERPRET_COMMENTS_INSTRUCTION）——mode="patch" 不使用这个参数，
    patch 场景是定向最小修改，不走这条"从需求重新实现"的路径。
    diagnosis：不为 None 时，拼进 mode="new" 生成 prompt 里的一段"之前几份独立候选
    的错误诊断"（比如 reconcile.judge_diverging_fields() 产出的根因诊断）——跟
    understanding 不是一回事：understanding 是需求本身的符号/公式翻译准备工作，
    diagnosis 是已经跑出来、发现互相分歧、逐项核对需求后定位到的具体错误点，
    两者可以同时传、互不冲突。
    """
    requirement = requirement_override if requirement_override is not None else extract_requirement(ref["prompt"])
    is_patch = mode == "patch" and prev_generation is not None
    if is_patch:
        prompt = build_patch_prompt(ref, prev_generation, feedback_text or "（无具体反馈）", requirement)
    else:
        prompt = build_generate_prompt(ref, retrieved or [], requirement, understanding=understanding,
                                        extra_instruction=extra_instruction, diagnosis=diagnosis)
        if feedback_text:
            prompt += f"\n#注意：之前的实现有以下问题，这次请重新完整实现并避免它们：\n{feedback_text}\n"

    content, usage = call_llm(model_key, [{"role": "user", "content": prompt}], thinking=thinking, max_tokens=max_tokens)

    if is_patch:
        # patch 模式现在要的是查找/替换块，不是完整函数体——模型不需要把没改的
        # 部分再抄一遍，从机制上避免"追加而不是替换"导致内容重复的问题（实测
        # 过：旧的"输出完整函体"式 patch 会在某些轮次把新内容追加在旧内容
        # 后面而不是整体替换，导致同一段变量声明/逻辑出现两次）。
        blocks = parse_search_replace_blocks(content)
        if blocks:
            # 有块但应用失败（SEARCH 摘抄得不准/不唯一）时，不能退化去当"完整
            # 函数体"处理——这时 content 大概率就是裸的查找/替换块文本，没有
            # ```c``` 围栏，extract_c_code 找不到围栏会把整段原文（包括
            # <<<<<<< SEARCH 这些字面标记）当代码用，直接把 diff 标记写进
            # 编译文件（实测过，gcc 报"version control conflict marker"）。
            # 应用失败就老老实实保留上一轮的实现，不引入新的破坏。
            patched, applied = apply_search_replace_blocks(prev_generation, blocks)
            generation = patched if applied else prev_generation
        else:
            # 完全没有查找/替换块——可能是模型没听指令，直接给了完整函数体，
            # 这种情况才值得按老逻辑当完整函数体解析；解析结果里如果还带着
            # 查找/替换标记残留（说明只是格式没写完整，不是真的给了完整实现），
            # 同样不采用，保留上一轮的实现。
            fallback = extract_c_code(content)
            generation = (
                fallback if fallback.strip() and "<<<<<<<" not in fallback else prev_generation
            )
    else:
        generation = extract_c_code(content)

    if _DIFF_MARKER_RE.search(generation):
        # 最后一道防线：不管上面哪条分支产出的结果，只要字面还带着 diff 标记，
        # 就不采用。实测过一次具体案例——查找/替换块本身被正确解析、应用了，
        # 但模型在 REPLACE 的新代码里自己又写了一段"反思/重新考虑"的大段中文
        # 注释，里面嵌了一次没写完整（缺 >>>>>>> REPLACE 收尾）的查找/替换
        # 示意，这段没写完的嵌套标记就跟着 REPLACE 内容一起被换进去了。不去猜
        # 具体是哪个分支漏的，统一在这里兜底：patch 模式退回上一轮实现，
        # 没有上一轮可退（mode="new"）就直接把这几行标记摘掉。
        if is_patch:
            generation = prev_generation
        else:
            generation = _DIFF_MARKER_RE.sub("", generation)

    generation = strip_duplicate_signature(generation, ref)
    generation = fix_pointer_dot_notation(generation, ref)
    generation = symbol_oracle.fix_stdint_types(generation)
    generation = symbol_oracle.fix_libc_mem_functions(generation)
    generation = fix_identifier_case(generation, ref)
    generation = fix_closing_brace(generation, ref)
    return generation, usage, prompt


def strip_duplicate_signature(generation, ref):
    """确定性兜底：去掉生成内容开头重复的函数签名。

    prompt 里明确要求"只输出函数体的内容（从函数签名后面的第一行开始
    写...）"，但模型有时候会忽略这条指令，在生成开头又完整写一遍函数
    签名——跟 ref["declaration"] 重复。ref["prompt"] 已经以这个签名+
    左花括号结尾，生成开头再来一遍会变成"签名接签名"，肯定编译不过。

    用 ref["declaration"] 里的函数名去匹配生成开头是不是也有一个同名的
    函数定义（返回类型 函数名(参数...) {），有就把这部分（连它自己的
    左花括号）整个切掉，只保留真正的函数体。不要求跟 ref["declaration"]
    逐字符一致——模型重复签名时常见空格/tab 用法不同、参数间距不同，
    这里用 \\s* 兜住这类无关格式差异，只认函数名和大致的"签名(...)  {"
    结构。
    """
    declaration = ref.get("declaration") or ""
    name_match = re.search(r"(\w+)\s*\(", declaration)
    if not name_match:
        return generation
    func_name = name_match.group(1)
    dup_re = re.compile(
        rf"^\s*[\w\s\*]+?\b{re.escape(func_name)}\s*\([^)]*\)\s*\{{",
        re.DOTALL,
    )
    m = dup_re.match(generation)
    if m:
        return generation[m.end():]
    return generation


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_MEMBER_ACCESS_RE = re.compile(r"(->|\.)([A-Za-z_]\w*)")


def _build_case_map(context_text):
    variants = {}
    for name in _IDENTIFIER_RE.findall(context_text):
        variants.setdefault(name.lower(), set()).add(name)
    # 只有这个小写形式在上下文里只出现过一种大小写时才敢改——出现多种大小写
    # 说明本来就是两个不同标识符（比如同名但大小写不同的宏和变量），不能瞎猜。
    return {lower: next(iter(names)) for lower, names in variants.items() if len(names) == 1}


def fix_identifier_case(generation, ref):
    """确定性兜底：结构体成员访问的大小写纠正为 ref['prompt'] 里声明的真实大小写。

    实测过（llm_decompose 的分步实验里发现）：模型有时会把需求表格里数学
    符号命名（比如 S_{MODE}）自己"翻译"成看起来更像 C 变量习惯的小写形式
    （smode），但结构体字段实际声明是 Smode——这种大小写不一致在 C 里是
    货真价实的编译错误（未声明标识符），跟指针 ./-> 的坑同一类"模型自己
    二次翻译引入的语法错误"，同样能确定性修：只处理 a.b / a->b 里的成员名，
    在 ref['prompt']（结构体/变量声明原文）里找到大小写无歧义的正确写法就改，
    找不到或有歧义就不动。
    """
    case_map = _build_case_map(ref.get("prompt") or "")

    def _replace(m):
        sep, name = m.group(1), m.group(2)
        canonical = case_map.get(name.lower())
        if canonical and canonical != name:
            return sep + canonical
        return m.group(0)

    return _MEMBER_ACCESS_RE.sub(_replace, generation)


_PARAM_LIST_RE = re.compile(r"\(([^)]*)\)")


def _parse_params(declaration):
    """从函数签名里解析出每个参数的 (名字, 是不是指针)。"""
    m = _PARAM_LIST_RE.search(declaration)
    if not m:
        return []
    params = []
    for part in m.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        is_pointer = "*" in part
        name_match = re.search(r"(\w+)\s*(?:\[[^\]]*\])?$", part)
        if name_match:
            params.append((name_match.group(1), is_pointer))
    return params


def fix_pointer_dot_notation(generation, ref):
    """确定性兜底：把指针参数访问语法改对——两个方向都管。

    这不是猜的——实测过两个方向的案例都真实出现过：
    1. llm_understand 的"符号映射表"会把指针参数（比如函数签名里的
       TYPE *output）写成 output.field 而不是 output->field，这段错误的
       写法还会被"照理解翻译，不要自己重新翻译"这条指令原样抄进最终代码。
    2. 反过来，模型也会把**按值传递**（没有 *，比如 TYPE input）的参数
       写成 input->field（实测到：参数按值传递，模型却用了 ->，gcc 报
       "invalid type argument of '->'"）。
    这里从 ref["declaration"] 里确定性地解析出每个参数是不是指针，按各自
    该用的访问语法统一改，不依赖 LLM 这次有没有听话——双保险，prompt 里
    的提醒是第一道，这里是第二道。
    """
    declaration = ref.get("declaration") or ""
    for name, is_pointer in _parse_params(declaration):
        if is_pointer:
            generation = re.sub(rf"\b{re.escape(name)}\.", f"{name}->", generation)
        else:
            generation = re.sub(rf"\b{re.escape(name)}->", f"{name}.", generation)
    return generation


def fix_closing_brace(generation, ref):
    """确定性兜底：扫描整个生成内容，剥掉任何位置上"没有对应左括号可关"的
    多余右花括号，缺函数自己的收尾 } 时补上一个。

    prompt 里统一要求"不要写函数最后的右花括号"，这条指令只对**大多数**
    题目成立——那些题目的 ref["test"] 自己以 } 开头，build_source() 拼接
    时会补上这个 }。但少数题目的 ref["test"] 根本不以 } 开头（test 直接
    从一个辅助函数定义开始），这些题目的收尾 } 其实必须由生成内容自己
    提供——对这些题目，"不要写收尾括号"这条通用指令反而是错的。

    做法分两步：
    1. 扫描整段文本按花括号深度计数，找出所有"深度已经是 0、却又冒出一个
       }"的位置——这些都是没有对应 { 可关的孤儿括号，不管出现在生成末尾
       还是中间。
    2. 这些孤儿括号里，只有**最后一个、且后面只剩空白**的那一个，才可能是
       "ref["test"] 不提供收尾 } 时，生成自己该有的合法收尾括号"，予以
       保留（仅当 ref["test"] 确实不提供收尾 } 时）；其余孤儿括号（不管
       在中间，还是虽然在末尾但 ref["test"] 已经提供收尾 }）一律当多余的
       丢弃。孤儿括号数量为 0 且 ref["test"] 不提供收尾 } 时，说明生成
       内容缺一个收尾 }，补上一个。

    **踩过的坑**：
    1. 最初只检查生成末尾，且没检查 ref["test"] 是不是真的以 } 开头——
       在 ref["test"] 不以 } 开头的题目上，把本来必须保留的收尾 } 误剥了
       （因此从能编译变成编译失败）。
    2. 只处理末尾治不了"多余的 } 出现在生成中间"这种情况——实测过
       gcc-patch 的查找/替换机制在某一轮替换时，新写的代码块自己带了个
       不需要的收尾 }，把函数体在中间提前关闭，导致后面的代码全部跑到
       函数外面，gcc 报 "undeclared here (not in a function)"。
    3. 一开始想用"深度下限设成 -1"来统一处理"生成自己需要提供收尾 }"这种
       情况，结果发现深度计数本身分不清"这个 } 是函数末尾的合法收尾"还是
       "中间冒出来的多余括号"——两种情况深度计数结果一样。改成显式判断
       "是不是最后一个孤儿括号、且后面只剩空白"，才能把两种情况分开。
    """
    test_has_own_close = (ref.get("test") or "").lstrip().startswith("}")

    orphan_positions = []
    depth = 0
    for i, ch in enumerate(generation):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
            else:
                orphan_positions.append(i)

    last_orphan_is_tail = bool(orphan_positions) and not generation[orphan_positions[-1] + 1:].strip()
    keep_last_orphan_as_final_close = (not test_has_own_close) and last_orphan_is_tail

    drop_positions = set(orphan_positions[:-1] if keep_last_orphan_as_final_close else orphan_positions)
    result = "".join(ch for i, ch in enumerate(generation) if i not in drop_positions)

    if not test_has_own_close and not keep_last_orphan_as_final_close:
        result = result.rstrip() + "\n}"
    return result




_REDEFINITION_RE = re.compile(r"error: redefinition of '(\w+)'\s*\n\s*\d+\s*\|(.*)\n")
_COMMENT_RE = re.compile(r"//.*|/\*.*?\*/", re.DOTALL)


def _normalize_decl_line(line):
    return _COMMENT_RE.sub("", line).strip()


def strip_duplicate_declarations(generation, compile_message):
    """确定性兜底：gcc 报"redefinition of 'X'"、且两处声明去掉注释后逐字
    相同时，直接删掉多出来的那一份，不用为这种情况专门跑一轮 LLM patch。

    这是"精简输入信息再修复"的思路——跟符号裁判（往输入里加真实候选清单）
    刚好相反方向：符号裁判是给模型补充信息，这里是在喂给 LLM 之前先把
    "本来就没有歧义、纯粹是复读导致的多余内容"直接删掉，省掉一轮不必要
    的 LLM 调用，也避免 LLM 在这种没有实际二选一的地方节外生枝乱改。

    只处理"两处声明逐字相同"这种唯一安全的情况（实测过：patch 轮次里模型
    把一整段注释+声明原样复读了一遍，两处声明一字不差）——如果两处声明
    内容真不一样（比如同一个变量两轮猜了两个
    不同的表达式，这是真的存在歧义、需要判断哪个对），不属于这里能安全
    处理的范围，原样交给后面的 LLM patch 判断，不冒险瞎删。

    gcc 的报错本身就带着两处的源码原文（"NNN | 代码"这一行），不需要
    自己再算行号偏移去 generation 里定位——直接拿这段原文去 generation
    里找"逐字相同、且恰好出现两次"的那一行，删掉靠后的一份。要求恰好
    两次匹配才动手，出现次数对不上（可能是巧合撞到了别的地方）就跳过。
    """
    lines = generation.split("\n")
    to_remove = set()
    for m in _REDEFINITION_RE.finditer(compile_message):
        name, new_text = m.group(1), m.group(2)
        prev_re = re.compile(
            r"note: previous (?:definition|declaration) of '" + re.escape(name) + r"'[^\n]*\n\s*\d+\s*\|(.*)\n"
        )
        prev_match = prev_re.search(compile_message, m.end())
        if not prev_match:
            continue
        target_norm = _normalize_decl_line(new_text)
        if not target_norm or target_norm != _normalize_decl_line(prev_match.group(1)):
            continue
        matches = [i for i, line in enumerate(lines) if _normalize_decl_line(line) == target_norm]
        if len(matches) == 2:
            to_remove.add(matches[1])

    if not to_remove:
        return generation, False
    return "\n".join(line for i, line in enumerate(lines) if i not in to_remove), True
