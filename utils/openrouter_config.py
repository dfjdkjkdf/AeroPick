#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenRouter 的连接配置：密钥、可选模型、默认参数。

密钥从环境变量读取：OPENROUTER_API_KEY（Qwen 等 OpenRouter 模型）、DEEPSEEK_API_KEY（DeepSeek）。
不在仓库中保存任何密钥。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FALLBACK_KEY_PATH = os.path.join(HERE, "..", "openrouter.key")  # optional local file, git-ignored

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# DeepSeek 走独立 API（不经过 OpenRouter）；密钥用环境变量 DEEPSEEK_API_KEY，或可选的本地配置文件。
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
HERMES_CONFIG_PATH = os.path.expanduser("~/.hermes/config.yaml")


def load_deepseek_api_key():
    env_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_key:
        return env_key.strip()
    with open(HERMES_CONFIG_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("api_key:"):
                return line.split(":", 1)[1].strip()
    raise RuntimeError(f"没在 {HERMES_CONFIG_PATH} 里找到 api_key")


# 可选模型：model_name 是对应服务商上的真实 model id；provider 决定走哪个
# base_url/密钥（见 PROVIDERS）；thinking 是思考模式开关，只对 OpenRouter 这几个
# Qwen 模型有意义（对应请求体里的 reasoning.enabled）——DeepSeek 走自己的默认
# 行为，不用这个字段。
MODEL_CONFIGS = {
    "qwen3.5-9b": {"model_name": "qwen/qwen3.5-9b", "thinking": False, "provider": "openrouter"},
    "qwen3.5-35b-a3b": {"model_name": "qwen/qwen3.5-35b-a3b", "thinking": False, "provider": "openrouter"},
    "qwen3.6-35b-a3b": {"model_name": "qwen/qwen3.6-35b-a3b", "thinking": False, "provider": "openrouter"},
    "deepseek-v4-flash": {"model_name": "deepseek-v4-flash", "thinking": False, "provider": "deepseek"},
}
PROVIDERS = {
    "openrouter": {"base_url": BASE_URL, "load_api_key": lambda: load_api_key()},
    "deepseek": {"base_url": DEEPSEEK_BASE_URL, "load_api_key": lambda: load_deepseek_api_key()},
}
DEFAULT_MODEL = "qwen3.5-9b"

# 采样参数照 Qwen3.5 官方模型卡（huggingface.co/Qwen/Qwen3.5-9B）的 best practice 分模式给：
# 思考模式温度用"精确编码(WebDev)"档的 0.6（比"通用任务"档的 1.0 更适合写代码），
# 不思考模式用"通用任务"档的 0.7（非思考模式官方没有单独给编码档）；top_p/top_k 两档一致。
# presence_penalty 原来两种模式都没设，是复读循环复现时最容易被忽略的一个杠杆——
# 官方原话"设 0~2 之间的 presence_penalty 减少无休止重复"。实测过：同一道会复读的题，
# presence_penalty=1.0（自己拍的折中值）能压住一部分复读，但会把"逐字复读"变成
# "同义词轮换复读"，没根治；改用官方"通用任务"档的 1.5 之后两次实测都没有复读，
# 且明显比"精确编码"档的 0.0 更快更省 token，所以两种模式统一改成 1.5
# （不是"精确编码"档的 0.0——我们的复读问题出在推理文字上，更像通用推理场景，
# 用编码档的低惩罚反而压不住）。
SAMPLING_PARAMS = {
    True: {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},   # 思考模式
    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},   # 不思考
}
DEFAULT_TEMPERATURE = 0.7  # 向后兼容用；实际请求按 SAMPLING_PARAMS[thinking] 取值
# 关了思考模式后，这几个模型遇到规格有歧义/信息缺失的题目时，倾向于把推理过程
# 直接写成代码里的大段注释（没有独立的思考通道可用）。4096 经常不够把这种
# "边想边写"的过程走完，写到一半被截断——不光是没写完代码，截断点还可能停在
# markdown 代码块结尾的 ``` 之前，导致围栏没闭合。调大预算降低被硬截断的概率
# （不能根治"话多"这个行为本身，但能避免"话多"直接等价于"写不完代码"）。
DEFAULT_MAX_TOKENS = 16384
REQUEST_TIMEOUT = 180
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))  # 2026-10-07：上游限流时可用环境变量加大重试次数（默认不变）


def load_api_key():
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        return env_key.strip()
    with open(FALLBACK_KEY_PATH, encoding="utf-8") as f:
        return f.read().strip()


def resolve_model(model_key):
    """把 'qwen3.5-9b' 这种简称解析成完整的模型配置字典。"""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"未知模型 {model_key!r}，可选：{list(MODEL_CONFIGS)}")
    return MODEL_CONFIGS[model_key]
