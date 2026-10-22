#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验记录器：把每一轮迭代的 token 消耗、用时、反馈信息、输入输出，
以及每道题的总 token 消耗都落盘到实验目录下的 record/ 里。

参考 ICSE-2027-revision 里 TokenAnalyzer 的字段设计，精简成一个更小的类。
"""
import json
import os
import re
import time
from datetime import datetime


def safe_filename(task_id):
    """task_id 里常有中文/斜杠/括号，落盘前统一替换成下划线。"""
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", task_id).strip("_")


class ExperimentRecorder:
    """out_dir 下应已经建好 generate/ output/ record/ result/ 四个子目录。"""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.generate_dir = os.path.join(out_dir, "generate")
        self.output_dir = os.path.join(out_dir, "output")
        self.record_dir = os.path.join(out_dir, "record")
        self.result_dir = os.path.join(out_dir, "result")
        for d in (self.generate_dir, self.output_dir, self.record_dir, self.result_dir):
            os.makedirs(d, exist_ok=True)
        self._log_path = os.path.join(self.output_dir, "run.log")

    def log(self, message):
        """人类可读的进度日志，同时打印到 stderr 和写入 output/run.log。"""
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_round(self, task_id, round_idx, stage, usage, elapsed_time,
                  feedback_text, input_text, output_text):
        """追加写一轮迭代的完整记录：轮次号、阶段、token 消耗、用时、反馈、输入输出。"""
        record = {
            "task_id": task_id,
            "round": round_idx,
            "stage": stage,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_tokens": (usage or {}).get("prompt_tokens", 0),
            "completion_tokens": (usage or {}).get("completion_tokens", 0),
            "total_tokens": (usage or {}).get("total_tokens", 0),
            "elapsed_time": elapsed_time,
            "feedback": feedback_text,
            "input_prompt": input_text,
            "output_content": output_text,
            # 思考模式打开时，模型的推理过程在这里；关闭时恒为空字符串。
            "reasoning": (usage or {}).get("reasoning", ""),
        }
        path = os.path.join(self.record_dir, f"{safe_filename(task_id)}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def log_task_summary(self, task_id, num_rounds, total_tokens, total_time, outcome):
        """每道题跑完后写一行汇总：总迭代轮数、总 token 消耗、总用时、最终结果。"""
        summary = {
            "task_id": task_id,
            "num_rounds": num_rounds,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "outcome": outcome,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = os.path.join(self.record_dir, "_summary.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return summary

    def save_generation(self, task_id, generation):
        path = os.path.join(self.generate_dir, f"{safe_filename(task_id)}.c")
        with open(path, "w", encoding="utf-8") as f:
            f.write(generation)
        return path

    def has_generation(self, task_id):
        """断点续跑用：这道题是否已经跑完（存在最终生成结果）。"""
        return os.path.exists(os.path.join(self.generate_dir, f"{safe_filename(task_id)}.c"))

    def load_generation(self, task_id):
        path = os.path.join(self.generate_dir, f"{safe_filename(task_id)}.c")
        with open(path, encoding="utf-8") as f:
            return f.read()


class TaskTimer:
    """给单道题计总时长用的小工具：with TaskTimer() as t: ... ; t.elapsed"""

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.monotonic() - self._start
