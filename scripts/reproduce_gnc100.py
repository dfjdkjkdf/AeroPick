"""Recompute the gnc100 safe-pass@1 numbers of the paper from the cached results (no LLM calls, no Docker).

execution judge: divergent pools use the reasoning execution judge, other pools the scenario grouping;
code judge:      every pool uses the reasoning code judge;
AeroPick:        divergent pools use the execution judge, other pools the code judge.
Each value is the expectation of a uniform pick inside the chosen group; reported as mean ± sd over 3 runs."""
import json, os, glob, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE = {"deepseek-v4-flash": "_judge_pilot187_think_gloss.json", "qwen3.6-35b-a3b": "_judge_pilot221_think_gloss.json"}

for model, jf in JUDGE.items():
    d = os.path.join(ROOT, "logs/cache", model, "gnc100/_bprobe/")
    pools = {(r["t"], r["p"]): r for r in json.load(open(d + "_result_llmtests_pools100.json"))}
    ex = {(j["t"], j["p"]): j for j in json.load(open(d + jf))}
    cj = {}
    for f in glob.glob(d + "_codejudge_result*.json"):
        cj.update({(r["t"], r["p"]): r for r in json.load(open(f))})
    methods = {
        "execution judge": lambda k: ex[k]["final"] if k in ex else pools[k]["hier"],
        "code judge": lambda k: cj[k]["final"],
        "AeroPick": lambda k: ex[k]["final"] if k in ex else cj[k]["final"],
        "best of 3 (upper bound)": lambda k: float(any(pools[k]["S"])),
    }
    print(model)
    for name, fn in methods.items():
        runs = [100 * st.mean(fn(k) for k in pools if k[1] == p) for p in (1, 2, 3)]
        print(f"  {name:24s} {st.mean(runs):5.1f} ± {st.stdev(runs):.1f}")
