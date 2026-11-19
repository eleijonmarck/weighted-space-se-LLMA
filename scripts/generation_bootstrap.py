import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generation_check as gc

from wsp_sv import core

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def per_prompt(samples, lex):
    return np.array([gc.mismatches(t, lex)[0] for t in samples]), np.array([len(gc.mismatches(t, lex)[1]) for t in samples])


def main():
    """Mismatch-rate difference against base, with a bootstrap over prompts, since errors cluster within a reply."""
    lex = {r["noun"]: r["gender"] for r in core.load_lexicon(ROOT / "data" / "lexicon.json")}
    arms = {}
    g = json.loads((RES / "generation.json").read_text())
    for k in ("base", "dev_selected", "zero_feedback", "random_seed0"):
        arms[k] = g[k]["samples"]
    for f, k in (("posthoc_no_adj.json", "posthoc_no_adj"), ("posthoc_budget_L22_k05.json", "posthoc_budget_L22_k05")):
        arms[k] = json.loads((RES / f).read_text())["generation"]["samples"]
    pc = json.loads((RES / "generation_prompt_control.json").read_text())
    for k in pc:
        arms[k] = pc[k]["samples"]
    nb, eb = per_prompt(arms["base"], lex)
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(nb), len(nb)) for _ in range(2000)]
    out = {}
    for k, s in arms.items():
        n, e = per_prompt(s, lex)
        d = [e[b].sum() / max(1, n[b].sum()) - eb[b].sum() / max(1, nb[b].sum()) for b in boots]
        out[k] = {"rate": float(e.sum() / n.sum()), "delta_vs_base": float(e.sum() / n.sum() - eb.sum() / nb.sum()),
                  "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "np": int(n.sum()), "errors": int(e.sum())}
        print(f"{k:24s} rate={out[k]['rate']:.3f} ({out[k]['errors']}/{out[k]['np']}) delta={out[k]['delta_vs_base']:+.3f} ci95=[{out[k]['ci95'][0]:+.3f}, {out[k]['ci95'][1]:+.3f}]")
    (RES / "generation_bootstrap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
