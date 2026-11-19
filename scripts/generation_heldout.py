import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generation_check as gc

from wsp_sv import core

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Rates on the 160 prompts written after all edit choices were made: prompts 40 to 199 of generation_200."""
    lex = {r["noun"]: r["gender"] for r in core.load_lexicon(ROOT / "data" / "lexicon.json")}
    g = json.loads((ROOT / "results" / "generation_200.json").read_text())
    arms = [k for k in g if not k.startswith("_")]
    per = {}
    for k in arms:
        s = g[k]["samples"][40:]
        per[k] = (np.array([gc.mismatches(t, lex)[0] for t in s]), np.array([len(gc.mismatches(t, lex)[1]) for t in s]))
    nb, eb = per["base"]
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(nb), len(nb)) for _ in range(2000)]
    out = {}
    for k, (n, e) in per.items():
        d = [e[b].sum() / n[b].sum() - eb[b].sum() / nb[b].sum() for b in boots]
        out[k] = {"rate": float(e.sum() / n.sum()), "np": int(n.sum()), "errors": int(e.sum()),
                  "delta_vs_base": float(e.sum() / n.sum() - eb.sum() / nb.sum()),
                  "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        print(f"{k:24s} rate={out[k]['rate']:.3f} ({out[k]['errors']}/{out[k]['np']}) delta={out[k]['delta_vs_base']:+.3f} ci95=[{out[k]['ci95'][0]:+.3f}, {out[k]['ci95'][1]:+.3f}]")
    (ROOT / "results" / "generation_heldout160.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
