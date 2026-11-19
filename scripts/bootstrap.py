import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_BOOT = 1000


def table(name):
    rows = json.loads((RES / f"items_{name}.json").read_text())
    nouns = sorted({r["noun"] for r in rows})
    idx = {n: i for i, n in enumerate(nouns)}
    T = sorted({r["template"] for r in rows})
    m = np.zeros((len(nouns), len(T)))
    y = np.zeros(len(nouns), dtype=bool)
    for r in rows:
        m[idx[r["noun"]], r["template"]] = r["lp_ett"] - r["lp_en"]
        y[idx[r["noun"]]] = r["gender"] == "neu"
    return nouns, m, y


def auc(m, y, sel):
    return float(np.mean([roc_auc_score(y[sel], m[sel, t]) for t in range(m.shape[1])]))


def main():
    arms = json.loads((RES / "arms.json").read_text())
    nouns, base, y = table("base")
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(nouns), len(nouns)) for _ in range(N_BOOT)]
    base_b = np.array([auc(base, y, b) for b in boots])
    out = {}
    for name in arms:
        n2, m, y2 = table(name)
        assert n2 == nouns and (y2 == y).all()
        d = np.array([auc(m, y, b) for b in boots]) - base_b
        out[name] = {"auc": auc(m, y, np.arange(len(nouns))), "delta_vs_base": float(d.mean()),
                     "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        print(f"{name:18s} auc={out[name]['auc']:.3f} delta={out[name]['delta_vs_base']:+.3f} ci95=[{out[name]['ci95'][0]:+.3f}, {out[name]['ci95'][1]:+.3f}]")
    (RES / "bootstrap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
