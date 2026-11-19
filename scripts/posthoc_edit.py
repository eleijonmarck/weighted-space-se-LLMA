import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generation_check as gc
import run_experiment as rx
from generation_stats import distinct4

from wsp_sv import core, edits

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
NO_ADJ = [core.DEV_TEMPLATES[0], core.DEV_TEMPLATES[2]]
LAYERS = list(range(16, 24))
KS = [1.0, 2.0, 4.0]


def main():
    """Post-hoc, exploratory arms, added after the main run.
    Default: same method, but the direction comes only from dev templates without an adjective, with layer
    and strength chosen again on dev. Added after the 'nyckel' artifact was seen in generations.
    With --layer and --k: a fixed edit from all dev templates, e.g. the best dev point inside the perplexity budget."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int)
    ap.add_argument("--k", type=float)
    ap.add_argument("--name", default="posthoc_no_adj")
    args = ap.parse_args()
    core.seed_all(rx.SEED)
    tok, model = core.load()
    rows = core.load_lexicon(ROOT / "data" / "lexicon.json")
    split = core.split_nouns(rows, tok, seed=rx.SEED)
    dev_dir = core.balance(split["dev"], seed=rx.SEED + 1, per_gender=rx.N_DIR_PER_GENDER)
    sel_items = core.make_items(core.balance(split["dev"], seed=rx.SEED, per_gender=rx.N_SELECT_PER_GENDER), core.DEV_TEMPLATES)
    test_items = core.make_items(core.balance(split["test"], seed=rx.SEED, per_gender=rx.N_TEST_PER_GENDER), core.TEST_TEMPLATES)
    fixed = args.layer is not None
    templates = core.DEV_TEMPLATES if fixed else NO_ADJ
    stats = edits.paired_activations(model, tok, core.make_items(dev_dir, templates), [args.layer] if fixed else LAYERS)
    grid = []
    for L in [] if fixed else LAYERS:
        for k in KS:
            with edits.applied(model, [edits.gender_edit(stats, L, k)]):
                s = core.summarize(core.evaluate_items(model, tok, sel_items, sentence=False))
            grid.append({"layer": L, "k": k, "gender_auc": s["gender_auc"]})
            print("grid", L, k, round(s["gender_auc"], 4), flush=True)
    best = {"layer": args.layer, "k": args.k} if fixed else max(grid, key=lambda g: g["gender_auc"])
    e = edits.gender_edit(stats, best["layer"], best["k"], name=args.name)
    blimp = {n: core.blimp_pairs(n, rx.N_BLIMP, seed=rx.SEED) for n in rx.BLIMP_TARGET + rx.BLIMP_OTHER}
    sv, en = rx.ppl_texts()
    lex = {r["noun"]: r["gender"] for r in rows}
    with edits.applied(model, [e]):
        target = core.summarize(core.evaluate_items(model, tok, test_items))
        reg = rx.regressions(model, tok, sv, en, blimp)
        texts = gc.generate(model, tok, gc.PROMPTS)
    checked, errs = 0, []
    for t in texts:
        c, er = gc.mismatches(t, lex)
        checked += c
        errs += er
    words = [w for t in texts for w in t.lower().split()]
    gen = {"np_checked": checked, "mismatches": len(errs), "mismatch_rate": len(errs) / max(1, checked), "errors": errs,
           "nyckel_per_1k": 1000 * sum(w.startswith("nyckel") for w in words) / max(1, len(words)),
           "distinct_4gram": distinct4(texts), "samples": texts}
    out = {"grid": grid, "best": best, "edit": edits.edit_stats(model, e), "target": target, "regression": reg, "generation": gen}
    (RES / f"{args.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("grid",)} | {"generation": {k: v for k, v in gen.items() if k != "samples"}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
