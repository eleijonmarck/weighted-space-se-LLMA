import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx

from wsp_sv import core, edits

ROOT = Path(__file__).resolve().parents[1]
LAYERS = list(range(16, 24))
KS = [0.25, 0.5, 1.0, 2.0]


def main():
    """Dev-only trade-off: gender AUC gain on dev nouns and dev templates against Swedish perplexity cost."""
    core.seed_all(rx.SEED)
    tok, model = core.load()
    rows = core.load_lexicon(ROOT / "data" / "lexicon.json")
    split = core.split_nouns(rows, tok, seed=rx.SEED)
    sel_items = core.make_items(core.balance(split["dev"], seed=rx.SEED, per_gender=rx.N_SELECT_PER_GENDER), core.DEV_TEMPLATES)
    dev_items = core.make_items(core.balance(split["dev"], seed=rx.SEED + 1, per_gender=rx.N_DIR_PER_GENDER), core.DEV_TEMPLATES)
    sv, _ = rx.ppl_texts()
    stats = edits.paired_activations(model, tok, dev_items, LAYERS)
    base_auc = core.summarize(core.evaluate_items(model, tok, sel_items, sentence=False))["gender_auc"]
    base_ppl = core.perplexity(model, tok, sv)
    out = {"base_dev_auc": base_auc, "base_ppl_sv": base_ppl, "points": []}
    for L in LAYERS:
        for k in KS:
            e = edits.gender_edit(stats, L, k)
            with edits.applied(model, [e]):
                auc = core.summarize(core.evaluate_items(model, tok, sel_items, sentence=False))["gender_auc"]
                ppl = core.perplexity(model, tok, sv)
            out["points"].append({"layer": L, "k": k, "dev_auc_gain": auc - base_auc, "ppl_sv_rel": ppl / base_ppl - 1,
                                  "rel_frob": edits.edit_stats(model, e)["rel_frob"]})
            print(L, k, round(auc - base_auc, 4), round(ppl / base_ppl - 1, 4), flush=True)
            (ROOT / "results" / "frontier.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
