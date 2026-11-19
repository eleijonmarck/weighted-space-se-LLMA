import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx

from wsp_sv import core
from wsp_sv.lora import general_swedish, train_lora

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_GENERAL = 1200


def main():
    """Fairer fine-tuning control: LoRA on the same dev sentences, mixed 1:1 with general Swedish text.
    Adds the arm 'lora_mixed' to results/arms.json so that the bootstrap and the figures pick it up."""
    core.seed_all(rx.SEED)
    tok, model = core.load()
    rows = core.load_lexicon(ROOT / "data" / "lexicon.json")
    split = core.split_nouns(rows, tok, seed=rx.SEED)
    dev_items = core.make_items(core.balance(split["dev"], seed=rx.SEED + 1, per_gender=rx.N_DIR_PER_GENDER), core.DEV_TEMPLATES)
    test_items = core.make_items(core.balance(split["test"], seed=rx.SEED, per_gender=rx.N_TEST_PER_GENDER), core.TEST_TEMPLATES)
    blimp = {n: core.blimp_pairs(n, rx.N_BLIMP, seed=rx.SEED) for n in rx.BLIMP_TARGET + rx.BLIMP_OTHER}
    sv, en = rx.ppl_texts()

    m = train_lora(model, tok, dev_items, seed=rx.SEED, general=general_swedish(N_GENERAL))
    res = core.evaluate_items(m, tok, test_items)
    arm = {"target": core.summarize(res), "regression": rx.regressions(m, tok, sv, en, blimp), "edits": [],
           "config": {"n_template_sentences": len(dev_items), "n_general_texts": N_GENERAL, "general_source":
                      "HuggingFaceFW/fineweb-2 swe_Latn train/000_00000.parquet rows 0-1199, cut to 128 tokens"}}
    (RES / "items_lora_mixed.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    arms = json.loads((RES / "arms.json").read_text())
    arms["lora_mixed"] = arm
    (RES / "arms.json").write_text(json.dumps(arms, ensure_ascii=False, indent=1))
    print("lora_mixed", json.dumps(arm["target"]), json.dumps(arm["regression"]), flush=True)


if __name__ == "__main__":
    main()
