import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx

from transformers import AutoTokenizer

from wsp_sv import core

ROOT = Path(__file__).resolve().parents[1]
BASE_ID = "Qwen/Qwen3.5-0.8B-Base"


def main():
    """Same test items and regressions on the pretrained-only checkpoint. The subject model is post-trained,
    and all likelihoods are scored as raw text after <|endoftext|>, so this shows how much of a metric is format."""
    rev = sys.argv[1]
    core.seed_all(rx.SEED)
    tok, model = core.load(BASE_ID, rev)
    subj_tok = AutoTokenizer.from_pretrained(core.MODEL_ID, revision=core.MODEL_REV)
    rows = core.load_lexicon(ROOT / "data" / "lexicon.json")
    split = core.split_nouns(rows, subj_tok, seed=rx.SEED)
    test_items = core.make_items(core.balance(split["test"], seed=rx.SEED, per_gender=rx.N_TEST_PER_GENDER), core.TEST_TEMPLATES)
    blimp = {n: core.blimp_pairs(n, rx.N_BLIMP, seed=rx.SEED) for n in rx.BLIMP_TARGET + rx.BLIMP_OTHER}
    sv, en = rx.ppl_texts()
    out = {"model": BASE_ID, "revision": rev, "target": core.summarize(core.evaluate_items(model, tok, test_items)),
           "regression": rx.regressions(model, tok, sv, en, blimp)}
    (ROOT / "results" / "base_model_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
