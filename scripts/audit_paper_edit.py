import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx

from wsp_sv import core

ROOT = Path(__file__).resolve().parents[1]


def tensors(repo, rev):
    out = {}
    for f in list_repo_files(repo, revision=rev):
        if f.endswith(".safetensors"):
            with safe_open(hf_hub_download(repo, f, revision=rev), "pt") as h:
                for k in h.keys():
                    out[k] = (f, k)
    return out


def get(repo, rev, ref):
    f, k = ref
    with safe_open(hf_hub_download(repo, f, revision=rev), "pt") as h:
        return h.get_tensor(k)


def main():
    base = tensors(core.MODEL_ID, core.MODEL_REV)
    edited = tensors(core.EDITED_ID, core.EDITED_REV)
    changed = {}
    for k in sorted(set(base) & set(edited)):
        a = get(core.MODEL_ID, core.MODEL_REV, base[k])
        b = get(core.EDITED_ID, core.EDITED_REV, edited[k])
        if a.shape != b.shape or not torch.equal(a, b):
            d = (b.float() - a.float())
            changed[k] = {"rel_frob": float(d.norm() / a.float().norm()), "rank": int(torch.linalg.matrix_rank(d)),
                          "cols_changed": int((d.abs().sum(0) > 0).sum()), "tensor": b}
    print("changed tensors:", {k: {kk: vv for kk, vv in v.items() if kk != "tensor"} for k, v in changed.items()})

    tok, model = core.load()
    rows = core.load_lexicon(ROOT / "data" / "lexicon.json")
    split = core.split_nouns(rows, tok, seed=rx.SEED)
    test = core.balance(split["test"], seed=rx.SEED, per_gender=rx.N_TEST_PER_GENDER)
    items = core.make_items(test, core.TEST_TEMPLATES)
    blimp = {n: core.blimp_pairs(n, rx.N_BLIMP, seed=rx.SEED) for n in rx.BLIMP_TARGET + rx.BLIMP_OTHER}
    sv, en = rx.ppl_texts()

    out = {"changed": {k: {kk: vv for kk, vv in v.items() if kk != "tensor"} for k, v in changed.items()}}
    out["base"] = {"target": core.summarize(core.evaluate_items(model, tok, items)),
                   "regression": rx.regressions(model, tok, sv, en, blimp)}
    params = dict(model.named_parameters())
    for k, v in changed.items():
        name = k.replace("model.language_model.", "model.")
        params[name].data.copy_(v["tensor"].to(params[name].dtype))
    out["paper_edit"] = {"target": core.summarize(core.evaluate_items(model, tok, items)),
                         "regression": rx.regressions(model, tok, sv, en, blimp)}
    (ROOT / "results" / "paper_edit_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
