import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

from wsp_sv import core, edits

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEED = 0
K_GRID = [1.0, 2.0, 4.0]
N_SELECT_PER_GENDER = 100
N_DIR_PER_GENDER = 200
N_TEST_PER_GENDER = 250
N_BLIMP = 400
N_PPL_DOCS = 150

BLIMP_TARGET = ["adjectives-mascfem_to_neut", "adjectives-neut_and_adverbs_to_masfem"]
BLIMP_OTHER = ["finite-to-nonfinite", "active-to-passive", "adjectives-sg-to-pl"]


def save(name, obj):
    RES.mkdir(exist_ok=True)
    (RES / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def ppl_texts():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("HuggingFaceFW/fineweb-2", "data/swe_Latn/test/000_00000.parquet", repo_type="dataset")
    sv_texts = pd.read_parquet(path, columns=["text"]).text.iloc[5000 : 5000 + N_PPL_DOCS].tolist()
    en = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    en_texts = [t for t in en["text"] if len(t) > 400][:N_PPL_DOCS]
    return sv_texts, en_texts


def regressions(model, tok, sv_texts, en_texts, blimp):
    out = {"ppl_sv": core.perplexity(model, tok, sv_texts), "ppl_en": core.perplexity(model, tok, en_texts)}
    for name, pairs in blimp.items():
        out["blimp_" + name] = core.blimp_acc(model, tok, pairs)
    return out


def probe(model, tok, split, layers):
    """Can a linear probe read noun gender? Weights only (first-token unembedding row) and
    residual stream at the noun's last token, train on dev nouns, test on test nouns."""
    W = model.get_output_embeddings().weight.detach().float().cpu()

    def feats_w(nouns):
        return np.stack([W[tok.encode(" " + n)[0]].numpy() for n, _, _ in nouns])

    bos = [tok.convert_tokens_to_ids("<|endoftext|>")]

    @torch.no_grad()
    def feats_h(nouns):
        out = {L: [] for L in layers}
        for n, _, _ in nouns:
            ids = torch.tensor([bos + tok.encode(" " + n)]).to(core.device())
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            for L in layers:
                out[L].append(hs[L + 1][0, -1].float().cpu().numpy())
        return {L: np.stack(v) for L, v in out.items()}

    tr, te = split["dev"], split["test"]
    ytr = np.array([g == "neu" for _, g, _ in tr])
    yte = np.array([g == "neu" for _, g, _ in te])
    multi = np.array([k > 1 for _, _, k in te])

    def fit(Xtr, Xte):
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(Xtr, ytr)
        p = clf.predict(Xte)

        def bal(mask):
            m = mask
            return float(((p[m & yte] == 1).mean() + (p[m & ~yte] == 0).mean()) / 2)

        return {"all": bal(np.ones_like(yte)), "single": bal(~multi), "multi": bal(multi)}

    res = {"weights_first_token": fit(feats_w(tr), feats_w(te))}
    rng = np.random.default_rng(SEED)
    ytr_shuf = rng.permutation(ytr)
    Htr, Hte = feats_h(tr), feats_h(te)
    for L in layers:
        res[f"resid_L{L}"] = fit(Htr[L], Hte[L])
    clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(Htr[layers[-1]], ytr_shuf)
    p = clf.predict(Hte[layers[-1]])
    res["control_shuffled_labels"] = float(((p[yte] == 1).mean() + (p[~yte] == 0).mean()) / 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default=str(ROOT / "data" / "lexicon.json"))
    ap.add_argument("--skip-lora", action="store_true")
    args = ap.parse_args()
    core.seed_all(SEED)
    t0 = time.time()
    tok, model = core.load()
    rows = core.load_lexicon(args.lexicon)
    split = core.split_nouns(rows, tok, seed=SEED)
    dev_sel = core.balance(split["dev"], seed=SEED, per_gender=N_SELECT_PER_GENDER)
    test = core.balance(split["test"], seed=SEED, per_gender=N_TEST_PER_GENDER)
    dev_dir = core.balance(split["dev"], seed=SEED + 1, per_gender=N_DIR_PER_GENDER)
    dev_items = core.make_items(dev_dir, core.DEV_TEMPLATES)
    sel_items = core.make_items(dev_sel, core.DEV_TEMPLATES)
    test_items = core.make_items(test, core.TEST_TEMPLATES)
    save("splits.json", {"dev": split["dev"], "test": split["test"], "dev_select": dev_sel, "test_balanced": test,
                         "dev_templates": core.DEV_TEMPLATES, "test_templates": core.TEST_TEMPLATES})
    print(f"nouns dev={len(split['dev'])} test={len(split['test'])} sel_items={len(sel_items)} test_items={len(test_items)}", flush=True)

    blimp = {n: core.blimp_pairs(n, N_BLIMP, seed=SEED) for n in BLIMP_TARGET + BLIMP_OTHER}
    sv_texts, en_texts = ppl_texts()

    layers = list(range(model.config.num_hidden_layers))
    probes = probe(model, tok, {"dev": dev_dir, "test": test}, layers)
    save("probe.json", probes)
    print("probe", {k: v for k, v in probes.items() if k in ("weights_first_token", "control_shuffled_labels", "resid_L11", "resid_L23")}, flush=True)

    arms = {}

    def run_arm(name, edit_list=(), prefix="", m=None):
        mm = m or model
        with edits.applied(mm, list(edit_list)):
            res = core.evaluate_items(mm, tok, test_items, prefix=prefix)
            reg = regressions(mm, tok, sv_texts, en_texts, blimp) if not prefix else {}
        arms[name] = {"target": core.summarize(res),
                      "regression": reg, "edits": [edits.edit_stats(mm, e) for e in edit_list]}
        save(f"items_{name}.json", res)
        save("arms.json", arms)
        print(name, json.dumps(arms[name]["target"]), json.dumps(reg), f"{time.time() - t0:.0f}s", flush=True)

    run_arm("base")

    stats = edits.paired_activations(model, tok, dev_items, layers)
    sep = {L: float(stats[L]["res_diff"].mean(0).norm() / stats[L]["res_std"]) for L in layers}

    L0 = max(sep, key=sep.get)
    zero_fb = edits.gender_edit(stats, L0, 1.0, name="zero_feedback")

    grid = []
    for L in layers:
        for k in K_GRID:
            e = edits.gender_edit(stats, L, k)
            with edits.applied(model, [e]):
                s = core.summarize(core.evaluate_items(model, tok, sel_items, sentence=False))
            grid.append({"layer": L, "k": k, **s})
            print("grid", L, k, round(s["gender_auc"], 4), flush=True)
    with edits.applied(model, []):
        base_sel = core.summarize(core.evaluate_items(model, tok, sel_items, sentence=False))
    best = max(grid, key=lambda g: g["gender_auc"])
    save("selection.json", {"separation_by_layer": sep, "zero_feedback_layer": L0, "grid": grid, "best": best,
                            "base_dev_select": base_sel, "n_dev_evals": len(grid)})
    sel = edits.gender_edit(stats, best["layer"], best["k"], name="dev_selected")

    w_dir = edits.unembed_gender_dir(model, tok, split["dev"])
    weights_write = edits.gender_edit(stats, best["layer"], best["k"], write_dir=w_dir, name="weights_write_dir")

    run_arm("zero_feedback", [zero_fb])
    run_arm("dev_selected", [sel])
    run_arm("weights_write_dir", [weights_write])
    for s in range(5):
        run_arm(f"random_seed{s}", [edits.random_like(sel, 1000 + s)])
    run_arm("fewshot_prompt", prefix=core.FEWSHOT)

    if not args.skip_lora:
        from wsp_sv.lora import train_lora

        lora_model = train_lora(model, tok, dev_items, seed=SEED)
        run_arm("lora_dev", m=lora_model)

    save("run_meta.json", {"model": core.MODEL_ID, "revision": core.MODEL_REV, "seed": SEED, "k_grid": K_GRID,
                           "n_select_per_gender": N_SELECT_PER_GENDER, "n_test_per_gender": N_TEST_PER_GENDER,
                           "n_blimp": N_BLIMP, "n_ppl_docs": N_PPL_DOCS, "torch": torch.__version__,
                           "device": core.device(), "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
