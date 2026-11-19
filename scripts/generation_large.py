import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generation_check as gc
import run_experiment as rx
from generation_prompt_control import generate
from generation_stats import distinct4

from wsp_sv import core, edits
from wsp_sv.lora import general_swedish, train_lora

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "generation_200.json"

FRAMES = [
    "Beskriv {t}.",
    "Berätta om {t} för en ny kollega.",
    "Vad är viktigt att veta om {t}?",
    "Skriv en kort text om {t}.",
    "Förklara för en tolvåring vad {t} innebär.",
    "Ge tre råd inför {t}.",
    "Skriv ett mejl till en vän om {t}.",
    "Jämför {t} förr och nu.",
]
TOPICS = [
    "ett besök på vårdcentralen", "en flytt till en ny stad", "ett föräldramöte i skolan", "en ansökan om bygglov",
    "ett nytt kollektivavtal", "en semester i fjällen", "ett kommunalt badhus", "en deklaration till Skatteverket",
    "ett bibliotek i en liten ort", "en tågresa till Malmö", "ett renoveringsprojekt i köket", "en hund i en lägenhet",
    "ett företag som säljer cyklar", "en lokal fotbollsklubb", "ett museum för barn", "en julmarknad",
    "ett val till riksdagen", "en ny busslinje", "ett sommarjobb", "en storm över västkusten",
]
PROMPTS = gc.PROMPTS + [f.format(t=t) for f in FRAMES for t in TOPICS]


def score(texts, lex):
    n = np.array([gc.mismatches(t, lex)[0] for t in texts])
    e = np.array([len(gc.mismatches(t, lex)[1]) for t in texts])
    words = [w for t in texts for w in t.lower().split()]
    return n, e, {"np_checked": int(n.sum()), "mismatches": int(e.sum()), "mismatch_rate": float(e.sum() / max(1, n.sum())),
                  "nyckel_per_1k": 1000 * sum(w.startswith("nyckel") for w in words) / max(1, len(words)),
                  "distinct_4gram": distinct4(texts)}


def main():
    """Free generation on 200 prompts: the 40 original prompts plus 8 frames x 20 topics.
    Arms: base, the pre-registered dev-selected edit, the post-hoc budget edit, the few-shot prefix, mixed LoRA."""
    t0 = time.time()
    core.seed_all(rx.SEED)
    tok, model = core.load()
    rows = core.load_lexicon(ROOT / "data" / "lexicon.json")
    lex = {r["noun"]: r["gender"] for r in rows}
    split = core.split_nouns(rows, tok, seed=rx.SEED)
    dev_items = core.make_items(core.balance(split["dev"], seed=rx.SEED + 1, per_gender=rx.N_DIR_PER_GENDER), core.DEV_TEMPLATES)
    sel = json.loads((RES / "selection.json").read_text())["best"]
    stats = edits.paired_activations(model, tok, dev_items, sorted({sel["layer"], 22}))
    arms = {
        "base": dict(edit=[], kw={}),
        "dev_selected": dict(edit=[edits.gender_edit(stats, sel["layer"], sel["k"])], kw={}),
        "posthoc_budget_L22_k05": dict(edit=[edits.gender_edit(stats, 22, 0.5)], kw={}),
        "fewshot_prefix": dict(edit=[], kw={"prefix": core.FEWSHOT}),
    }
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    per = {}
    for name, a in arms.items():
        if name not in out:
            with edits.applied(model, a["edit"]):
                texts = generate(model, tok, PROMPTS, **a["kw"])
            out[name] = {"samples": texts}
            OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        n, e, s = score(out[name]["samples"], lex)
        out[name].update(s)
        per[name] = (n, e)
        print(name, s, f"{time.time() - t0:.0f}s", flush=True)
    if "lora_mixed" not in out:
        m = train_lora(model, tok, dev_items, seed=rx.SEED, general=general_swedish(1200))
        out["lora_mixed"] = {"samples": generate(m, tok, PROMPTS)}
        del m
    n, e, s = score(out["lora_mixed"]["samples"], lex)
    out["lora_mixed"].update(s)
    per["lora_mixed"] = (n, e)
    print("lora_mixed", s, f"{time.time() - t0:.0f}s", flush=True)

    nb, eb = per["base"]
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(nb), len(nb)) for _ in range(2000)]
    for name, (n, e) in per.items():
        d = [e[b].sum() / max(1, n[b].sum()) - eb[b].sum() / max(1, nb[b].sum()) for b in boots]
        out[name]["delta_vs_base"] = out[name]["mismatch_rate"] - float(eb.sum() / nb.sum())
        out[name]["ci95_by_prompt"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        print(f"{name:24s} rate={out[name]['mismatch_rate']:.3f} delta={out[name]['delta_vs_base']:+.3f} ci95={out[name]['ci95_by_prompt']}")
    out["_meta"] = {"n_prompts": len(PROMPTS), "max_new_tokens": gc.MAX_NEW, "decoding": "greedy, thinking off"}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
