import json
import re
import time
from pathlib import Path

import torch
from lingua import Language, LanguageDetectorBuilder

from wsp_sv import core, edits

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MAX_NEW = 200

PROMPTS = [
    "Beskriv ditt drömhus i fem meningar.",
    "Vad kan man hitta i ett vanligt svenskt kök?",
    "Berätta om en resa med tåg genom Norrland.",
    "Beskriv ett klassrum i en svensk skola.",
    "Vad behöver man för att starta ett företag i Sverige?",
    "Skriv en kort berättelse om ett barn som hittar ett gammalt brev.",
    "Beskriv en typisk midsommarafton.",
    "Vilka saker brukar finnas i ett badrum?",
    "Förklara hur ett kraftverk fungerar.",
    "Beskriv ett museum som du gärna skulle besöka.",
    "Skriv en annons för en begagnad cykel.",
    "Vad är skillnaden mellan ett lån och ett bidrag?",
    "Beskriv en promenad i en skog på hösten.",
    "Skriv ett kort brev till en vän om ett nytt jobb.",
    "Beskriv ett sjukhus ur en patients perspektiv.",
    "Vad gör man på ett bibliotek?",
    "Berätta om ett djur som lever i Sverige.",
    "Beskriv en lägenhet i Stockholm.",
    "Förklara vad ett avtal är och ge ett exempel.",
    "Skriv en kort saga om ett troll och en prinsessa.",
    "Beskriv ett fönster med utsikt över havet.",
    "Vilka problem kan uppstå i ett nytt bostadsområde?",
    "Beskriv ett kontor där många människor arbetar.",
    "Berätta om ett minne från en sommar på landet.",
    "Beskriv ett träd som du tycker om.",
    "Vad bör ett bra cv innehålla?",
    "Skriv en recension av ett café.",
    "Beskriv ett fotbollslag och dess spelare.",
    "Förklara hur ett flygplan kan flyga.",
    "Beskriv en marknad en lördagsmorgon.",
    "Skriv en instruktion för att byta ett däck.",
    "Beskriv ett hotellrum.",
    "Vad är ett ekosystem? Ge exempel.",
    "Skriv om ett möte mellan två gamla vänner på ett tåg.",
    "Beskriv ett kök i en restaurang under lunchrusningen.",
    "Berätta om ett yrke som du tycker är viktigt.",
    "Beskriv ett bröllop i en liten kyrka.",
    "Vad kan man göra på ett regnigt sportlov?",
    "Beskriv en båt och ett hav i storm.",
    "Skriv ett kort tal till en kollega som går i pension.",
]


def mismatches(text, lex):
    words = re.findall(r"[a-zåäöA-ZÅÄÖ]+", text)
    low = [w.lower() for w in words]
    checked, errs = 0, []
    for i, w in enumerate(low):
        if w not in ("en", "ett"):
            continue
        for j in (i + 1, i + 2):
            if j < len(low) and low[j] in lex:
                g = lex[low[j]]
                checked += 1
                if core.ART[g] != w:
                    errs.append(" ".join(words[i : j + 1]))
                break
    return checked, errs


@torch.no_grad()
def generate(model, tok, prompts):
    outs = []
    for p in prompts:
        s = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        ids = tok(s, return_tensors="pt").to(core.device())
        o = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False)
        outs.append(tok.decode(o[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip())
    return outs


def main():
    t0 = time.time()
    core.seed_all(0)
    tok, model = core.load()
    lex = {r["noun"]: r["gender"] for r in core.load_lexicon(ROOT / "data" / "lexicon.json")}
    sel = json.loads((RES / "selection.json").read_text())
    splits = json.loads((RES / "splits.json").read_text())
    dev_dir = core.balance([tuple(x) for x in splits["dev"]], seed=1, per_gender=200)
    dev_items = core.make_items(dev_dir, core.DEV_TEMPLATES)
    L, k = sel["best"]["layer"], sel["best"]["k"]
    stats = edits.paired_activations(model, tok, dev_items, [L, sel["zero_feedback_layer"]])
    arms = {
        "base": [],
        "dev_selected": [edits.gender_edit(stats, L, k)],
        "zero_feedback": [edits.gender_edit(stats, sel["zero_feedback_layer"], 1.0)],
        "random_seed0": [edits.random_like(edits.gender_edit(stats, L, k), 1000)],
    }
    det = LanguageDetectorBuilder.from_languages(Language.SWEDISH, Language.BOKMAL, Language.NYNORSK,
                                                 Language.DANISH, Language.ENGLISH).build()
    out = {}
    for name, e in arms.items():
        with edits.applied(model, e):
            texts = generate(model, tok, PROMPTS)
        checked, errs, sv = 0, [], 0
        for t in texts:
            c, er = mismatches(t, lex)
            checked += c
            errs += er
            sv += det.detect_language_of(t) == Language.SWEDISH
        out[name] = {"np_checked": checked, "mismatches": len(errs), "mismatch_rate": len(errs) / max(1, checked),
                     "swedish_share": sv / len(texts), "errors": errs, "samples": texts}
        print(name, checked, len(errs), round(len(errs) / max(1, checked), 4), sv, f"{time.time() - t0:.0f}s", flush=True)
        (RES / "generation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
