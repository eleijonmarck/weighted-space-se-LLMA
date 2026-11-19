import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generation_check as gc
from generation_stats import distinct4

from wsp_sv import core

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "Svara på korrekt svenska. Var noga med att använda rätt artikel, en eller ett, och rätt böjning av adjektiv."


@torch.no_grad()
def generate(model, tok, prompts, system=None, prefix=""):
    outs = []
    for p in prompts:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prefix + p}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(s, return_tensors="pt").to(core.device())
        o = model.generate(**ids, max_new_tokens=gc.MAX_NEW, do_sample=False)
        outs.append(tok.decode(o[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip())
    return outs


def main():
    """Prompt-only controls for the free-generation metric: a few-shot gender prefix and a system instruction."""
    core.seed_all(0)
    tok, model = core.load()
    lex = {r["noun"]: r["gender"] for r in core.load_lexicon(ROOT / "data" / "lexicon.json")}
    out = {}
    for name, kw in (("fewshot_prefix", {"prefix": core.FEWSHOT}), ("system_instruction", {"system": SYSTEM})):
        texts = generate(model, tok, gc.PROMPTS, **kw)
        checked, errs = 0, []
        for t in texts:
            c, er = gc.mismatches(t, lex)
            checked += c
            errs += er
        words = [w for t in texts for w in t.lower().split()]
        out[name] = {"np_checked": checked, "mismatches": len(errs), "mismatch_rate": len(errs) / max(1, checked),
                     "errors": errs, "nyckel_per_1k": 1000 * sum(w.startswith("nyckel") for w in words) / max(1, len(words)),
                     "distinct_4gram": distinct4(texts), "samples": texts}
        print(name, checked, len(errs), round(len(errs) / max(1, checked), 4), round(out[name]["distinct_4gram"], 3), flush=True)
        (ROOT / "results" / "generation_prompt_control.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
