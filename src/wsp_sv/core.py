import json
import random
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REV = "2fc0636471"
EDITED_ID = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
EDITED_REV = "d1671a1787"

ART = {"utr": "en", "neu": "ett"}
OTHER = {"utr": "neu", "neu": "utr"}

DEV_TEMPLATES = [
    ("Det här är {a}", " {n}."),
    ("Vi behöver {a} {adj}", " {n} till huset."),
    ("Hon pratade länge om {a}", " {n} med sina vänner."),
]
TEST_TEMPLATES = [
    ("Jag läste om {a}", " {n} i tidningen igår."),
    ("Där borta fanns {a} {old}", " {n} som ingen använde."),
    ("Han berättade om {a}", " {n} som ingen kände till."),
]
ADJ = {"adj": {"utr": "ny", "neu": "nytt"}, "old": {"utr": "gammal", "neu": "gammalt"}}

FEWSHOT = "Exempel på korrekt svenska: ett hus, en bil, ett bord, en stol, ett äpple, en lampa, ett barn, en hund.\n"
FEWSHOT_NOUNS = {"hus", "bil", "bord", "stol", "äpple", "lampa", "barn", "hund"}


@dataclass(frozen=True)
class Item:
    noun: str
    gender: str
    template: int
    prefix_ok: str
    prefix_bad: str
    cont: str
    n_tokens: int


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_id=MODEL_ID, revision=MODEL_REV, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype).to(device()).eval()
    return tok, model


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def load_lexicon(path):
    return json.load(open(path))["rows"]


def split_nouns(rows, tok, seed=0, dev_frac=0.4):
    rng = random.Random(seed)
    out = {"dev": [], "test": []}
    for g in ("utr", "neu"):
        nouns = sorted(r["noun"] for r in rows if r["gender"] == g and r["noun"] not in FEWSHOT_NOUNS)
        rng.shuffle(nouns)
        k = int(len(nouns) * dev_frac)
        for n in nouns[:k]:
            out["dev"].append((n, g, len(tok.encode(" " + n))))
        for n in nouns[k:]:
            out["test"].append((n, g, len(tok.encode(" " + n))))
    return out


def balance(nouns, seed=0, per_gender=None):
    rng = random.Random(seed)
    by = {g: [x for x in nouns if x[1] == g] for g in ("utr", "neu")}
    k = min(len(by["utr"]), len(by["neu"]))
    if per_gender:
        k = min(k, per_gender)
    picked = []
    for g in ("utr", "neu"):
        xs = list(by[g])
        rng.shuffle(xs)
        picked += xs[:k]
    return picked


def make_items(nouns, templates):
    items = []
    for n, g, nt in nouns:
        for ti, (pre, cont) in enumerate(templates):
            def fill(gg):
                return pre.format(a=ART[gg], adj=ADJ["adj"][gg], old=ADJ["old"][gg])
            items.append(Item(n, g, ti, fill(g), fill(OTHER[g]), cont.format(n=n), nt))
    return items


@torch.no_grad()
def score(model, tok, pairs, max_tokens=1024, bos=True):
    """Sum log-prob of each continuation given its prefix. pairs: list of (prefix, continuation).
    Batches are packed by padded token count to bound the size of the full-vocabulary logits."""
    dev = device()
    bos_ids = [tok.convert_tokens_to_ids("<|endoftext|>")] if bos else []
    seqs, spans = [], []
    for pre, cont in pairs:
        a = bos_ids + tok.encode(pre)
        b = tok.encode(cont)
        seqs.append(a + b)
        spans.append((len(a), len(a) + len(b)))
    out = [0.0] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    batches, cur = [], []
    for i in order:
        if cur and len(seqs[i]) * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    for idx in batches:
        L = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, : len(seqs[i])] = torch.tensor(seqs[i])
            att[r, : len(seqs[i])] = 1
        ids, att = ids.to(dev), att.to(dev)
        logits = model(input_ids=ids, attention_mask=att).logits
        for r, i in enumerate(idx):
            a, b = spans[i]
            lp = logits[r, a - 1 : b - 1].float().log_softmax(-1)
            out[i] = lp.gather(-1, ids[r, a:b, None]).sum().item()
        del logits
    return out


def evaluate_items(model, tok, items, prefix="", sentence=True):
    """Per item: log P(noun | prefix with 'ett') and log P(noun | prefix with 'en'),
    and optionally the full-sentence log-prob of the correct and the wrong sentence."""
    w = 4 if sentence else 2
    pairs = []
    for it in items:
        pairs += [(prefix + it.prefix_ok, " " + it.noun), (prefix + it.prefix_bad, " " + it.noun)]
        if sentence:
            pairs += [(prefix, it.prefix_ok + it.cont), (prefix, it.prefix_bad + it.cont)]
    s = score(model, tok, pairs)
    res = []
    for k, it in enumerate(items):
        ok, bad = s[w * k], s[w * k + 1]
        lp_ett, lp_en = (ok, bad) if it.gender == "neu" else (bad, ok)
        r = {"noun": it.noun, "gender": it.gender, "template": it.template, "n_tokens": it.n_tokens,
             "lp_ett": lp_ett, "lp_en": lp_en}
        if sentence:
            r["sent_margin"] = s[w * k + 2] - s[w * k + 3]
        res.append(r)
    return res


def _auc(res):
    from sklearn.metrics import roc_auc_score

    y = [r["gender"] == "neu" for r in res]
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, [r["lp_ett"] - r["lp_en"] for r in res]))


def _mismatch(res):
    import math

    def mass(rs, key):
        if not rs:
            return 0.0
        m = max(r[key] for r in rs)
        return math.exp(m) * sum(math.exp(r[key] - m) for r in rs)

    utr = [r for r in res if r["gender"] == "utr"]
    neu = [r for r in res if r["gender"] == "neu"]
    k = len(utr) / max(1, len(neu))
    out = []
    for key, wrong, right in (("lp_ett", utr, neu), ("lp_en", neu, utr)):
        mw = mass(wrong, key) * (1 / k if wrong is utr else k)
        mr = mass(right, key) * (k if right is neu else 1 / k)
        out.append(mw / (mw + mr) if mw + mr > 0 else float("nan"))
    return out


def summarize(res):
    """Gender AUC and mismatch mass are computed per template, then averaged."""
    tmpl = sorted({r["template"] for r in res})

    def per_t(f, rs):
        vals = [f([r for r in rs if r["template"] == t]) for t in tmpl]
        return sum(vals) / len(vals)

    one = [r for r in res if r["n_tokens"] == 1]
    multi = [r for r in res if r["n_tokens"] > 1]
    mm = [_mismatch([r for r in res if r["template"] == t]) for t in tmpl]
    out = {
        "gender_auc": per_t(_auc, res),
        "gender_auc_single_token": per_t(_auc, one),
        "gender_auc_multi_token": per_t(_auc, multi),
        "mismatch_mass_after_ett": sum(m[0] for m in mm) / len(mm),
        "mismatch_mass_after_en": sum(m[1] for m in mm) / len(mm),
        "n": len(res), "n_single": len(one), "n_multi": len(multi),
    }
    if res and "sent_margin" in res[0]:
        acc = {g: sum(r["sent_margin"] > 0 for r in res if r["gender"] == g) / max(1, sum(r["gender"] == g for r in res))
               for g in ("utr", "neu")}
        out.update({"sentence_balanced_acc": (acc["utr"] + acc["neu"]) / 2,
                    "sentence_acc_utr": acc["utr"], "sentence_acc_neu": acc["neu"]})
    return out


def blimp_pairs(name, n, seed=0):
    from datasets import load_dataset

    d = load_dataset("liu-nlp/swedish-blimp-" + name, split="train")
    idx = list(range(len(d)))
    random.Random(seed).shuffle(idx)
    return [(d[i]["original"], d[i]["corrupted"]) for i in idx[:n] if d[i]["original"] != d[i]["corrupted"]]


def blimp_acc(model, tok, pairs):
    flat = []
    for good, bad in pairs:
        flat += [("", good), ("", bad)]
    s = score(model, tok, flat)
    return sum(s[2 * i] > s[2 * i + 1] for i in range(len(pairs))) / len(pairs)


@torch.no_grad()
def perplexity(model, tok, texts, max_len=256, batch_size=16):
    pairs = []
    for t in texts:
        ids = tok.encode(t)[:max_len]
        pairs.append(("", tok.decode(ids)))
    lps = score(model, tok, pairs)
    ntok = sum(len(tok.encode(c)) for _, c in pairs)
    return float(torch.tensor(-sum(lps) / ntok).exp())
