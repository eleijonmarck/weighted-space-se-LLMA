import re, json, collections, sys
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "HuggingFaceFW/fineweb-2"
FILES = ["data/swe_Latn/test/000_00000.parquet", "data/swe_Latn/train/000_00000.parquet"]
REV = "main"
OUT = sys.argv[1]
BLOCK = set(open(sys.argv[2]).read().split())

word = re.compile(r"[a-zåäöéü]+")
art = collections.Counter()
form = collections.Counter()
for f in FILES:
    p = hf_hub_download(REPO, f, repo_type="dataset", revision=REV)
    df = pd.read_parquet(p, columns=["text"])
    for t in df.text:
        toks = word.findall(t.lower())
        form.update(toks)
        for a, b in zip(toks, toks[1:]):
            if a == "en" or a == "ett":
                art[(a, b)] += 1

def suffix_counts(w):
    if w[-1] in "aeiouyåäö":
        return form[w + "n"], form[w + "t"]
    return form[w + "en"], form[w + "et"]

rows = []
for w in {b for _, b in art}:
    n_en, n_ett = art[("en", w)], art[("ett", w)]
    if n_en + n_ett < 15 or len(w) < 3:
        continue
    s_utr, s_neu = suffix_counts(w)
    r_art = n_ett / (n_en + n_ett)
    if s_utr + s_neu < 5:
        continue
    r_suf = s_neu / (s_utr + s_neu)
    if w in BLOCK:
        continue
    if w[-1] not in "aeiouyåäö" and form[w + "t"] > 0.1 * form[w]:
        continue
    if r_art <= 0.05 and r_suf <= 0.1:
        g = "utr"
    elif r_art >= 0.85 and r_suf >= 0.3:
        g = "neu"
    else:
        continue
    rows.append({"noun": w, "gender": g, "n_en": n_en, "n_ett": n_ett, "def_utr": s_utr, "def_neu": s_neu, "freq": form[w]})
rows.sort(key=lambda r: -r["freq"])
json.dump({"source": f"{REPO}@{REV}:{FILES}", "rows": rows}, open(OUT, "w"), ensure_ascii=False, indent=0)
c = collections.Counter(r["gender"] for r in rows)
print(len(rows), c)
