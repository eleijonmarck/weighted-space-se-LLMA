import json, sys, re, random, collections
src, out, min_freq = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(src))
rows = [r for r in d["rows"] if r["freq"] >= min_freq and re.fullmatch(r"[a-zåäö]{3,}", r["noun"])]
json.dump({"source": d["source"], "min_freq": min_freq, "rows": rows}, open(out, "w"), ensure_ascii=False, indent=0)
print(len(rows), collections.Counter(r["gender"] for r in rows))
random.seed(1)
for g in ("neu", "utr"):
    print(g, " ".join(random.sample([r["noun"] for r in rows if r["gender"] == g], 80)))
