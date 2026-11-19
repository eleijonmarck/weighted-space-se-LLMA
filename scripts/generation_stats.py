import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

from wsp_sv import core

ROOT = Path(__file__).resolve().parents[1]


def distinct4(texts):
    grams = [tuple(w[i : i + 4]) for w in (re.findall(r"\w+", t.lower()) for t in texts) for i in range(len(w) - 3)]
    return len(set(grams)) / max(1, len(grams))


def main(path):
    g = json.loads(Path(path).read_text())
    tok = AutoTokenizer.from_pretrained(core.MODEL_ID, revision=core.MODEL_REV)
    print("tokens of ' nyckel':", [tok.decode([i]) for i in tok.encode(" nyckel")], " ' ny':", tok.encode(" ny"))
    for name, v in g.items():
        words = [w for t in v["samples"] for w in re.findall(r"\w+", t.lower())]
        ny = sum(w.startswith("nyckel") for w in words)
        v["stats"] = {"words": len(words), "nyckel_words": ny, "nyckel_per_1k": 1000 * ny / max(1, len(words)),
                      "distinct_4gram": distinct4(v["samples"])}
        print(f"{name:16s} mismatch={v['mismatch_rate']:.3f} ({v['mismatches']}/{v['np_checked']}) nyckel/1k={v['stats']['nyckel_per_1k']:.2f} distinct4={v['stats']['distinct_4gram']:.3f}")
    Path(path).write_text(json.dumps(g, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
