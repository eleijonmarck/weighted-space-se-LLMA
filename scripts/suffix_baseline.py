import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    """How much of noun gender is predictable from spelling alone? Train on the dev nouns used for the edit
    and the LoRA control, test on the same balanced test nouns as the main experiment."""
    sp = json.loads((ROOT / "results" / "splits.json").read_text())
    dev = [tuple(x) for x in sp["dev"]]
    test = [tuple(x) for x in sp["test_balanced"]]
    out = {}
    for n_dev, label in ((400, "dev_400_like_edit"), (len(dev), "all_dev")):
        rng = np.random.default_rng(0)
        d = [dev[i] for i in rng.permutation(len(dev))[:n_dev]] if n_dev < len(dev) else dev
        for kind, ana, ngr in (("last_chars_1to4", "char", (1, 4)),):
            vec = CountVectorizer(analyzer=lambda w: ["$" + w[-k:] for k in range(ngr[0], ngr[1] + 1)])
            Xtr = vec.fit_transform([n for n, _, _ in d])
            Xte = vec.transform([n for n, _, _ in test])
            ytr = np.array([g == "neu" for _, g, _ in d])
            yte = np.array([g == "neu" for _, g, _ in test])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
            out[f"{label}_{kind}"] = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    print(out)
    (ROOT / "results" / "suffix_baseline.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
