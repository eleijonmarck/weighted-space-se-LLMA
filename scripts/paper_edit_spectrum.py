import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_paper_edit import get, tensors

from wsp_sv import core

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = tensors(core.MODEL_ID, core.MODEL_REV)
    edited = tensors(core.EDITED_ID, core.EDITED_REV)
    out = {}
    for L in (15, 19):
        k = f"model.language_model.layers.{L}.mlp.down_proj.weight"
        a = get(core.MODEL_ID, core.MODEL_REV, base[k]).float()
        b = get(core.EDITED_ID, core.EDITED_REV, edited[k]).float()
        d = b - a
        cols = (d.abs().sum(0) > 0).nonzero().flatten()
        s = torch.linalg.svdvals(d[:, cols])
        e = (s ** 2) / (s ** 2).sum()
        u = torch.linalg.svd(d[:, cols]).U[:, 0]
        proj = a[:, cols] - torch.outer(u, u @ a[:, cols])
        out[L] = {"dtype": str(get(core.MODEL_ID, core.MODEL_REV, base[k]).dtype), "cols_changed": len(cols),
                  "energy_top1": float(e[0]), "energy_top3": e[:3].tolist(),
                  "rel_err_vs_rank1_projection": float((proj - b[:, cols]).norm() / d[:, cols].norm())}
        print(L, out[L])
    (ROOT / "results" / "paper_edit_spectrum.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
