from dataclasses import dataclass

import torch

from .core import device


def down_proj(model, layer):
    return model.model.layers[layer].mlp.down_proj


@torch.no_grad()
def paired_activations(model, tok, items, layers, batch_size=32):
    """Mean over noun-predicting positions of (MLP input to down_proj, residual after layer),
    for the 'ett' prefix minus the 'en' prefix of the same item. Returns per-layer tensors
    act_diff [N, 3584], res_diff [N, 1024], plus act_mean (both conditions) [3584]."""
    dev = device()
    bos = [tok.convert_tokens_to_ids("<|endoftext|>")]
    seqs, spans = [], []
    for it in items:
        ett, en = (it.prefix_ok, it.prefix_bad) if it.gender == "neu" else (it.prefix_bad, it.prefix_ok)
        for pre in (ett, en):
            a = bos + tok.encode(pre)
            b = tok.encode(it.cont)
            seqs.append(a + b)
            spans.append((len(a) - 1, len(a) - 1 + it.n_tokens))
    store = {}

    def hook(L):
        def f(_m, inp):
            store[L] = inp[0]
        return f

    hs = [down_proj(model, L).register_forward_pre_hook(hook(L)) for L in layers]
    acts = {L: [] for L in layers}
    ress = {L: [] for L in layers}
    try:
        for s in range(0, len(seqs), batch_size):
            chunk = list(range(s, min(len(seqs), s + batch_size)))
            Lmax = max(len(seqs[i]) for i in chunk)
            ids = torch.zeros((len(chunk), Lmax), dtype=torch.long)
            att = torch.zeros((len(chunk), Lmax), dtype=torch.long)
            for r, i in enumerate(chunk):
                ids[r, : len(seqs[i])] = torch.tensor(seqs[i])
                att[r, : len(seqs[i])] = 1
            out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True)
            for L in layers:
                for r, i in enumerate(chunk):
                    p, q = spans[i]
                    acts[L].append(store[L][r, p:q].float().mean(0).cpu())
                    ress[L].append(out.hidden_states[L + 1][r, p:q].float().mean(0).cpu())
    finally:
        for h in hs:
            h.remove()
    result = {}
    for L in layers:
        A = torch.stack(acts[L])
        R = torch.stack(ress[L])
        result[L] = {
            "act_diff": A[0::2] - A[1::2],
            "res_diff": R[0::2] - R[1::2],
            "act_mean": A.mean(0),
            "res_std": R.std(0).norm(),
        }
    return result


@dataclass
class RankOneEdit:
    layer: int
    write: torch.Tensor
    read: torch.Tensor
    name: str = ""
    offset: float = 0.0

    def delta(self):
        return torch.outer(self.write, self.read)


def gender_edit(stats, layer, k, write_dir=None, name="gender"):
    """W' = W + write @ read^T on layer's down_proj.
    read = unit mean paired MLP-input difference (ett minus en).
    write is scaled so the edit adds k times the mean residual difference between conditions."""
    s = stats[layer]
    read = s["act_diff"].mean(0)
    mu = s["act_mean"] / s["act_mean"].norm()
    read = read - (read @ mu) * mu
    read = read / read.norm()
    gap = (s["act_diff"] @ read).mean()
    d = s["res_diff"].mean(0)
    if write_dir is not None:
        d = write_dir / write_dir.norm() * d.norm()
    write = k * d / gap
    return RankOneEdit(layer, write, read, name, float((s["act_mean"] @ read) / gap))


def random_like(edit, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(edit.write.shape, generator=g)
    return RankOneEdit(edit.layer, r / r.norm() * edit.write.norm(), edit.read, f"random{seed}", edit.offset)


def unembed_gender_dir(model, tok, nouns):
    """Weights only: mean unembedding row of the first token of neuter nouns minus common-gender nouns."""
    W = model.get_output_embeddings().weight.detach().float().cpu()
    rows = {"utr": [], "neu": []}
    for n, g, _ in nouns:
        rows[g].append(W[tok.encode(" " + n)[0]])
    return torch.stack(rows["neu"]).mean(0) - torch.stack(rows["utr"]).mean(0)


class applied:
    def __init__(self, model, edits):
        self.model, self.edits, self.saved = model, edits, []

    def __enter__(self):
        for e in self.edits:
            W = down_proj(self.model, e.layer).weight
            self.saved.append((W, W.detach().clone()))
            W.data += e.delta().to(W.device, W.dtype)
        return self

    def __exit__(self, *a):
        for W, old in self.saved:
            W.data.copy_(old)


def edit_stats(model, edit):
    W = down_proj(model, edit.layer).weight.detach().float().cpu()
    D = edit.delta()
    return {"layer": edit.layer, "rank": 1, "rel_frob": float(D.norm() / W.norm()), "n_params_touched": int(D.numel()),
            "offset_over_gap": edit.offset, "name": edit.name}
