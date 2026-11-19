import copy
import random

import torch
from peft import LoraConfig, get_peft_model

from .core import device

LORA_CFG = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": ["down_proj"]}
TRAIN_CFG = {"lr": 2e-4, "epochs": 2, "batch_size": 16}


def train_lora(model, tok, items, seed=0):
    """Standard fine-tuning control: LoRA on down_proj, LM loss on the correct dev sentences only."""
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    m.train()
    m = get_peft_model(m, LoraConfig(**LORA_CFG))
    bos = [tok.convert_tokens_to_ids("<|endoftext|>")]
    texts = [it.prefix_ok + it.cont for it in items]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=TRAIN_CFG["lr"])
    rng = random.Random(seed)
    bs = TRAIN_CFG["batch_size"]
    for _ in range(TRAIN_CFG["epochs"]):
        rng.shuffle(texts)
        for s in range(0, len(texts), bs):
            seqs = [bos + tok.encode(t) for t in texts[s : s + bs]]
            L = max(map(len, seqs))
            ids = torch.zeros((len(seqs), L), dtype=torch.long)
            lab = torch.full((len(seqs), L), -100, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for r, q in enumerate(seqs):
                ids[r, : len(q)] = torch.tensor(q)
                lab[r, 1 : len(q)] = torch.tensor(q[1:])
                att[r, : len(q)] = 1
            loss = m(input_ids=ids.to(device()), attention_mask=att.to(device()), labels=lab.to(device())).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    m = m.merge_and_unload()
    m.eval()
    return m
