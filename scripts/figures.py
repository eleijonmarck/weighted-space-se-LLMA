import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#9a9890"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_layers():
    sel = json.loads((RES / "selection.json").read_text())
    base = sel["base_dev_select"]["gender_auc"]
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=SURFACE)
    style(ax)
    for k, c in zip((1.0, 2.0, 4.0), (S1, S2, S3)):
        pts = sorted((g["layer"], g["gender_auc"]) for g in sel["grid"] if g["k"] == k)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, linewidth=2, marker="o", markersize=4, label=f"k = {k:g}")
    ax.axhline(base, color=NEUTRAL, linewidth=1.5, linestyle="--")
    ax.text(0.2, base + 0.004, "base model", color=INK2, fontsize=9)
    ax.set_xlabel("layer whose down_proj is edited", color=INK2)
    ax.set_ylabel("gender AUC, dev nouns", color=INK2)
    ax.set_title("Amplifying the article-gender signal only helps in the last layers", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "layer_sweep.png", dpi=200)


def fig_tradeoff():
    arms = json.loads((RES / "arms.json").read_text())
    boot = json.loads((RES / "bootstrap.json").read_text())
    post = json.loads((RES / "posthoc_no_adj.json").read_text())
    base_ppl = arms["base"]["regression"]["ppl_sv"]
    base_auc = boot["base"]["auc"]
    pts = []
    for name, label in (("zero_feedback", "edit, no feedback"), ("dev_selected", "edit, dev-selected"),
                        ("weights_write_dir", "edit, weights-only write"), ("lora_dev", "LoRA, same dev data")):
        pts.append((label, arms[name]["regression"]["ppl_sv"] / base_ppl - 1, boot[name]["delta_vs_base"], S1 if "edit" in label else S2))
    pts.append(("edit, post-hoc no adjective", post["regression"]["ppl_sv"] / base_ppl - 1, post["target"]["gender_auc"] - base_auc, S1))
    bud = json.loads((RES / "posthoc_budget_L22_k05.json").read_text())
    pts.append(("edit, post-hoc in budget", bud["regression"]["ppl_sv"] / base_ppl - 1, bud["target"]["gender_auc"] - base_auc, S1))
    pts.append(("few-shot prompt", 0.0, boot["fewshot_prompt"]["delta_vs_base"], S3))
    rnd = [(arms[f"random_seed{s}"]["regression"]["ppl_sv"] / base_ppl - 1, boot[f"random_seed{s}"]["delta_vs_base"]) for s in range(5)]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    style(ax)
    ax.axhline(0, color=NEUTRAL, linewidth=1)
    ax.axvline(0, color=NEUTRAL, linewidth=1)
    ax.scatter([r[0] * 100 for r in rnd], [r[1] for r in rnd], s=40, color=NEUTRAL, edgecolor=SURFACE, linewidth=1.5, label="random direction, 5 seeds")
    offsets = {"edit, no feedback": (-40, 16), "edit, dev-selected": (8, -10), "edit, post-hoc no adjective": (14, 8),
               "edit, post-hoc in budget": (8, -10), "edit, weights-only write": (6, 6), "LoRA, same dev data": (-110, -4),
               "few-shot prompt": (6, 4)}
    for label, x, y, c in pts:
        ax.scatter([x * 100], [y], s=60, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(label, (x * 100, y), xytext=offsets.get(label, (6, 4)), textcoords="offset points", fontsize=8.5, color=INK)
    ax.scatter([0], [0], s=60, color=INK, zorder=3)
    ax.annotate("base", (0, 0), xytext=(6, -12), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("Swedish perplexity change, %", color=INK2)
    ax.set_ylabel("gender AUC change on test nouns", color=INK2)
    ax.set_title("Edits gain little gender AUC per unit of Swedish perplexity", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.png", dpi=200)


def fig_frontier():
    fr = json.loads((RES / "frontier.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=SURFACE)
    style(ax)
    ax.axvspan(-2, 2, color="#eef4fb", zorder=0)
    ax.axhline(0.03, color=NEUTRAL, linewidth=1, linestyle="--")
    ax.text(6.2, 0.032, "required gain 0.03", color=INK2, fontsize=8)
    ax.text(-1.9, -0.04, "allowed cost < 2%", color=INK2, fontsize=8)
    shown = [p for p in fr["points"] if p["ppl_sv_rel"] < 0.15]
    xs = [p["ppl_sv_rel"] * 100 for p in shown]
    ys = [p["dev_auc_gain"] for p in shown]
    ax.scatter(xs, ys, s=36, color=S1, edgecolor=SURFACE, linewidth=1.5)
    for p, x, y in zip(shown, xs, ys):
        if y > 0.035 or (x < 2 and y > 0.025):
            off = (-52, 4) if (p["layer"], p["k"]) == (22, 0.5) else (5, 3)
            if (p["layer"], p["k"]) == (21, 0.5):
                off = (-52, -10)
            ax.annotate(f"L{p['layer']} k={p['k']:g}", (x, y), xytext=off, textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xlabel("Swedish perplexity change, %. Layer 23 points above 15% are left out", color=INK2)
    ax.set_ylabel("gender AUC gain, dev", color=INK2)
    ax.set_title("Inside the 2% budget the best dev gain is +0.032, at layer 22, k = 0.5", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "frontier.png", dpi=200)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_layers()
    fig_tradeoff()
    if (RES / "frontier.json").exists():
        fig_frontier()
