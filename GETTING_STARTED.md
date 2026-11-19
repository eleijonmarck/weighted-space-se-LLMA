# Getting started

## What is in this repo

| File | Purpose |
|---|---|
| `README.md` | Research note: hypothesis, method, results, limits, recommendation |
| `DESIGN.md` | Plan: where weight editing fits in a Swedish LLM pipeline, 3 designs, self-review, roadmap |
| `TOOLING.md` | Which AI tools did what |
| `report.pdf` | PDF version of the research note |
| `src/wsp_sv/core.py` | Model loading, noun splits, carrier sentences, log-prob scoring, metrics |
| `src/wsp_sv/edits.py` | Paired activations, rank-1 `down_proj` edit, random control, apply and revert |
| `src/wsp_sv/lora.py` | LoRA fine-tuning control |
| `scripts/build_lexicon.py` | Builds the noun gender lexicon from FineWeb-2 Swedish |
| `scripts/filter_lexicon.py` | Keeps frequent nouns |
| `scripts/run_experiment.py` | Main experiment: probe, edit search on dev, all arms on test, regressions |
| `scripts/generation_check.py` | Free-generation en/ett mismatch count |
| `scripts/audit_paper_edit.py` | Side check: the paper's own Qwen edit, evaluated on Swedish |
| `scripts/paper_edit_spectrum.py` | Checks that the paper's edit is rank 1 and finds its strength |
| `scripts/posthoc_edit.py` | Post-hoc arms: direction without the adjective sentence, or a fixed layer and k |
| `scripts/frontier.py` | Dev-only trade-off of gender AUC gain against Swedish perplexity |
| `scripts/generation_prompt_control.py` | Prompt-only controls in free generation |
| `scripts/generation_stats.py`, `scripts/generation_bootstrap.py` | Repetition, "nyckel" leak, CIs by prompt |
| `scripts/bootstrap.py`, `scripts/suffix_baseline.py` | AUC CIs by noun, and the spelling-only baseline |
| `scripts/figures.py`, `scripts/build_report.py` | Figures and `report.pdf` |
| `results/` | All JSON outputs, per-item scores and generations |

## Requirements

- macOS with Apple silicon, or any Linux or Windows machine with a CPU. A GPU is optional.
- [uv](https://docs.astral.sh/uv/).
- About 10 GB of disk for the model and one FineWeb-2 shard.
- About 16 GB RAM.

On an M5 Pro with MPS, `run_experiment.py` takes about 40 minutes and all steps together about 1.5 hours. On CPU only it is slower. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if an op is missing on MPS.

## Reproduce

```bash
uv sync
uv run python scripts/build_lexicon.py data/lexicon_raw.json data/blocklist.txt
uv run python scripts/filter_lexicon.py data/lexicon_raw.json data/lexicon.json 800
uv run python scripts/run_experiment.py
uv run python scripts/generation_check.py
uv run python scripts/generation_stats.py results/generation.json
uv run python scripts/audit_paper_edit.py
uv run python scripts/paper_edit_spectrum.py
uv run python scripts/posthoc_edit.py
uv run python scripts/frontier.py
uv run python scripts/posthoc_edit.py --layer 22 --k 0.5 --name posthoc_budget_L22_k05
uv run python scripts/generation_prompt_control.py
uv run python scripts/bootstrap.py
uv run python scripts/generation_bootstrap.py
uv run python scripts/suffix_baseline.py
uv run python scripts/figures.py
uv run python scripts/build_report.py
```

The order matters: later scripts read `results/selection.json` and `results/splits.json`.

`data/lexicon.json` is committed, so you can skip the first two steps. Building the lexicon downloads two FineWeb-2 parquet shards, about 5 GB.

## Pinned versions

- Model: `Qwen/Qwen3.5-0.8B` at revision `2fc0636471`.
- Paper edit: `RikiyaT/Qwen3.5-0.8B-weight-space-programming` at `d1671a1787`.
- Seed: 0 for all splits and sampling. Random-direction controls use seeds 1000 to 1004. Generation is greedy.
- Edit settings: `K_GRID`, the sample sizes and the LoRA settings are constants at the top of `scripts/run_experiment.py` and `src/wsp_sv/lora.py`. `results/run_meta.json` records them for the run.
- Python packages: see `uv.lock`.

## Data and licenses

| Data | License | Use here |
|---|---|---|
| Qwen3.5-0.8B and the edited checkpoint | Apache-2.0 | subject model |
| FineWeb-2 `swe_Latn` | ODC-By 1.0 | noun lexicon counts, Swedish perplexity |
| WikiText-103 test | CC BY-SA 3.0 | English perplexity |
| `liu-nlp/swedish-blimp-*`, the subsets used here | Apache-2.0 | external target and regression tests |
| Carrier sentences and generation prompts | written for this repo, MIT | evaluation |

`data/lexicon.json` holds word counts derived from FineWeb-2 and inherits ODC-By.

## How to extend

- **Other model.** Change `MODEL_ID` and `MODEL_REV` in `core.py`. `edits.down_proj` assumes the layout `model.model.layers[i].mlp.down_proj`, which Llama, Qwen, Gemma and Mistral all use.
- **Other phenomenon.** Replace `make_items` with your own minimal pairs. The edit only needs two prefixes per item that differ in the feature you care about.
- **Frontier model as the "programmer".** Give it `results/probe.json`, `results/selection.json` and the weights of one layer, and ask for a write and a read vector. Then score its vectors with `run_arm`. This is the closest replication of the paper's setup.
