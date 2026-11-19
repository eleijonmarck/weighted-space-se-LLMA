# Tooling log

| Tool | What it did | What a human must check |
|---|---|---|
| Claude Code, with model Claude Opus 5.5 | Read the paper and the SE.LLMA page, ran the scouting probes, chose the design, wrote all code and docs, ran all experiments | Lexicon quality, template naturalness, the choice of metrics |
| Claude Code sub-agents, 3 in parallel | Web research: critique sources for the paper, including the logs repo and follow-ups; Swedish datasets, models and licenses; the weight-editing literature | Each agent was told to separate verified quotes from guesses. I re-checked key facts that the design depends on, such as the dataset licenses on the HF Hub and the edited tensors, with code |
| Hugging Face Hub API | Model, dataset and license lookups, downloads | |

No frontier model acted as the "weight programmer" inside the experiment. All edits are built by fixed code from activations and weights. That choice is deliberate: it gives a baseline that any LLM-written edit must beat. The `Frontier model as the "programmer"` section in `GETTING_STARTED.md` explains how to add that arm.

Every number in `README.md` comes from a JSON file in `results/` that the scripts wrote. No number was typed by hand.
