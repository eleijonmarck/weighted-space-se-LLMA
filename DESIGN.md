# Design: where weight-space programming fits in a Swedish LLM

This file is the plan. `README.md` is the research note with results. `GETTING_STARTED.md` tells you how to run things.

## 1. How SE.LLMA will most likely build a strong Swedish model

SE.LLMA says it will not train from scratch. It adapts open models with open weights and data, and it starts small and scales up. The pipeline that follows from that, and what each stage costs:

| Stage | What it does | Cost | What can go wrong in Swedish |
|---|---|---|---|
| 1. Pick base | Choose an open model with some Swedish ability | Low | Weak Swedish grammar, bad tokenizer fit |
| 2. Tokenizer | Keep it or extend it with Swedish tokens | Medium | Swedish words split into many pieces |
| 3. Continued pretraining | Train on curated Swedish text from news, books and public sector | High, most of the compute | Forgetting, English capability loss |
| 4. Instruction tuning and preference tuning | Teach chat behavior in Swedish | Medium | Language drift, translated-data style |
| 5. Evaluation | EuroEval, Swedish BLiMP, ScaLA, own tasks | Low per run | Metrics that hide real errors |
| 6. Patch and iterate | Fix specific failures, then go to the next model size | Unknown | Every fix needs a new training run |

Weight-space editing can only help in two places. Stage 6 is the obvious one: a cheap, local fix for a known failure without a new training run. Stage 3 is the less obvious and more valuable one. Before you spend compute on a failure, a weight-level diagnosis can tell you whether the model lacks the skill or has it but does not use it. The paper itself says edits "could remove an obstacle that hid an ability, but … could not add an ability the model lacked." If that holds, a cheap test for "obstacle versus missing ability" tells SE.LLMA where data and compute must go and where a patch is enough.

So the question worth testing is not "can a frontier model edit weights". It is: **for a real Swedish failure in a small open model, is the error an obstacle that a targeted edit can remove, and does that edit beat cheap alternatives without damage?**

## 2. What the paper shows, in one paragraph

The evidence on reading weights is real but comes from small, toy-sized networks. The evidence on writing weights in a real LLM comes from one experiment: the Qwen3.5-0.8B looping fix. There, the published logs show that each of the 50 forward passes returned a practice accuracy. The chosen edit was the top-scoring point of a sweep of about 8 candidates that varied strength and size. Gains at 10 and 25 passes were small. That is score-driven search over a known kind of edit, a single-direction removal as in Arditi et al. 2024, applied to a failure that Qwen's own model card documents. Full critique is in `README.md`.

## 3. Three designs that I considered

### Design A: fix language confusion by removing a direction
Idea: small models answer Swedish prompts in English, Norwegian or Danish. Find the "not Swedish" direction and remove it, as Arditi did for refusal.
Check before building: I ran 16 Swedish prompts on Qwen3.5-0.8B with greedy decoding and thinking off. 15 replies were Swedish. The 16th was a short Swedish sentence that the language detector labeled Nynorsk.
Verdict: **rejected.** The failure is close to absent in this model. I would have to provoke it first, and a fix for a provoked failure has no value for SE.LLMA. The Language Confusion Benchmark also has no Swedish.

### Design B: audit the paper's own edit on Swedish
Idea: the edited Qwen checkpoint is public. Only two `down_proj` tensors changed. Measure what the edit does to Swedish perplexity and Swedish grammar.
Value: it is a cheap, direct collateral-damage check of the headline result in our language.
Limit: it does not test whether an edit can fix a Swedish problem. Long reasoning runs with up to 32k tokens are also too slow on this machine with the reference linear-attention kernels.
Verdict: **kept as a side check**, not as the main experiment.

### Design C: en/ett gender agreement, obstacle or missing ability (chosen)
Idea: Swedish nouns are common gender, *en*, or neuter, *ett*. The article, adjectives and pronouns must agree. In 16 free answers the model wrote "en ekonomisk begrepp" and "en mekanisk system". In a quick probe, every noun it got wrong was split into 2 or 3 tokens.
Steps:
1. **Read.** Is noun gender linearly present in the weights, through the unembedding row, and in the residual stream? Compare with how well the model uses it.
2. **Write.** Build a rank-1 edit on one layer's `down_proj`, the same tensor the paper edited. It reads the article-gender signal from the MLP input and writes more of the existing article-gender difference into the residual stream. If the error is an obstacle, a weak signal that exists but is not used, then amplifying it should help. If the ability is missing, it should not.
3. **Feedback ladder.** Build the same edit with no score feedback at all, then with K dev evaluations, then compare with gradient training (LoRA). This tests the paper's key distinction: reasoning about weights versus search driven by feedback.
Why it is the best of the three: the failure is real and frequent in this model. It is cheap to measure with log-probs only, no judge model. It is purely Swedish. It is close to SE.LLMA's goal of "finer linguistic nuances". It also links to a concrete SE.LLMA decision, the tokenizer.

## 4. Self-grilling, as a design tree

Each line is a decision, my answer, and the evidence.

1. **Is the target failure real in the subject model?** Yes. Article errors on 15 to 25% of common nouns in minimal pairs, and visible in free text. Language confusion was not real, see Design A.
2. **Which model?** Qwen3.5-0.8B, Apache-2.0, the paper's own subject. Its edited checkpoint enables Design B.
3. **What does "accuracy" mean?** This changed after I looked at the data. A raw two-way comparison of `log P(noun | ett)` against `log P(noun | en)` gave 31% for common nouns. That is not a gender failure. After *en*, which also means "one" and works as a pronoun, all nouns lose probability. The fix is two offset-free metrics:
   - **Gender AUC** per template: does `log P(N|ett) − log P(N|en)` rank neuter nouns above common ones? This is the primary, pre-registered metric.
   - **Mismatch mass**: after *ett*, the share of balanced noun probability on common nouns, and the reverse. This is the quantity closest to generation errors.
   I also report sentence-level minimal-pair accuracy, which is the BLiMP standard. It mixes in the article prior.
4. **What leaks between selection and evaluation?** Nouns are split by lemma into dev and test. Templates are split too: dev uses 3 carrier sentences, test uses 3 different ones. All directions, layers and strengths are chosen on dev only. The LiU Swedish BLiMP gender sets are never used for selection.
5. **What would a fake fix look like?** Three things, and each has a control:
   - Pushing everything toward *ett*: the per-template AUC is blind to that, and the mismatch mass per article shows it.
   - Any large change helping by chance: a random write direction with the same norm, over 5 seeds.
   - Damage to the language: Swedish and English perplexity, and non-gender Swedish BLiMP.
6. **What are the cheap alternatives that a real method must beat?** A few-shot prompt with gender examples, and LoRA on the same dev sentences.
7. **What counts as "the model has the information"?** A linear probe trained on dev nouns and tested on test nouns, with a shuffled-label control. A probe alone does not prove use. That is why the edit is the causal test.
8. **Where must the edit act?** On positions that predict the noun tokens after the article. For multi-token nouns, gender must reach later sub-tokens. I report single-token and multi-token nouns separately because of the tokenizer finding.
9. **Decision rule, fixed before test numbers:**
   - Obstacle supported if:
     - test gender AUC rises by at least 0.03 over base,
     - the gain is more than all 5 random-direction seeds,
     - Swedish perplexity rises less than 2%,
     - non-gender Swedish BLiMP drops less than 1 point.
   - Missing ability, or no practical value, if any of these holds:
     - the gain is within the random spread,
     - the gain comes only with real damage,
     - few-shot or LoRA reach the same gain at lower cost.
10. **What will the result not prove?** It is one model, one phenomenon, and one edit family, with no frontier model acting as the "programmer". A null result rules out this edit family, not every possible edit.

## 5. Roadmap after the first step

1. **First step, this repo:** diagnosis plus rank-1 edit plus controls on Qwen3.5-0.8B for en/ett agreement.
2. **If the edit works:** test whether the same direction transfers to a model after Swedish continued pretraining, and whether training washes it out. That decides if editing is a "patch at the end" tool.
3. **If it does not work:** treat gender as a data and tokenizer problem. Measure gender AUC by token count before and after tokenizer extension and continued pretraining. Use the diagnosis as a cheap triage metric, not as a fix.
4. **The most informative follow-up either way:** run the same setup on the base and on an SE.LLMA continued-pretraining checkpoint of the same model. If continued pretraining closes the gap mainly on multi-token nouns, the tokenizer is the root cause and editing is the wrong tool.
