# moe-scaling-speculative-decoding

Three independent studies in efficient LLM training and inference: distributed
Mixture-of-Experts, scaling-law cost modeling, and speculative decoding.

## 1 — Distributed Mixture of Experts (`moe/`)

Two ways to parallelize an MoE layer across ranks, both on a hand-written MPI
collectives layer:

- **Tensor parallel** (`MoE_TP`): every rank holds a column shard of every expert,
  computes a partial output for the whole batch, and the ranks assemble the full
  result with an all-gather / all-reduce.
- **Expert parallel** (`MoE_EP`): each rank owns one whole expert; after routing,
  tokens are shipped to the owning rank with all-to-all and the results shipped
  back.

`moe/analysis.md` benchmarks both against a serial reference across batch size and
hidden dimension. The headline: TP scales cleanly and becomes compute-bound (up to
~3.5× over serial at the largest size), while EP stays communication-bound at this
scale — the pickle-based all-to-all carries a per-token serialization tax that
only pays off once an expert is too big to fit on one rank, which is exactly the
regime (DeepSeek-V3, Mixtral) that motivates EP in production.

```bash
mpiexec -n 4 python moe/test_moe.py
mpiexec -n 4 python moe/benchmark.py
```

## 2 — Scaling-law training-cost analysis (`scaling_laws/`)

`model_training_cost_analysis.py` computes parameter counts, forward FLOPs, and
peak training memory directly from a HuggingFace-style config, accounting for the
real architecture details:

- **Llama-3 8B**: GQA (32 query heads vs. 8 KV heads), SwiGLU MLP, RMSNorm,
  untied embeddings.
- **DeepSeek-V3** (bonus): MLA attention and the dense-then-MoE layer split —
  reports total vs. activated parameters per token (671B stored, ~37B active).
  `scaling_laws/moe.md` argues MoE's capacity-per-FLOP advantage against its
  memory/infrastructure cost at a fixed budget.

It also solves a design problem: under a $5M compute budget, pick the GPU
(H100 / H200 / B200 by effective FLOPs at 40% MFU) and use the scaling law to back
out the compute-optimal (N, D). `scaling_laws/my_model_config.json` is the
resulting ~98B-parameter design.

```bash
python scaling_laws/model_training_cost_analysis.py --model_config scaling_laws/llama3_8b_config.json
python scaling_laws/model_training_cost_analysis.py --training_budget 5000000
```

## 3 — Speculative decoding (`speculative_decoding/`)

A single-batch speculative decoder: a small draft model (Pythia-160M) proposes K
tokens, the larger target (Pythia-1.4B) verifies them in one batched forward pass,
and the run accepts the longest matching prefix. The win is checking K tokens with
one target call instead of K, plus returning the target's own next token at the
mismatch so no verification call is wasted.

Measured on an RTX 3060 Laptop (fp16, greedy, 100 tokens):

| K  | acceptance | speedup vs. target-only |
|---:|-----------:|------------------------:|
| 2  | 97%        | 1.25× |
| 4  | 95%        | 1.40× |
| 8  | 92%        | 1.51× |
| 16 | 86%        | 1.43× |

A bonus prompt-lookup-decoding variant (n-gram proposals with a draft-model
fallback) pushes peak speedup to ~4.3× on repetitive prompts. The full sweep is in
`speculative_decoding/report.md`; the implementation is the notebook.

```bash
jupyter notebook speculative_decoding/speculative_decoding.ipynb
```

## Layout

```
moe/                     TP and EP Mixture-of-Experts on MPI + benchmark/analysis
scaling_laws/            param/FLOP/memory cost model, budget-optimal model design
speculative_decoding/    draft/target speculative decoder notebook + report
```

---

Coursework for CSE 291 / DSC 291 (Machine Learning Systems), UC San Diego.
