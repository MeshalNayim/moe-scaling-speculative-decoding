# PA3 Part 3 Report

Target is `EleutherAI/pythia-1.4b-deduped`, draft is `EleutherAI/pythia-160m-deduped`, both in float16 on an RTX 3060 Laptop. Both decoders are greedy so verification can compare draft tokens against the target's argmax. Each run generates 100 tokens averaged over 3 trials. Baseline is `target_model.generate()` with the same greedy settings.

## Part 3.2 results (K = 8)

| Prompt | Acceptance | Speedup |
|---|---|---|
| The future of artificial intelligence is | 91.7% | 1.27x |
| Write a short story about a robot learning to feel emotions | 91.4% | 1.32x |
| Write the lyrics to the song 'Happy Birthday' | 93.8% | 1.47x |

Both targets (>= 1.0x speedup, >= 75% acceptance) clear on every prompt.

## Part 3.3 sweep

Prompt 1, 100 tokens, 3 runs. Baseline is 3.71 s.

| K | Acceptance | Speedup |
|---|---|---|
| 2 | 97.1% | 1.25x |
| 4 | 95.2% | 1.40x |
| 8 | 91.7% | 1.51x |
| 16 | 85.7% | 1.43x |

Acceptance drops as K grows because one mismatch wastes every following draft token. Speedup peaks in the middle: small K wastes the target verification call on too few tokens, large K wastes too many draft proposals after the first miss. K = 8 is the best point on this hardware.

## Optimizations

The vectorized verification is the main win: K draft tokens checked in one target forward pass instead of K. Returning the target's argmax at the mismatch (or at the slot past the draft) as a third value from `verify_tokens_vectorized` saves an extra target call per round. Without it most of the speedup disappears.

Both models run in fp16, which halves VRAM (3.2 GB instead of 6.3 GB) and is necessary to fit on the 6.4 GB GPU. `use_cache=True` lets the draft reuse its KV cache across the K draft steps within a round. Both forward passes are wrapped in `torch.no_grad()` to skip autograd graph construction.

The draft does not reuse its KV cache across rounds; each round re-encodes the full prefix. Pythia-160M is small enough that this is not the bottleneck on this hardware, but with a larger draft it would matter.

## Bonus 3.B: Prompt Lookup Decoding

PLD replaces the draft model call when the last n-gram of the running sequence has appeared earlier. The tokens that followed that earlier occurrence become the proposal. On a miss it falls back to the draft model. Verification is unchanged. I used n-gram size 3 and K = 8.

| Prompt | Baseline (s) | Vanilla SD | PLD + SD | PLD hit rate |
|---|---|---|---|---|
| The future of AI | 3.72 | 1.27x | 4.27x | 83.3% |
| Robot story | 3.75 | 1.32x | 3.02x | 69.2% |
| Happy Birthday lyrics | 3.72 | 1.47x | 4.31x | 83.3% |

Acceptance stays at 91-94% because the proposal source does not change verification. What drops is the cost of producing the proposal. Vanilla SD spends 8 Pythia-160M forward passes per round; on a PLD hit that becomes a few array operations. Pythia output is repetitive enough that hit rates land in the 70-85% range, so most rounds skip the draft model entirely.

On non-repetitive workloads PLD's hit rate would fall and the speedup would degrade back toward vanilla SD. The fallback keeps the worst case the same as before.

## Summary

Speedup depends on two things: how often the draft and target agree at the greedy argmax, and how cheap the verification step is relative to the baseline target call. PLD adds a third axis by removing the draft-model cost when the n-gram matches. Together that pushes peak speedup from 1.5x to 4.3x on these prompts.
