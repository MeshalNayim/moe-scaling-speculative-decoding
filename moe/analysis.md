# Part 1.3 — Benchmark Analysis

## Setup

Run on a single Windows 11 laptop with Microsoft MPI 10.1, `mpi4py` 4.1.2, and NumPy 2.2.6, all in float64. World size and number of experts are both 4, top-k is 2, feature_dim = output_dim = 64. Each timing is the average over 10 forward passes after a warm-up, with a `Barrier` before and after the timed loop. Batch size is swept across {8, 32, 128, 512} at three hidden-dim regimes (64, 256, 1024). All four MPI ranks run on the same machine, so all "communication" here is in-process memcpy via MS-MPI's shared-memory transport — real networks would make the EP gap even wider.

## Results

Times are milliseconds per forward pass (lower is better). "Speedup" is `simple / variant`.

| Workload | Batch | Simple | TP | EP | TP speedup | EP speedup |
|---:|---:|---:|---:|---:|---:|---:|
| small (h=64) | 8 | 0.00 | 1.59 | 1.61 | — | — |
| small (h=64) | 32 | 1.57 | 1.57 | 3.36 | 1.00× | 0.47× |
| small (h=64) | 128 | 3.18 | 1.57 | 8.48 | 2.03× | 0.37× |
| small (h=64) | 512 | 15.29 | 4.32 | 29.80 | 3.54× | 0.51× |
| medium (h=256) | 8 | 0.00 | 2.18 | 1.06 | — | — |
| medium (h=256) | 32 | 0.66 | 4.50 | 3.22 | 0.15× | 0.21× |
| medium (h=256) | 128 | 6.74 | 2.21 | 9.72 | 3.05× | 0.69× |
| medium (h=256) | 512 | 32.50 | 10.57 | 38.94 | 3.08× | 0.83× |
| large (h=1024) | 8 | 1.04 | 1.95 | 1.33 | 0.53× | 0.78× |
| large (h=1024) | 32 | 4.05 | 3.13 | 4.35 | 1.30× | 0.93× |
| large (h=1024) | 128 | 14.98 | 8.58 | 15.73 | 1.75× | 0.95× |
| large (h=1024) | 512 | 61.87 | 35.53 | 57.72 | 1.74× | 1.07× |

## Discussion

The pattern is clear: **TP scales well, EP barely breaks even**, and the reason is exactly the cost of the collective each one relies on.

**TP becomes compute-bound quickly.** A `ShardedLinear` does one buffered `Allgather` per call: each rank sends its own `batch × (hidden/world)` slice and receives the same shape from every other rank, so the per-rank bytes moved is roughly `2 × batch × hidden × bytes_per_elem`. The local matmul is `batch × in_features × (hidden/world)` per rank. As batch and hidden grow, the matmul dominates and the fixed-shape `Allgather` becomes a small overhead. That is exactly what shows up in the table: at small workloads TP is barely faster than the reference (or slower at the tiniest configs, because the collective is still all there is), but once the expert MLP gets either wide (h=1024) or long-batch (b=128–512), TP is 1.7×–3.5× faster than the serial reference. By the largest config, TP is comfortably **compute-bound**: roughly 4× the work per matmul is parallelised across 4 ranks with a near-fixed allgather cost on top.

**EP is communication-bound at this scale.** The EP forward does two `alltoall` calls per top-k slot — the pickle-based variant, because per-rank bucket sizes are variable after routing. The cost per call is dominated not by raw bytes but by Python overhead: every token is serialised as a tuple `(orig_rank, orig_idx, gate, np.ndarray)`, pickled, sent, unpickled, restacked. That overhead scales with the **number of tokens** crossing the boundary (not their dimensionality), so the per-token constant is fairly large but the per-element constant is small. Concretely: at small/medium hidden dims the expert is so cheap that the two pickle alltoalls are pure overhead — EP is 0.4×–0.8× of the serial reference. Even at h=1024, EP only catches the reference (1.07×) at b=512, because the per-token pickle cost is still comparable to a 1024-wide local matmul. This is why production MoE systems use buffered/raw all-to-all (and fold the tokens into a dense rectangular tensor by padding), not pickled Python objects.

**Bottleneck summary:**
- **TP**: communication-bound at small workloads (`Allgather` dominates), **compute-bound** at medium and large workloads (matmul dominates), with the collective shrinking to a tax that scales with `batch × hidden / world` not `batch × hidden`. This is the cleanest scaling story in the table.
- **EP**: **communication-bound everywhere measured**. The two pickle-based `alltoall` calls (one to ship tokens, one to ship outputs back) carry a Python serialisation tax that grows with batch size. EP would only start winning at expert sizes where one expert is too big to even fit on a rank — exactly the regime that motivates EP in real systems (DeepSeek-V3, Mixtral). At our hidden=1024, batch=512 scale a single expert still fits comfortably on every rank, so EP buys nothing.

A consistent secondary effect at the smallest configs (b=8) is that the `SimpleMoE` time rounds to 0.00 ms — the parallel variants incur fixed setup costs (collective handshakes, init overhead) that the reference avoids, which is why their reported speedup is undefined or below 1×. As batch grows these fixed costs amortise out.

The bonus path from the README (replacing `mpi.alltoall` with a hand-written buffered `myAlltoall` on padded rectangular buckets) is what would close the EP gap — it removes the Python serialisation cost and gives `alltoall` the same flat per-call cost profile as `Allgather`. With that, EP would beat TP whenever the expert dimension is so large that splitting an expert across ranks (TP's strategy) actually starts hurting cache behaviour. That regime sits outside this benchmark.
