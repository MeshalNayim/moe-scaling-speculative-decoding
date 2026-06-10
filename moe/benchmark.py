"""Benchmark harness for Part 1.3.

Sweeps batch_size across three hidden_dim settings (small / medium / large)
to expose how each parallel variant scales. With world_size = num_experts = 4,
this shows the regime change from comm-bound (small workload) to
compute-bound (large workload).

Run with:
    mpiexec -n 4 python part1/benchmark.py        (Windows / MS-MPI)
    mpirun  -n 4 python part1/benchmark.py        (Linux / Open MPI)
"""
import time

import numpy as np

from mpi_wrapper import mpi
from rng import get_rng, register_rng
from moe import SimpleMoE, MoE_EP, MoE_TP


def _reset(rank):
    register_rng("expert", np.random.RandomState(0))
    register_rng("router", np.random.RandomState(0))
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))


def run_moe(moe_type, batch_size, feature_dim, hidden_dim, output_dim,
            num_experts, topk=2, n_iters=10):
    rank = mpi.Get_rank()

    if rank == 0:
        X = get_rng().randn(batch_size, feature_dim)
    else:
        X = None
    X = mpi.bcast(X, root=0)

    model_cls = {"simple": SimpleMoE, "tp": MoE_TP, "ep": MoE_EP}[moe_type]
    moe = model_cls(input_dim=feature_dim, hidden_dim=hidden_dim,
                    output_dim=output_dim, num_experts=num_experts, topk=topk)

    _ = moe(X)        # warm-up (the first call sometimes pays one-time costs)
    mpi.Barrier()

    t0 = time.time()
    for _ in range(n_iters):
        _ = moe(X)
    mpi.Barrier()
    avg_ms = 1000 * (time.time() - t0) / n_iters
    return avg_ms


def benchmark_moe():
    rank = mpi.Get_rank()
    ws = mpi.Get_size()

    # Three workload sizes (hidden_dim drives expert MLP cost).
    # batch_size is the swept axis within each workload.
    hidden_sizes = [
        ("small",  64),
        ("medium", 256),
        ("large",  1024),
    ]
    batch_sizes = [8, 32, 128, 512]

    if rank == 0:
        print(f"world_size = {ws} (== num_experts)")
        print(f"feature_dim = output_dim = 64,  topk = 2,  n_iters = 10\n")
        print(f"{'workload':>8}  {'batch':>5}  "
              f"{'simple (ms)':>12}  {'tp (ms)':>10}  {'ep (ms)':>10}  "
              f"{'tp speedup':>11}  {'ep speedup':>11}")
        print("-" * 84)

    for label, hidden in hidden_sizes:
        for batch in batch_sizes:
            _reset(rank)
            t_simple = run_moe("simple", batch, 64, hidden, 64, ws)
            _reset(rank)
            t_tp = run_moe("tp",     batch, 64, hidden, 64, ws)
            _reset(rank)
            t_ep = run_moe("ep",     batch, 64, hidden, 64, ws)

            if rank == 0:
                sp_tp = t_simple / t_tp if t_tp > 0 else float("inf")
                sp_ep = t_simple / t_ep if t_ep > 0 else float("inf")
                print(f"{label:>8}  {batch:>5}  "
                      f"{t_simple:>12.2f}  {t_tp:>10.2f}  {t_ep:>10.2f}  "
                      f"{sp_tp:>10.2f}x  {sp_ep:>10.2f}x")
        if rank == 0:
            print()


if __name__ == "__main__":
    benchmark_moe()
