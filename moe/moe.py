"""Mixture-of-Experts: reference, tensor-parallel, and expert-parallel variants.

You will implement `ShardedLinear`, `MoE_TP`, and `MoE_EP` in this file. The
reference `SimpleMoE` and a pre-built `Router` are provided.
"""
import numpy as np

from mpi_wrapper import mpi
from rng import get_rng, rng_context


class Linear:
    """Simple linear layer y = xW + b."""

    def __init__(self, in_features, out_features):
        self.weight = get_rng().randn(in_features, out_features) * 0.01
        self.bias = np.zeros(out_features)

    def __call__(self, x):
        return np.dot(x, self.weight) + self.bias


class Expert:
    """Two-layer MLP expert with ReLU."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        with rng_context("expert"):
            self.fc1 = Linear(input_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class Router:
    """Softmax-gated top-k router (replicated across ranks)."""

    def __init__(self, input_dim, num_experts):
        self.linear = Linear(input_dim, num_experts)

    def __call__(self, x, topk=1):
        logits = self.linear(x)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        indices = np.argsort(-probs, axis=1)[:, :topk]
        gates = np.take_along_axis(probs, indices, axis=1)
        gates = gates / np.sum(gates, axis=1, keepdims=True)
        return indices, gates


# ---------------------------------------------------------------------------
# Reference implementation: not parallel. Use this to verify correctness.
# ---------------------------------------------------------------------------
class SimpleMoE:
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        with rng_context("router"):
            self.router = Router(input_dim, num_experts)

        with rng_context("expert"):
            self.experts = [
                Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
            ]

    def forward(self, x):
        batch_size = x.shape[0]
        indices, gates = self.router(x, self.topk)
        outputs = np.zeros((batch_size, self.output_dim))
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i : i + 1]
                expert_output = self.experts[expert_idx](item)
                outputs[i] += gate * expert_output[0]
        return outputs

    def __call__(self, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# Part 1.1 — Tensor Parallel MoE.
# ---------------------------------------------------------------------------
class ShardedLinear:
    """Linear layer whose weight is column-sharded across MPI ranks.

    Each rank stores a `(in_features, out_features // world_size)` slice of the
    weight matrix. The forward pass produces the *full* output of shape
    `(batch, out_features)` on every rank, which means a collective is required
    to reassemble the columns each rank computed.

    Requires that `out_features` is evenly divisible by the world size.
    """

    def __init__(self, in_features, out_features):
        self.rank = mpi.Get_rank()
        self.world_size = mpi.Get_size()

        assert out_features % self.world_size == 0, (
            f"Output features ({out_features}) must be evenly divisible by "
            f"world size ({self.world_size})"
        )

        self.in_features = in_features
        self.out_features_global = out_features
        self.local_out_features = out_features // self.world_size
        self.output_offset = self.rank * self.local_out_features

        # Initialize local weights and bias
        self.weight = get_rng().randn(in_features, self.local_out_features) * 0.01
        self.bias = get_rng().randn(self.local_out_features)

    def __call__(self, x):
        if x.shape[0] == 0:
            return np.zeros((0, self.out_features_global), dtype=np.float32)

        local_out = np.dot(x, self.weight) + self.bias
        slices = mpi.allgather(local_out)
        return np.concatenate(slices, axis=1)


class ShardedExpert:
    """Expert whose weights are sharded along the hidden / output dim."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        with rng_context("expert"):
            self.fc1 = ShardedLinear(input_dim, hidden_dim)
            self.fc2 = ShardedLinear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)
        return self.fc2(hidden)


class MoE_TP:
    """Mixture-of-Experts with tensor-parallel experts.

    Every rank holds a slice of every expert. Routing is replicated. After
    each expert's forward pass, ranks need a collective to reassemble the
    full output of that expert before applying the gate.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.rank = mpi.Get_rank()
        self.world_size = mpi.Get_size()

        with rng_context("router"):
            self.router = Router(input_dim, num_experts)

        with rng_context("expert"):
            self.experts = [
                ShardedExpert(input_dim, hidden_dim, output_dim)
                for _ in range(num_experts)
            ]

        if self.rank == 0:
            print(
                f"[MoE_TP] world_size={self.world_size}, num_experts={num_experts}, topk={self.topk}"
            )

    def forward(self, x):
        """
        Args:
            x: `(batch_size, input_dim)` — replicated on every rank.

        Returns:
            `(batch_size, output_dim)` — replicated on every rank.
        """
        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.output_dim))

        indices, gates = self.router(x, self.topk)

        # Batched by (expert, slot) so each ShardedExpert call carries multiple
        # tokens. Safe because indices is identical on every rank, so every
        # rank enters the same expert calls in the same order.
        for e in range(self.num_experts):
            for k in range(self.topk):
                mask = indices[:, k] == e
                if not mask.any():
                    continue
                tokens = x[mask]
                gates_e = gates[mask, k]
                expert_out = self.experts[e](tokens)
                outputs[mask] += gates_e[:, None] * expert_out

        return outputs

    def __call__(self, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# Part 1.2 — Expert Parallel MoE.
# ---------------------------------------------------------------------------
class MoE_EP:
    """Mixture-of-Experts with expert-parallel experts.

    Each rank owns *exactly one* expert. After routing, tokens that have been
    assigned to expert `e` must be sent to the rank that owns expert `e`. The
    expert computes its forward pass on the tokens it received and the results
    are sent back to the originating ranks.

    The natural collective for this pattern is **all-to-all**: each rank
    builds `world_size` buckets (one per destination rank) and exchanges them.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts  # == world size
        self.topk = min(topk, self.num_experts)
        self.rank = mpi.Get_rank()
        self.world_size = mpi.Get_size()

        assert num_experts == self.world_size, (
            "MoE_EP assumes one expert per rank; got "
            f"num_experts={num_experts}, world_size={self.world_size}"
        )

        with rng_context("router"):
            self.router = Router(input_dim, self.num_experts)

        # Per-rank seed so each rank's expert is different (the point of EP).
        with rng_context("expert_with_rank"):
            self.expert = Expert(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: `(batch_size, input_dim)` — replicated on every rank.

        Returns:
            `(batch_size, output_dim)` — replicated on every rank.
        """
        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.output_dim))

        indices, gates = self.router(x, self.topk)

        for k in range(self.topk):
            # Send tokens to the rank that owns their assigned expert.
            send_buckets = [[] for _ in range(self.world_size)]
            for i in range(batch_size):
                dest = int(indices[i, k])
                send_buckets[dest].append(
                    (self.rank, i, float(gates[i, k]), x[i].copy())
                )
            recv_buckets = mpi.alltoall(send_buckets)

            incoming_vecs = []
            incoming_meta = []
            for bucket in recv_buckets:
                for orig_rank, orig_idx, gate, vec in bucket:
                    incoming_vecs.append(vec)
                    incoming_meta.append((orig_rank, orig_idx, gate))

            if incoming_vecs:
                local_batch = np.stack(incoming_vecs, axis=0)
                local_out = self.expert(local_batch)
                gates_arr = np.array([m[2] for m in incoming_meta], dtype=local_out.dtype)
                gated = local_out * gates_arr[:, None]
            else:
                gated = np.zeros((0, self.output_dim))

            # Send gated outputs back to the rank that originated each token.
            send_back = [[] for _ in range(self.world_size)]
            for j, (orig_rank, orig_idx, _) in enumerate(incoming_meta):
                send_back[orig_rank].append((orig_idx, gated[j]))
            recv_back = mpi.alltoall(send_back)

            for bucket in recv_back:
                for orig_idx, vec in bucket:
                    outputs[orig_idx] += vec

        return outputs

    def __call__(self, x):
        return self.forward(x)
