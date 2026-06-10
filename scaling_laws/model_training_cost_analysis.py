"""Model training cost analysis for Part 2.

You will implement three functions:

  - `model_training_cost_analysis_llama(config_path)`
  - `model_training_cost_analysis_deepseek(config_path)`
  - `get_optimal_N_D_from_cost(cost_budget)`

Run from the command line:

  python model_training_cost_analysis.py --model_config llama3_8b_config.json
  python model_training_cost_analysis.py --model_config deepseek_v3_config.json
  python model_training_cost_analysis.py --training_budget 5000000
"""
import argparse
import json
import math


def model_training_cost_analysis_llama(model_config_path):
    """Analyze training cost of a dense Llama-style model.

    Returns:
        total_params:   total trainable parameter count (int)
        flops_layer_TF: forward FLOPs of a single transformer layer (TFLOPs)
        peak_memory_GB: peak forward memory of a single transformer layer (GB)

    See the Part 2.1 writeup for the sequence-length / batch convention.
    """
    with open(model_config_path, "r") as f:
        cfg = json.load(f)

    d    = cfg["hidden_size"]            # 4096
    i    = cfg["intermediate_size"]      # 14336
    L    = cfg["num_hidden_layers"]      # 32
    V    = cfg["vocab_size"]             # 128256
    n_q  = cfg["num_attention_heads"]    # 32
    n_kv = cfg["num_key_value_heads"]    # 8
    head_dim = d // n_q                  # 128
    
    embed = V*d
    
    q = d * (n_q * head_dim)
    k = d * (n_kv * head_dim)
    v = d * (n_kv * head_dim)
    o = (n_q * head_dim) * d
    attn = q + k + v + o
    mlp   = 3 * (d * i)              # SiGLU MLP     
    norms = 2 * d
    per_layer = attn + mlp + norms

    final_norm = d
    lm_head    = V * d                

    total_params = embed + L * per_layer + final_norm + lm_head
    
    s = cfg["max_position_embeddings"]   # 8192

    attn_proj_flops = 2*s*d*d + 2*2*s*d*(n_kv * head_dim) + 2*s*d*d
    attn_score_flops = 2 * n_q * s * s * head_dim          # Q @ K^T
    attn_value_flops = 2 * n_q * s * s * head_dim          # Attn @ V
    mlp_flops = 2 * (2*s*d*i) + 2*s*i*d                    # gate + up + down

    flops_total = attn_proj_flops + attn_score_flops + attn_value_flops + mlp_flops
    flops_layer_TF = flops_total / 1e12                    # convert to TFLOPs
    bytes_per_elem = 2   # bf16

    # All activations alive during the layer's forward pass:
    input_act    = s * d                          # input x
    q_act        = s * (n_q  * head_dim)          # = s * d
    k_act        = s * (n_kv * head_dim)
    v_act        = s * (n_kv * head_dim)
    attn_scores  = n_q * s * s                    # (n_q, s, s) — THE big one
    attn_out     = s * d
    mlp_gate     = s * i
    mlp_up       = s * i
    mlp_hidden   = s * i                          # gate * up
    mlp_out      = s * d
    total_elems = (input_act + q_act + k_act + v_act + attn_scores
               + attn_out + mlp_gate + mlp_up + mlp_hidden + mlp_out)

    peak_memory_GB = total_elems * bytes_per_elem / (1024 ** 3)
    
    return total_params, flops_layer_TF, peak_memory_GB


def model_training_cost_analysis_deepseek(model_config_path):
    """Analyze training cost of a DeepSeek-V3-style MoE model with MLA attention
    and a mixed dense+MoE layer stack. Same return signature as the Llama version.
    """
    with open(model_config_path, "r") as f:
        cfg = json.load(f)

    d        = cfg["hidden_size"]
    L        = cfg["num_hidden_layers"]
    V        = cfg["vocab_size"]
    L_dense  = cfg["first_k_dense_replace"]
    L_moe    = L - L_dense

    n_h     = cfg["num_attention_heads"]
    q_lora  = cfg["q_lora_rank"]
    kv_lora = cfg["kv_lora_rank"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    v_head  = cfg["v_head_dim"]
    qk_head = qk_nope + qk_rope

    i_dense  = cfg["intermediate_size"]
    i_moe    = cfg["moe_intermediate_size"]
    n_routed = cfg["n_routed_experts"]
    n_shared = cfg["n_shared_experts"]
    topk     = cfg["num_experts_per_tok"]

    # MLA: low-rank Q (d→q_lora→n_h·qk_head) and joint low-rank KV
    # (d→kv_lora+qk_rope, then up to per-head K nope and V).
    W_DQ  = d * q_lora
    W_UQ  = q_lora * (n_h * qk_head)
    W_DKV = d * (kv_lora + qk_rope)
    W_UK  = kv_lora * (n_h * qk_nope)
    W_UV  = kv_lora * (n_h * v_head)
    W_O   = (n_h * v_head) * d
    mla_norms = q_lora + kv_lora
    attn_params = W_DQ + W_UQ + W_DKV + W_UK + W_UV + W_O + mla_norms

    def swiglu_params(inner):
        return 3 * d * inner

    dense_mlp_params = swiglu_params(i_dense)
    router_params    = d * n_routed
    moe_block_params = (router_params
                        + n_routed * swiglu_params(i_moe)
                        + n_shared * swiglu_params(i_moe))
    layer_norms = 2 * d

    dense_layer_params = attn_params + dense_mlp_params + layer_norms
    moe_layer_params   = attn_params + moe_block_params + layer_norms

    embed     = V * d
    lm_head   = V * d
    final_norm = d

    total_params = (embed
                    + L_dense * dense_layer_params
                    + L_moe   * moe_layer_params
                    + final_norm + lm_head)

    # FLOPs for one MoE layer (representative — 58 of 61 layers).
    s = cfg["max_position_embeddings"]

    attn_proj_flops = (2*s*d*q_lora
                       + 2*s*q_lora*(n_h*qk_head)
                       + 2*s*d*(kv_lora + qk_rope)
                       + 2*s*kv_lora*(n_h*qk_nope)
                       + 2*s*kv_lora*(n_h*v_head)
                       + 2*s*(n_h*v_head)*d)
    attn_score_flops = 2 * n_h * s * s * qk_head
    attn_value_flops = 2 * n_h * s * s * v_head
    router_flops     = 2 * s * d * n_routed
    experts_per_token = topk + n_shared
    expert_flops     = experts_per_token * s * 6 * d * i_moe

    flops_total = (attn_proj_flops + attn_score_flops + attn_value_flops
                   + router_flops + expert_flops)
    flops_layer_TF = flops_total / 1e12

    bytes_per = 2  # bf16

    input_act     = s * d
    q_compressed  = s * q_lora
    q_full        = s * (n_h * qk_head)
    kv_compressed = s * (kv_lora + qk_rope)
    k_full        = s * (n_h * qk_nope)
    v_full        = s * (n_h * v_head)
    attn_scores   = n_h * s * s
    attn_out      = s * (n_h * v_head)
    residual      = s * d
    moe_hidden    = s * experts_per_token * i_moe
    moe_out       = s * d

    total_elems = (input_act + q_compressed + q_full
                   + kv_compressed + k_full + v_full
                   + attn_scores + attn_out + residual
                   + moe_hidden + moe_out)
    peak_memory_GB = total_elems * bytes_per / (1024 ** 3)

    return total_params, flops_layer_TF, peak_memory_GB


def get_optimal_N_D_from_cost(cost_budget):
    """Pick the GPU and (N, D) that minimize loss under a $ training budget.

    cost_budget: a monetary training budget (in dollars)
    Returns:
        N: optimal model parameter count (absolute number)
        D: optimal training token count (absolute number)
        training_budget_flops: effective total training FLOPs
        best_gpu: name of the selected GPU, one of {'H100', 'H200', 'B200'}

    See the Part 2.2 writeup for the scaling law, the GPU price / TFLOPs
    table, and the MFU assumption.
    """
    GPUS = {
        "H100": {"price": 3.0, "tflops": 989},
        "H200": {"price": 4.0, "tflops": 989},
        "B200": {"price": 6.0, "tflops": 2250},
    }
    MFU = 0.40
    SECONDS_PER_HOUR = 3600

    # --- Sub-problem 1
    best_gpu = None
    best_flops = 0
    for name, spec in GPUS.items():
        hours = cost_budget / spec["price"]
        effective_flops = hours * SECONDS_PER_HOUR * spec["tflops"] * 1e12 * MFU
        if effective_flops > best_flops:
            best_flops = effective_flops
            best_gpu = name

    training_budget_flops = best_flops      

    # --- Sub-problem 2
    def loss(N, D):
        return 406.4 / (N ** 0.34) + 410.7 / (D ** 0.29) + 1.69

    best_N, best_D, best_L = None, None, float("inf")
    N = 1e6
    while N < 1e12:
        D = training_budget_flops / (6.0 * N)
        L = loss(N, D)
        if L < best_L:
            best_L, best_N, best_D = L, N, D
        N *= 1.01   # 1% increments

    return best_N, best_D, training_budget_flops, best_gpu



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training cost analysis")
    parser.add_argument("--model_config", type=str, help="Path to model config")
    parser.add_argument("--training_budget", type=float, default=None,
                        help="Training budget in dollars")
    args = parser.parse_args()

    if args.model_config:
        if "deepseek" in args.model_config:
            num_parameters, num_flops, memory_cost = (
                model_training_cost_analysis_deepseek(args.model_config)
            )
        elif "llama" in args.model_config:
            num_parameters, num_flops, memory_cost = (
                model_training_cost_analysis_llama(args.model_config)
            )
        else:
            print("Unknown model type — name your config llama*.json or deepseek*.json")
            raise SystemExit(1)
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(
            args.training_budget
        )
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")
