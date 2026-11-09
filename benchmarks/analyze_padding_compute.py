"""Attribute measured-route matrix FLOPs savings to gates versus PAD queries.

This is an ANALYTICAL counterfactual, not a model run. It preserves the observed
early-exit path, all visited classifiers/router projections, and full padded
FFN work. Only routed attention shapes change. Keeping all valid queries still
projects all padded K/V positions, exactly as the corrected query runtime does.
The batched backend pays B*Kmax work, including dummy query rows.

Run only after the primary summary exists, to avoid CPU/I/O competition with
timing. Outputs contain aggregate arithmetic only, never input text or labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("router_loop", "router_batched", "hdc_loop", "hdc_batched")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def attention_flops(batch, padded_length, query_length, hidden):
    """Q/K/V/output projections and QK/AV products; MAC equals two FLOPs."""
    if query_length == 0:
        return 0
    query_projection = 2*batch*query_length*hidden*hidden
    key_value_projections = 4*batch*padded_length*hidden*hidden
    query_key_and_attention_value = 4*batch*query_length*padded_length*hidden
    output_projection = 2*batch*query_length*hidden*hidden
    return query_projection + key_value_projections + query_key_and_attention_value + output_projection


def routed_attention_cost(counts, padded_length, hidden, batched):
    if batched:
        return attention_flops(len(counts), padded_length, max(counts), hidden)
    return sum(attention_flops(1, padded_length, kept, hidden) for kept in counts)


def counterfactual_route(route, valid_lengths, hidden, batched):
    """Return actual/all-valid/all-padded costs, holding all other work fixed."""
    t, b = route["padded_length"], len(valid_lengths)
    require(b == route["batch_size"] and all(0 <= valid <= t for valid in valid_lengths),
            "Route shape differs from input lengths")
    actual = int(route["matmul_flops"])
    all_valid, all_padded = actual, actual
    for layer in route["stage_b_layers"]:
        kept = layer["k_per_sample"]
        require(len(kept) == b and all(0 <= k <= valid for k, valid in zip(kept, valid_lengths)),
                "Kept queries exceed valid tokens")
        observed_attention = routed_attention_cost(kept, t, hidden, batched)
        valid_attention = routed_attention_cost(valid_lengths, t, hidden, batched)
        padded_attention = routed_attention_cost([t]*b, t, hidden, batched)
        if batched:
            require(layer["packed_query_length"] == max(kept)
                    and layer["attention_matmul_flops"] == observed_attention,
                    "Observed batched attention accounting differs")
        all_valid += valid_attention-observed_attention
        all_padded += padded_attention-observed_attention
    require(actual <= all_valid <= all_padded, "Counterfactual attention ordering is inconsistent")
    return actual, all_valid, all_padded


def analyze(run_dir):
    benchmark = run_dir/"benchmark"
    require((benchmark/"summary.json").exists(), "Primary timing must complete before padding analysis")
    manifest, summary = read_json(benchmark/"manifest.json"), read_json(benchmark/"summary.json")
    require(manifest["protocol_sha256"] == summary["protocol_sha256"], "Summary identity mismatch")
    config = manifest["config"]
    source = ROOT/config["source_run"]
    resolved = read_json(source/"config_resolved.json")
    model_config = read_json(Path(resolved["model_name"])/"config.json")
    hidden = model_config["hidden_size"]
    # Deliberately delayed import and CPU mmap. No model, accelerator, or inference.
    import torch
    torch.set_num_threads(1)
    data = torch.load(source/"data.pt", map_location="cpu", weights_only=False, mmap=True)
    lengths = data["test"]["attention_mask"].sum(-1).tolist()
    source_rows = {(row["variant"], row["batch_size"]): row for row in summary["rows"]}
    rows = []
    for name in VARIANTS:
        batched = name.endswith("_batched")
        backend = "query_batched" if batched else "query_loop"
        for batch in manifest["batch_sizes"]:
            repeat_costs = []
            for repeat in range(config["benchmark_repeats"]):
                stem = f"latency_{name}_b{batch}_r{repeat}"
                timing = read_json(benchmark/(stem+".json"))
                totals, executions = [0, 0, 0], 0
                with (benchmark/(stem+"_routes.jsonl")).open(encoding="utf-8") as stream:
                    for line in stream:
                        route = json.loads(line)
                        indices = route["sample_indices"]
                        require(route["backend"] == backend and len(indices) == batch, "Wrong route backend/batch")
                        costs = counterfactual_route(route, [lengths[i] for i in indices], hidden, batched)
                        uses = route["executions"]
                        require(isinstance(uses, int) and uses > 0, "Invalid timed group repetition count")
                        totals = [total+value*uses for total, value in zip(totals, costs)]
                        executions += len(indices)*uses
                require(executions == timing["sample_executions"], "Counterfactual execution weighting mismatch")
                per_sample = [total/executions for total in totals]
                require(math.isclose(per_sample[0], timing["timed_inputs_matmul_flops_per_sample"], rel_tol=1e-12),
                        "Actual route totals differ from timing record")
                repeat_costs.append(per_sample)
            actual, valid, padded = [statistics.mean(cost[i] for cost in repeat_costs) for i in range(3)]
            dense = source_rows[("dense", batch)]["timed_inputs_matmul_flops_per_sample"]
            recorded = source_rows[(name, batch)]["timed_inputs_matmul_flops_per_sample"]
            require(math.isclose(actual, recorded, rel_tol=1e-12), "Actual aggregate differs from summary")
            learned, padding, exit_net = valid-actual, padded-valid, dense-padded
            total = dense-actual
            require(math.isclose(learned+padding+exit_net, total, rel_tol=1e-12, abs_tol=1e-5),
                    "Attribution terms do not add to total savings")
            row = {"variant": name, "batch_size": batch,
                   "dense_matmul_gflops_per_text": dense/1e9,
                   "actual_matmul_gflops_per_text": actual/1e9,
                   "all_valid_queries_counterfactual_gflops_per_text": valid/1e9,
                   "all_padded_queries_same_exit_counterfactual_gflops_per_text": padded/1e9,
                   "total_saved_percent_of_dense": 100*total/dense,
                   "learned_gates_extra_saved_percent_of_dense": 100*learned/dense,
                   "pad_query_omission_saved_percent_of_dense": 100*padding/dense,
                   "early_exit_net_head_router_overhead_saved_percent_of_dense": 100*exit_net/dense,
                   "learned_gates_fraction_of_total_saved_percent": 100*learned/total if total > 0 else None,
                   "repeat_costs_identical": all(cost == repeat_costs[0] for cost in repeat_costs)}
            rows.append(row)
    return {"analysis": "matrix FLOPs attribution at recorded paths; analytical counterfactual only",
            "source_protocol_sha256": manifest["protocol_sha256"], "rows": rows,
            "definitions": {
                "all_valid_queries": "Force every non-PAD query to be kept in each actually visited routed layer; full padded K/V, FFN, heads and router projections retained.",
                "all_padded_queries": "Same visited layers/heads/routers but attention also computes PAD query rows, matching dense per-layer attention shapes.",
                "learned_gates_extra_saved": "All-valid-query counterfactual minus actual routed FLOPs; isolates arithmetic saved beyond simply omitting PAD query rows.",
                "pad_query_omission_saved": "All-padded-query counterfactual minus all-valid-query counterfactual; only PAD query/output work, not PAD K/V or FFN.",
                "early_exit_net_head_router_overhead_saved": "Dense full-model FLOPs minus same-exit all-padded-query counterfactual. Includes early-exit savings net of added off-ramp/router work; can be negative for standalone routing.",
                "denominator": "All three component percentages use full Dense FLOPs as denominator and sum to total savings. Gate fraction of total uses total saved FLOPs instead.",
                "batch_padding": "Batched costs use B times maximum kept/valid query count and include dummy rows, including empty-query samples when other samples retain queries."},
            "limitations": ["No counterfactual model inference was run; these values describe matrix shapes only.",
                "Observed early-exit paths are held fixed. No counterfactual prediction, accuracy or latency is asserted.",
                "This decomposition measures this implementation's operations, not the best achievable padding-aware dense baseline.",
                "Softmax, LayerNorm, gather/scatter, control flow and memory movement remain excluded, as in the primary FLOPs protocol."]}


def self_test():
    # B=2,T=6,H=16; valid=[5,3], learned keep=[2,0]. Independent totals:
    # per-sample loop: actual 8,960; all-valid 23,552; all-padded 29,184.
    # padded batched: actual 17,920; all-valid 26,368; all-padded 29,184.
    for batched, expected in ((False, (8960, 23552, 29184)), (True, (17920, 26368, 29184))):
        observed = routed_attention_cost([2, 0], 6, 16, batched)
        entry = {"k_per_sample": [2, 0], "packed_query_length": 2, "attention_matmul_flops": observed}
        base = 24960  # Two padded FFN projections (24,576) + router projection (384).
        route = {"batch_size": 2, "padded_length": 6, "matmul_flops": base+observed,
                 "stage_b_layers": [entry]}
        result = counterfactual_route(route, [5, 3], 16, batched)
        require(tuple(value-base for value in result) == expected, "Independent attention attribution self-test failed")
        early = {**route, "matmul_flops": 1234, "stage_b_layers": []}
        require(counterfactual_route(early, [5, 3], 16, batched) == (1234, 1234, 1234), "Early-exit counterfactual changed work")
    print(json.dumps({"self_test": "passed"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"outputs/comparator_audit_20261109")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = analyze(args.run_dir.resolve())
    output = args.output or args.run_dir/"padding_compute_analysis.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"completed": True, "rows": len(result["rows"]),
                      "scope": "Analytical FLOPs attribution only; no latency/quality measurement"}))


if __name__ == "__main__":
    main()
