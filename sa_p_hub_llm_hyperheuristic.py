#!/usr/bin/env python3
"""
CAB-ready LLM-Controlled Hyper-Heuristic for the Uncapacitated
Single-Allocation p-Hub Median / p-Hub Location Problem (USApHMP / SA-p-HLP).

Key changes in this version:
  - No random sample data are generated automatically.
  - Phi-3/Ollama output is requested in minimal JSON and incomplete JSON can be recovered from w and r.
  - The program reads CSV inputs from a user-provided dataset folder.
  - Optional distances.csv is supported, so CAB benchmark distances can be used
    directly instead of approximating them from coordinates.
  - CAB-compatible flow normalization is supported through --normalize-flows.
  - Final article output table includes Reference Optimum, Greedy, Random, and an independent LLM-HH+Evaluator method.
  - In the final table, Greedy, Random, and LLM-HH are all run with the same number of iterations/evaluations; default = 20.
  - LLM-HH includes duplicate-control: if a recipe repeats a hub set, Python increases RCL and applies a soft anti-repeat penalty before recording the iteration.

Expected folder structure for a dataset, for example CAB_instances/CAB25/:
  nodes.csv            columns: node_id,x,y[,name]
  flows.csv            columns: origin,destination,flow
  distances.csv        columns: origin,destination,distance        optional but recommended for CAB
  candidate_hubs.csv   columns: hub_id OR node_id                  optional

CAB convention used by the generated files accompanying this script:
  - flows.csv contains normalized CAB OD flows, i.e., sum(flow)=1.
  - flows_raw.csv contains the original CAB flow matrix values.
  - distances.csv contains CAB distances scaled by 1/10000.
  - With collection_factor=1, distribution_factor=1 and transfer_factor=alpha,
    the objective values are on the same scale as the common CAB literature tables.

Example runs:
  # Interactive mode: you enter CAB size, number of hubs p, and alpha.
  python single_llm_p_hub_solver_CAB_ollama_phi3.py --interactive

  # Command-line mode using only the main inputs requested in the paper experiments.
  python single_llm_p_hub_solver_CAB_ollama_phi3.py --n 25 --p 3 --alpha 0.8 \
      --known-optima CAB_reference_optima.csv

  python single_llm_p_hub_solver_CAB_ollama_phi3.py --n 20 --p 4 --alpha 0.6 \
      --no-llm --compare-baselines

For local Ollama LLM mode with a GGUF model, import your model once, e.g.:
  ollama create phi3-cab -f Modelfile.phi3-cab
  python single_llm_p_hub_solver_CAB_ollama_phi3.py --n 25 --p 3 --alpha 0.8 \
      --llm-provider ollama --model phi3-cab-v2

For Gemini mode, install:
  pip install pandas numpy google-genai
and set GEMINI_API_KEY or GOOGLE_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# ============================================================
# USER SETTINGS
# ============================================================
# Keep this empty for safety. Put your key in GEMINI_API_KEY or GOOGLE_API_KEY.
API_KEY = ""
DEFAULT_P = 3
BASE_DIR = Path(__file__).resolve().parent
VALID_CAB_SIZES = [5, 10, 15, 20, 25]


def get_api_key() -> str:
    """Return API key from this file first, then environment variables."""
    return (API_KEY.strip() or os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip())


@dataclass
class PHubInstance:
    nodes: List[str]
    candidates: List[str]
    coords: Dict[str, Tuple[float, float]]
    flows: Dict[Tuple[str, str], float]
    dist: Dict[Tuple[str, str], float]
    p: int
    collection_factor: float = 1.0
    transfer_factor: float = 0.75
    distribution_factor: float = 1.0
    name: str = "instance"
    node_index: Optional[Dict[str, int]] = None
    flow_matrix: Optional[np.ndarray] = None
    dist_matrix: Optional[np.ndarray] = None
    out_flow: Optional[np.ndarray] = None
    in_flow: Optional[np.ndarray] = None


@dataclass
class Solution:
    hubs: List[str]
    allocation: Dict[str, str]
    objective: float = math.inf
    source: str = ""
    explanation: str = ""


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _read_candidates(candidates_path: Optional[str], nodes: List[str]) -> List[str]:
    if candidates_path and Path(candidates_path).exists():
        cand_df = pd.read_csv(candidates_path)
        if "hub_id" in cand_df.columns:
            candidates = cand_df["hub_id"].astype(str).tolist()
        elif "node_id" in cand_df.columns:
            candidates = cand_df["node_id"].astype(str).tolist()
        else:
            raise ValueError("candidate_hubs.csv must contain hub_id or node_id column")
    else:
        candidates = nodes[:]
    return candidates


def _read_distances(
    nodes: List[str],
    coords: Dict[str, Tuple[float, float]],
    distances_path: Optional[str],
    distance_multiplier: float,
) -> Dict[Tuple[str, str], float]:
    """Read a long-format distance table if available; otherwise use Euclidean coordinates."""
    dist: Dict[Tuple[str, str], float] = {}
    if distances_path and Path(distances_path).exists():
        ddf = pd.read_csv(distances_path)
        required = {"origin", "destination", "distance"}
        if not required.issubset(ddf.columns):
            raise ValueError(f"distances.csv must contain columns {required}")
        ddf["origin"] = ddf["origin"].astype(str)
        ddf["destination"] = ddf["destination"].astype(str)
        for r in ddf.itertuples(index=False):
            i = str(r.origin)
            j = str(r.destination)
            if i in nodes and j in nodes:
                dist[(i, j)] = float(r.distance) * distance_multiplier

    # Fill any missing distances from coordinates. This keeps the solver robust while
    # still giving priority to benchmark distance matrices such as CAB distances.
    for i in nodes:
        for j in nodes:
            if (i, j) not in dist:
                dist[(i, j)] = euclidean(coords[i], coords[j]) * distance_multiplier
    return dist


def load_instance(
    nodes_path: str,
    flows_path: str,
    candidates_path: Optional[str],
    distances_path: Optional[str],
    p: int,
    collection_factor: float,
    transfer_factor: float,
    distribution_factor: float,
    normalize_flows: bool = False,
    distance_multiplier: float = 1.0,
    instance_name: str = "instance",
) -> PHubInstance:
    nodes_file = Path(nodes_path)
    flows_file = Path(flows_path)
    if not nodes_file.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_file}")
    if not flows_file.exists():
        raise FileNotFoundError(f"Missing flows file: {flows_file}")

    nodes_df = pd.read_csv(nodes_file)
    flows_df = pd.read_csv(flows_file)

    required_nodes = {"node_id", "x", "y"}
    required_flows = {"origin", "destination", "flow"}
    if not required_nodes.issubset(nodes_df.columns):
        raise ValueError(f"nodes.csv must contain columns {required_nodes}")
    if not required_flows.issubset(flows_df.columns):
        raise ValueError(f"flows.csv must contain columns {required_flows}")

    nodes_df["node_id"] = nodes_df["node_id"].astype(str)
    flows_df["origin"] = flows_df["origin"].astype(str)
    flows_df["destination"] = flows_df["destination"].astype(str)

    nodes = nodes_df["node_id"].tolist()
    coords = {r.node_id: (float(r.x), float(r.y)) for r in nodes_df.itertuples(index=False)}
    candidates = _read_candidates(candidates_path, nodes)

    unknown = [h for h in candidates if h not in coords]
    if unknown:
        raise ValueError(f"Candidate hubs not found among nodes: {unknown}")
    if p <= 0 or p > len(candidates):
        raise ValueError("p must be positive and not larger than the number of candidate hubs")

    flows: Dict[Tuple[str, str], float] = {}
    for r in flows_df.itertuples(index=False):
        i = str(r.origin)
        j = str(r.destination)
        if i not in coords or j not in coords:
            continue
        val = float(r.flow)
        if val > 0:
            flows[(i, j)] = flows.get((i, j), 0.0) + val

    if normalize_flows:
        total_flow = sum(flows.values())
        if total_flow <= 0:
            raise ValueError("Cannot normalize flows because total positive flow is zero")
        flows = {key: value / total_flow for key, value in flows.items()}

    dist = _read_distances(nodes, coords, distances_path, distance_multiplier)

    node_index = {node: idx for idx, node in enumerate(nodes)}
    flow_matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    dist_matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    for (i, j), val in flows.items():
        flow_matrix[node_index[i], node_index[j]] = val
    for i in nodes:
        for j in nodes:
            dist_matrix[node_index[i], node_index[j]] = dist[(i, j)]

    return PHubInstance(
        nodes=nodes,
        candidates=candidates,
        coords=coords,
        flows=flows,
        dist=dist,
        p=p,
        collection_factor=collection_factor,
        transfer_factor=transfer_factor,
        distribution_factor=distribution_factor,
        name=instance_name,
        node_index=node_index,
        flow_matrix=flow_matrix,
        dist_matrix=dist_matrix,
        out_flow=flow_matrix.sum(axis=1),
        in_flow=flow_matrix.sum(axis=0),
    )


def is_feasible_hubs(instance: PHubInstance, hubs: List[str]) -> bool:
    return len(hubs) == instance.p and len(set(hubs)) == instance.p and all(h in instance.candidates for h in hubs)


def nearest_allocation(instance: PHubInstance, hubs: List[str]) -> Dict[str, str]:
    return {i: min(hubs, key=lambda h: instance.dist[(i, h)]) for i in instance.nodes}


def repair_allocation(instance: PHubInstance, hubs: List[str], allocation: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Force a complete single allocation to selected hubs only."""
    allocation = allocation or {}
    repaired = {}
    for i in instance.nodes:
        h = allocation.get(i)
        repaired[i] = h if h in hubs else min(hubs, key=lambda hub: instance.dist[(i, hub)])
    return repaired


def objective(instance: PHubInstance, hubs: List[str], allocation: Dict[str, str]) -> float:
    """Evaluate the USApHMP / SA-p-HLP objective. Lower is better.

    The implementation uses vectorized matrices when available, which is much
    faster for CAB benchmark comparisons.
    """
    if not is_feasible_hubs(instance, hubs):
        return math.inf
    for i in instance.nodes:
        if allocation.get(i) not in hubs:
            return math.inf

    if instance.node_index is not None and instance.flow_matrix is not None and instance.dist_matrix is not None:
        idx = instance.node_index
        try:
            assigned = np.array([idx[allocation[node]] for node in instance.nodes], dtype=int)
        except KeyError:
            return math.inf
        D = instance.dist_matrix
        W = instance.flow_matrix
        out_flow = instance.out_flow if instance.out_flow is not None else W.sum(axis=1)
        in_flow = instance.in_flow if instance.in_flow is not None else W.sum(axis=0)
        node_range = np.arange(len(instance.nodes))
        collection = float(np.dot(out_flow, D[node_range, assigned]))
        distribution = float(np.dot(in_flow, D[assigned, node_range]))
        transfer = float((W * D[np.ix_(assigned, assigned)]).sum())
        return (
            instance.collection_factor * collection
            + instance.transfer_factor * transfer
            + instance.distribution_factor * distribution
        )

    c = instance.collection_factor
    a = instance.transfer_factor
    d = instance.distribution_factor
    total = 0.0
    for (i, j), wij in instance.flows.items():
        hi = allocation[i]
        hj = allocation[j]
        total += wij * (c * instance.dist[(i, hi)] + a * instance.dist[(hi, hj)] + d * instance.dist[(hj, j)])
    return float(total)


def improve_allocation(instance: PHubInstance, hubs: List[str], allocation: Dict[str, str], max_passes: int = 8) -> Dict[str, str]:
    """Local reassignment of nodes for a fixed hub set."""
    allocation = repair_allocation(instance, hubs, allocation)
    best_obj = objective(instance, hubs, allocation)
    for _ in range(max_passes):
        improved = False
        for node in instance.nodes:
            current = allocation[node]
            best_h = current
            local_best = best_obj
            for h in hubs:
                if h == current:
                    continue
                trial = dict(allocation)
                trial[node] = h
                val = objective(instance, hubs, trial)
                if val + 1e-9 < local_best:
                    local_best = val
                    best_h = h
            if best_h != current:
                allocation[node] = best_h
                best_obj = local_best
                improved = True
        if not improved:
            break
    return allocation


def canonical_solution(instance: PHubInstance, hubs: List[str], allocation: Optional[Dict[str, str]] = None, source: str = "") -> Solution:
    hubs = list(dict.fromkeys([str(h) for h in hubs]))
    if not is_feasible_hubs(instance, hubs):
        return Solution(hubs=hubs, allocation={}, objective=math.inf, source=source)
    alloc = repair_allocation(instance, hubs, allocation)
    alloc = improve_allocation(instance, hubs, alloc)
    return Solution(hubs=hubs, allocation=alloc, objective=objective(instance, hubs, alloc), source=source)


def objective_for_partial_hub_set(instance: PHubInstance, hubs: List[str]) -> float:
    """Evaluate partial hub sets during constructive heuristics using nearest allocation."""
    if not hubs or len(set(hubs)) != len(hubs) or any(h not in instance.candidates for h in hubs):
        return math.inf
    allocation = nearest_allocation(instance, hubs)

    if instance.node_index is not None and instance.flow_matrix is not None and instance.dist_matrix is not None:
        idx = instance.node_index
        assigned = np.array([idx[allocation[node]] for node in instance.nodes], dtype=int)
        D = instance.dist_matrix
        W = instance.flow_matrix
        out_flow = instance.out_flow if instance.out_flow is not None else W.sum(axis=1)
        in_flow = instance.in_flow if instance.in_flow is not None else W.sum(axis=0)
        node_range = np.arange(len(instance.nodes))
        collection = float(np.dot(out_flow, D[node_range, assigned]))
        distribution = float(np.dot(in_flow, D[assigned, node_range]))
        transfer = float((W * D[np.ix_(assigned, assigned)]).sum())
        return (
            instance.collection_factor * collection
            + instance.transfer_factor * transfer
            + instance.distribution_factor * distribution
        )

    c = instance.collection_factor
    a = instance.transfer_factor
    d = instance.distribution_factor
    total = 0.0
    for (i, j), wij in instance.flows.items():
        hi = allocation[i]
        hj = allocation[j]
        total += wij * (c * instance.dist[(i, hi)] + a * instance.dist[(hi, hj)] + d * instance.dist[(hj, j)])
    return float(total)


def greedy_initial_solution(instance: PHubInstance) -> Solution:
    selected: List[str] = []
    remaining = instance.candidates[:]
    for _ in range(instance.p):
        best = min(remaining, key=lambda h: objective_for_partial_hub_set(instance, selected + [h]))
        selected.append(best)
        remaining.remove(best)
    return canonical_solution(instance, selected, None, source="greedy")


def greedy_repeated_search(instance: PHubInstance, iterations: int = 20) -> Solution:
    """Run the deterministic Greedy baseline a fixed number of times.

    Greedy itself is deterministic, so all repetitions normally return the same
    hub set. The loop is kept deliberately so the reported Greedy baseline has
    the same evaluation budget as Random and LLM-HH in the final article table.
    """
    best = Solution([], {}, math.inf, source="greedy")
    reps = max(1, int(iterations))
    for _ in range(reps):
        sol = greedy_initial_solution(instance)
        if sol.objective < best.objective:
            best = sol
    best.source = f"greedy; iterations={reps}; deterministic repeated baseline"
    return best


def single_swap_local_search(instance: PHubInstance, start: Solution) -> Tuple[Solution, List[Tuple[str, str]]]:
    best = canonical_solution(instance, start.hubs, start.allocation, source=start.source + "+swap")
    rejected_moves: List[Tuple[str, str]] = []
    improved = True
    while improved:
        improved = False
        best_neighbor = best
        best_move = None
        for old in best.hubs:
            for new in instance.candidates:
                if new in best.hubs:
                    continue
                hubs = [new if h == old else h for h in best.hubs]
                sol = canonical_solution(instance, hubs, None, source="swap")
                if sol.objective + 1e-9 < best_neighbor.objective:
                    best_neighbor = sol
                    best_move = (old, new)
                else:
                    rejected_moves.append((old, new))
        if best_move is not None:
            best = best_neighbor
            improved = True
    best.source = start.source + "+swap"
    return best, rejected_moves


def random_multistart_search(instance: PHubInstance, starts: int = 20, seed: int = 42) -> Solution:
    """Random multi-start baseline without hub-swap local search.

    The method samples p candidate hubs randomly and evaluates each hub set
    using the deterministic Python evaluator and allocation repair. It does
    not perform any hub exchange move, so the article column represents a
    pure Random baseline.
    """
    rng = random.Random(seed)
    starts = max(1, int(starts))
    best = Solution([], {}, math.inf, source="random")
    for _ in range(starts):
        hubs = rng.sample(instance.candidates, instance.p)
        sol = canonical_solution(instance, hubs, None, source="random")
        if sol.objective < best.objective:
            best = sol
    best.source = f"random; iterations={starts}; independent random hub sets"
    return best

def grasp_construction(instance: PHubInstance, rng: random.Random, rcl_size: int = 3) -> List[str]:
    selected: List[str] = []
    remaining = instance.candidates[:]
    for _ in range(instance.p):
        ranked = sorted(remaining, key=lambda h: objective_for_partial_hub_set(instance, selected + [h]))
        rcl = ranked[: max(1, min(rcl_size, len(ranked)))]
        chosen = rng.choice(rcl)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def grasp_search(instance: PHubInstance, iterations: int = 50, rcl_size: int = 3, seed: int = 42) -> Solution:
    rng = random.Random(seed)
    best = Solution([], {}, math.inf, source="grasp")
    for _ in range(max(1, iterations)):
        hubs = grasp_construction(instance, rng, rcl_size=rcl_size)
        sol, _ = single_swap_local_search(instance, canonical_solution(instance, hubs, None, source="grasp"))
        if sol.objective < best.objective:
            best = sol
            best.source = "grasp+swap"
    return best


def simulated_annealing_search(
    instance: PHubInstance,
    iterations: int = 500,
    start_temp: float = 1.0,
    cooling: float = 0.995,
    seed: int = 42,
) -> Solution:
    rng = random.Random(seed)
    current = canonical_solution(instance, rng.sample(instance.candidates, instance.p), None, source="sa")
    best = current
    temp = max(start_temp, 1e-9)
    for _ in range(max(1, iterations)):
        old = rng.choice(current.hubs)
        possible = [h for h in instance.candidates if h not in current.hubs]
        if not possible:
            break
        new = rng.choice(possible)
        hubs = [new if h == old else h for h in current.hubs]
        trial = canonical_solution(instance, hubs, None, source="sa")
        delta = trial.objective - current.objective
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            current = trial
            if current.objective < best.objective:
                best = current
        temp *= cooling
    best, _ = single_swap_local_search(instance, best)
    best.source = "simulated_annealing+swap"
    return best


def genetic_algorithm_search(
    instance: PHubInstance,
    population_size: int = 30,
    generations: int = 80,
    mutation_rate: float = 0.15,
    seed: int = 42,
) -> Solution:
    rng = random.Random(seed)
    cache: Dict[Tuple[str, ...], Solution] = {}

    def random_chromosome() -> Tuple[str, ...]:
        return tuple(sorted(rng.sample(instance.candidates, instance.p)))

    def evaluate(chrom: Tuple[str, ...]) -> Solution:
        if chrom not in cache:
            cache[chrom] = canonical_solution(instance, list(chrom), None, source="ga")
        return cache[chrom]

    def tournament(pop: List[Tuple[str, ...]], k: int = 3) -> Tuple[str, ...]:
        picks = rng.sample(pop, min(k, len(pop)))
        return min(picks, key=lambda ch: evaluate(ch).objective)

    def crossover(a: Tuple[str, ...], b: Tuple[str, ...]) -> Tuple[str, ...]:
        pool = list(dict.fromkeys(list(a[: len(a) // 2]) + list(b) + list(a)))
        while len(pool) < instance.p:
            h = rng.choice(instance.candidates)
            if h not in pool:
                pool.append(h)
        return tuple(sorted(pool[: instance.p]))

    def mutate(ch: Tuple[str, ...]) -> Tuple[str, ...]:
        hubs = list(ch)
        if rng.random() < mutation_rate:
            idx = rng.randrange(len(hubs))
            choices = [h for h in instance.candidates if h not in hubs]
            if choices:
                hubs[idx] = rng.choice(choices)
        return tuple(sorted(hubs))

    pop = [random_chromosome() for _ in range(max(population_size, 4))]
    best = min((evaluate(ch) for ch in pop), key=lambda s: s.objective)
    for _ in range(max(1, generations)):
        ranked = sorted(pop, key=lambda ch: evaluate(ch).objective)
        new_pop = ranked[:2]
        while len(new_pop) < len(pop):
            child = crossover(tournament(pop), tournament(pop))
            child = mutate(child)
            new_pop.append(child)
        pop = new_pop
        candidate = evaluate(pop[0])
        if candidate.objective < best.objective:
            best = candidate
    best, _ = single_swap_local_search(instance, best)
    best.source = "genetic_algorithm+swap"
    return best


def summarize_instance(instance: PHubInstance, max_nodes: int = 25, max_flows: int = 60) -> Dict[str, Any]:
    nodes_part = [
        {"node_id": n, "x": round(instance.coords[n][0], 3), "y": round(instance.coords[n][1], 3)}
        for n in instance.nodes[:max_nodes]
    ]
    flows_sorted = sorted(instance.flows.items(), key=lambda kv: kv[1], reverse=True)[:max_flows]
    flows_part = [{"origin": i, "destination": j, "flow": round(v, 6)} for (i, j), v in flows_sorted]
    return {
        "problem": "uncapacitated single-allocation p-hub median/location problem",
        "instance_name": instance.name,
        "number_of_nodes": len(instance.nodes),
        "p": instance.p,
        "cost_factors": {
            "collection": instance.collection_factor,
            "transfer": instance.transfer_factor,
            "distribution": instance.distribution_factor,
        },
        "candidate_hubs": instance.candidates,
        "nodes": nodes_part,
        "largest_flows": flows_part,
        "output_format": "JSON with selected hubs, complete single allocation, explanation, and swap moves",
    }


def build_prompt(instance: PHubInstance, memory: Dict[str, Any], temperature: float) -> str:
    """Ultra-compact TOON prompt for local Phi-3/Ollama.

    The model returns ONLY selected hub IDs. Python repairs allocation and
    evaluates the objective. This is intentionally tiny to avoid timeouts on CPU.
    """
    # Keep only a few high-importance node scores instead of sending many OD flows.
    scores = []
    for n in instance.nodes:
        idx = instance.node_index[n] if instance.node_index else instance.nodes.index(n)
        out_v = float(instance.out_flow[idx]) if instance.out_flow is not None else 0.0
        in_v = float(instance.in_flow[idx]) if instance.in_flow is not None else 0.0
        scores.append((n, out_v + in_v))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:min(8, len(scores))]
    score_lines = "\n".join(f"  {n},{v:.5g}" for n, v in scores) or "  none"

    best_rows = memory.get("best_solutions", [])[:3]
    best_lines = "\n".join(
        f"  {','.join(map(str, row.get('hubs', [])))},{row.get('objective', '')}"
        for row in best_rows
    ) or "  none"

    # JSON example must match p length as closely as possible.
    example_hubs = instance.candidates[:instance.p]
    example_json = json.dumps({"hubs": example_hubs}, ensure_ascii=False)

    return f"""
task: choose_hubs
n: {len(instance.nodes)}
p: {instance.p}
alpha: {instance.transfer_factor}
candidates: {','.join(instance.candidates)}
best[hubs,obj]:
{best_lines}
important_nodes[node,flow_score]:
{score_lines}
rules:
  - return JSON only
  - key: hubs
  - hubs length must be {instance.p}
  - choose only from candidates
example: {example_json}
answer:
""".strip()

def call_gemini(prompt: str, model: str, temperature: float, max_output_tokens: int = 2048) -> str:
    if genai is None or types is None:
        raise RuntimeError("google-genai is not installed. Install it with: pip install google-genai")
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        ),
    )
    return response.text or ""


def call_ollama(
    prompt: str,
    model: str,
    temperature: float,
    max_output_tokens: int = 64,
    ollama_url: str = "http://localhost:11434",
    timeout_seconds: int = 180,
    use_json_format: bool = False,
) -> str:
    """Call local Ollama with strict small output limits.

    Notes for Phi-3-mini on CPU:
    - Do not force Ollama JSON grammar by default; it can be slow.
    - Keep num_ctx and num_predict small.
    - Use stop tokens matching Phi-3 chat format.
    """
    import socket
    import urllib.error
    import urllib.request

    endpoint = ollama_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": float(temperature),
            "top_p": 0.9,
            "num_ctx": 768,
            "num_predict": int(max_output_tokens),
            "stop": ["<|end|>", "<|user|>", "<|system|>", "<|assistant|>"],
        },
    }
    if use_json_format:
        payload["format"] = "json"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise RuntimeError(f"timed out after {timeout_seconds}s") from exc
    except socket.timeout as exc:
        raise RuntimeError(f"timed out after {timeout_seconds}s") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(
            f"Could not get response from Ollama at {ollama_url}. Reason: {reason}. "
            "Make sure Ollama is running and the model name is correct."
        ) from exc

    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {data}")
    return str(data.get("response") or "")

def call_llm(
    prompt: str,
    provider: str,
    model: str,
    temperature: float,
    max_output_tokens: int = 64,
    ollama_url: str = "http://localhost:11434",
    ollama_timeout: int = 180,
    ollama_json_format: bool = False,
) -> str:
    provider = (provider or "ollama").strip().lower()
    if provider == "ollama":
        return call_ollama(
            prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            ollama_url=ollama_url,
            timeout_seconds=ollama_timeout,
            use_json_format=ollama_json_format,
        )
    if provider == "gemini":
        return call_gemini(prompt, model=model, temperature=temperature, max_output_tokens=max_output_tokens)
    raise ValueError(f"Unknown LLM provider: {provider}")

def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def parse_llm_solution(instance: PHubInstance, text: str) -> Tuple[Solution, List[Tuple[str, str]]]:
    data = extract_json(text)
    hubs = [str(h) for h in data.get("hubs", [])]
    raw_alloc = data.get("allocation") or {}
    allocation = {str(k): str(v) for k, v in raw_alloc.items()} if isinstance(raw_alloc, dict) else {}
    moves: List[Tuple[str, str]] = []
    for mv in data.get("swap_moves", []) or []:
        if isinstance(mv, dict):
            old = str(mv.get("remove", ""))
            new = str(mv.get("add", ""))
        elif isinstance(mv, (list, tuple)) and len(mv) == 2:
            old, new = str(mv[0]), str(mv[1])
        else:
            continue
        if old in instance.candidates and new in instance.candidates and old != new:
            moves.append((old, new))
    sol = canonical_solution(instance, hubs, allocation, source="LLM")
    sol.explanation = str(data.get("explanation", ""))
    return sol, moves


def solution_key(sol: Solution) -> Tuple[str, ...]:
    return tuple(sorted(sol.hubs))


def solve_with_llm(
    instance: PHubInstance,
    iterations: int,
    model: str,
    use_llm: bool,
    seed: int = 42,
    fallback_on_llm_failure: bool = True,
    llm_provider: str = "ollama",
    ollama_url: str = "http://localhost:11434",
    ollama_timeout: int = 180,
    llm_max_tokens: int = 80,
    print_prompt: bool = False,
    ollama_json_format: bool = False,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    temperature = 0.35 if str(llm_provider).lower() == "ollama" else 0.9
    no_improve = 0
    memory: Dict[str, Any] = {
        "best_solutions": [],
        "infeasible_solutions": [],
        "bad_moves": [],
        "note": "The objective is calculated only by Python, not by the LLM.",
        "llm_provider": llm_provider,
        "llm_model": model,
    }

    best, bad = single_swap_local_search(instance, greedy_initial_solution(instance))
    memory["best_solutions"].append({"hubs": best.hubs, "objective": round(best.objective, 6), "source": best.source})
    memory["bad_moves"].extend([{"remove": a, "add": b} for a, b in bad[:20]])
    seen = {solution_key(best)}

    for it in range(1, iterations + 1):
        print(f"\nIteration {it}/{iterations} | best = {best.objective:.6f} | hubs = {best.hubs}")
        candidate_solutions: List[Solution] = []
        llm_moves: List[Tuple[str, str]] = []

        if use_llm:
            try:
                prompt = build_prompt(instance, memory, temperature)
                if print_prompt:
                    print("\n--- LLM PROMPT START ---")
                    print(prompt)
                    print("--- LLM PROMPT END ---\n")
                raw = call_llm(prompt, provider=llm_provider, model=model, temperature=temperature, max_output_tokens=llm_max_tokens, ollama_url=ollama_url, ollama_timeout=ollama_timeout, ollama_json_format=ollama_json_format)
                print("LLM response:", raw[:500].replace("\n", " "), "...")
                sol, llm_moves = parse_llm_solution(instance, raw)
                if math.isfinite(sol.objective):
                    candidate_solutions.append(sol)
                else:
                    memory["infeasible_solutions"].append({"hubs": sol.hubs, "reason": "invalid hub set"})
            except Exception as exc:
                print(f"LLM call failed. Reason: {exc}")
                memory["llm_status"] = "failed"
                memory["llm_failure_reason"] = str(exc)
                if not fallback_on_llm_failure:
                    break
                print("Using heuristic variation because fallback_on_llm_failure=True.")
                use_llm = False

        if not use_llm:
            hubs = rng.sample(instance.candidates, instance.p)
            sol, bad = single_swap_local_search(instance, canonical_solution(instance, hubs, None, source="random"))
            candidate_solutions.append(sol)
            memory["bad_moves"].extend([{"remove": a, "add": b} for a, b in bad[:10]])

        for old, new in llm_moves:
            if old in best.hubs and new not in best.hubs:
                hubs = [new if h == old else h for h in best.hubs]
                candidate_solutions.append(canonical_solution(instance, hubs, None, source=f"LLM swap {old}->{new}"))
            else:
                memory["bad_moves"].append({"remove": old, "add": new})

        improved = False
        for sol in candidate_solutions:
            if not math.isfinite(sol.objective):
                memory["infeasible_solutions"].append({"hubs": sol.hubs, "reason": "infeasible objective"})
                continue
            key = solution_key(sol)
            if key in seen and sol.objective >= best.objective - 1e-9:
                continue
            seen.add(key)
            print(f"Candidate {sol.hubs} objective = {sol.objective:.6f} source = {sol.source}")
            if sol.objective + 1e-9 < best.objective:
                best = sol
                improved = True
                no_improve = 0
                memory["best_solutions"].insert(0, {"hubs": best.hubs, "objective": round(best.objective, 6), "source": best.source})
                memory["best_solutions"] = memory["best_solutions"][:10]
        if not improved:
            no_improve += 1
        if no_improve >= 3:
            temperature = min(0.8 if str(llm_provider).lower() == "ollama" else 1.35, temperature + 0.15)
            no_improve = 0
        else:
            temperature = max(0.45, temperature * 0.97)

    return {
        "instance": instance.name,
        "n": len(instance.nodes),
        "p": instance.p,
        "collection_factor": instance.collection_factor,
        "transfer_factor": instance.transfer_factor,
        "distribution_factor": instance.distribution_factor,
        "best_hubs": best.hubs,
        "allocation": best.allocation,
        "objective": best.objective,
        "source": best.source,
        "llm_status": memory.get("llm_status", "ok" if use_llm else "heuristic_fallback"),
        "memory": memory,
    }


def lookup_known_optimum(path: Optional[str], instance: PHubInstance) -> Optional[float]:
    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    if not {"n", "p", "alpha", "optimal_objective"}.issubset(df.columns):
        return None
    alpha = round(float(instance.transfer_factor), 10)
    rows = df[(df["n"].astype(int) == len(instance.nodes)) & (df["p"].astype(int) == instance.p)]
    rows = rows[np.isclose(rows["alpha"].astype(float), alpha, atol=1e-9)]
    if rows.empty:
        return None
    return float(rows.iloc[0]["optimal_objective"])


def lookup_known_optimum_details(path: Optional[str], instance: PHubInstance) -> Dict[str, Any]:
    """Return the reference optimum row for this instance/p/alpha, if available."""
    details: Dict[str, Any] = {
        "known_optimum": None,
        "reference": None,
        "reference_table": None,
        "objective_scale": None,
        "note": None,
        "status": "not_found",
    }
    if not path or not Path(path).exists():
        details["status"] = "reference_file_missing"
        return details
    df = pd.read_csv(path)
    required = {"n", "p", "alpha", "optimal_objective"}
    if not required.issubset(df.columns):
        details["status"] = "invalid_reference_file"
        return details
    alpha = round(float(instance.transfer_factor), 10)
    rows = df[(df["n"].astype(int) == len(instance.nodes)) & (df["p"].astype(int) == instance.p)]
    rows = rows[np.isclose(rows["alpha"].astype(float), alpha, atol=1e-9)]
    if rows.empty:
        return details
    row = rows.iloc[0]
    details["known_optimum"] = float(row["optimal_objective"])
    details["reference"] = str(row.get("reference", "")) if "reference" in rows.columns else None
    details["reference_table"] = str(row.get("reference_table", "")) if "reference_table" in rows.columns else None
    details["objective_scale"] = str(row.get("objective_scale", "")) if "objective_scale" in rows.columns else None
    details["note"] = str(row.get("note", "")) if "note" in rows.columns else None
    details["status"] = "available"
    return details


def _format_hubs(hubs: Optional[List[str]]) -> str:
    return "" if not hubs else ",".join(str(h) for h in hubs)


def _gap_percent(objective_value: Optional[float], known_optimum: Optional[float]) -> Optional[float]:
    if objective_value is None or known_optimum is None or not math.isfinite(float(objective_value)) or abs(float(known_optimum)) <= 1e-12:
        return None
    return 100.0 * (float(objective_value) - float(known_optimum)) / abs(float(known_optimum))


def _solution_from_result(result: Dict[str, Any], source: str = "LLM+Evaluator") -> Solution:
    return Solution(
        hubs=[str(h) for h in result.get("best_hubs", [])],
        allocation={str(k): str(v) for k, v in result.get("allocation", {}).items()},
        objective=float(result.get("objective", math.inf)),
        source=source,
    )



# ============================================================
# LLM-CONTROLLED HYPER-HEURISTIC
# ============================================================
# In this design, the LLM is NOT a solver and does NOT repeat the
# classical Greedy local search. The LLM controls/parameterizes a
# low-level constructive heuristic. Python constructs hubs, repairs
# allocation, checks feasibility, and evaluates the objective.


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) <= 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def candidate_metrics(instance: PHubInstance) -> Dict[str, Dict[str, float]]:
    """Compute normalized node metrics used by the low-level heuristic.

    These metrics are deterministic and transparent. They are sent to the
    LLM in compact TOON form, and Python uses them to instantiate the
    LLM-proposed scoring rule.
    """
    node_ids = instance.nodes
    D = instance.dist_matrix
    W = instance.flow_matrix
    idx = instance.node_index or {n: k for k, n in enumerate(node_ids)}
    if D is None:
        D = np.array([[instance.dist[(i, j)] for j in node_ids] for i in node_ids], dtype=float)
    if W is None:
        W = np.zeros((len(node_ids), len(node_ids)), dtype=float)
        for (i, j), v in instance.flows.items():
            W[idx[i], idx[j]] = float(v)

    out_flow = W.sum(axis=1)
    in_flow = W.sum(axis=0)
    flow_raw = (out_flow + in_flow).astype(float).tolist()
    mean_dist_raw = D.mean(axis=1).astype(float).tolist()
    flow_norm = _minmax(flow_raw)
    mean_dist_norm = _minmax(mean_dist_raw)
    # Higher centrality is better; lower mean distance is better.
    centrality_norm = [1.0 - x for x in mean_dist_norm]

    metrics: Dict[str, Dict[str, float]] = {}
    for k, n in enumerate(node_ids):
        metrics[n] = {
            "flow": float(flow_norm[k]),
            "centrality": float(centrality_norm[k]),
            "distance_penalty": float(mean_dist_norm[k]),
            "raw_flow": float(flow_raw[k]),
            "mean_distance": float(mean_dist_raw[k]),
        }
    return metrics


def build_hh_prompt(instance: PHubInstance, memory: Dict[str, Any], temperature: float) -> str:
    """Compact model-guided DSL prompt for Phi-3/Ollama hyper-heuristic.

    The LLM is explicitly asked to avoid repeated hub sets by designing a
    logically different weighting recipe. The output remains a short DSL so a
    small local Phi-3 model can follow it reliably:
        W=a,b,c,d,e;R=k;A=none
    where W are weights for [flow, centrality, dispersion, interaction,
    distance_penalty], R is the restricted candidate-list size, and A is an
    optional short avoid list for overused hubs in this iteration.
    """
    metrics = candidate_metrics(instance)
    rows = []
    for n in instance.candidates:
        m = metrics[n]
        rows.append((n, m["flow"], m["centrality"], m["distance_penalty"]))
    rows = sorted(rows, key=lambda r: r[1], reverse=True)[: min(8, len(rows))]
    metric_lines = ";".join(f"{n}:{f:.2f},{c:.2f},{d:.2f}" for n, f, c, d in rows)

    recent_sets = memory.get("recent_sets", [])[:6]
    seen_lines = ";".join(
        f"{','.join(x.get('hubs', []))}:{float(x.get('objective', 0)):.4g}"
        for x in recent_sets
        if x.get("objective") is not None
    ) or "none"

    best_rows = memory.get("valid_recipes", [])[:2]
    best_lines = ";".join(
        f"{','.join(x.get('hubs', []))}:{float(x.get('objective', 0)):.4g}"
        for x in best_rows
        if x.get("objective") is not None
    ) or "none"

    bad_sets = memory.get("bad_hub_sets", [])[:6]
    bad_lines = ";".join(",".join(map(str, x)) for x in bad_sets if x) or "none"

    # Overused hubs are shown to the model so it can decide whether to lower
    # exploitation weights or add an optional A=... avoid list. The model still
    # has to keep a logical weighting pattern; Python does not invent weights.
    hub_freq: Dict[str, int] = {}
    for row in recent_sets:
        for h in row.get("hubs", []) or []:
            hub_freq[str(h)] = hub_freq.get(str(h), 0) + 1
    overused = sorted(hub_freq.items(), key=lambda kv: (-kv[1], kv[0]))[: min(4, len(hub_freq))]
    overused_lines = ";".join(f"{h}:{c}" for h, c in overused) or "none"

    return (
        "Output only one line in this exact format:\n"
        "W=0.35,0.18,0.30,0.10,0.07;R=3;A=none\n"
        "No JSON. No explanation. Use only W, R, and A.\n"
        f"Problem SA-p-hub n={len(instance.nodes)} p={instance.p} alpha={instance.transfer_factor}.\n"
        "W order: flow,centrality,dispersion,interaction,distance_penalty.\n"
        "Logic rules: weights must be numeric, nonnegative, and sum about 1.0; do not output random weights.\n"
        "If previous sets repeat, avoid the same hub set by increasing dispersion and/or R, and optionally set A to overused hubs.\n"
        "Keep flow and centrality meaningful, but do not let them always select the same dominant hubs.\n"
        "Use higher dispersion when search is stuck; use distance_penalty to avoid peripheral hubs; use interaction for complementary flow hubs.\n"
        "R is 1..5. Use R=1 for exploitation only when no repetition exists; use R=2..5 for controlled exploration.\n"
        "A is optional avoid list from candidates, at most p hubs, or A=none. Do not avoid all strong hubs unless repetition requires it.\n"
        f"Candidates: {','.join(instance.candidates)}\n"
        f"Top features node:flow,center,distpen: {metric_lines}\n"
        f"Best LLM-only sets: {best_lines}\n"
        f"Recent evaluated LLM sets: {seen_lines}\n"
        f"Repeated or rejected sets: {bad_lines}\n"
        f"Overused hubs count: {overused_lines}\n"
        "Answer:"
    )

def _parse_hh_dsl(text: str) -> Dict[str, Any]:
    """Parse Phi-3 friendly one-line DSL: W=a,b,c,d,e;R=k.

    This parser is deliberately permissive: it accepts W=[...], w: [...],
    semicolon/comma separators, and an optional R/r value.
    """
    raw = str(text or "").strip()
    # Remove common fences and assistant prefixes.
    raw = raw.replace("```", " ").replace("\n", " ")
    # Prefer explicit W= or W: segment.
    m = re.search(r"\bW\s*[:=]\s*\[?\s*([^;\]\n]+(?:[, ]+[^;\]\n]+){0,8})", raw, flags=re.I)
    if not m:
        # Also accept JSON-like compact key: "w": [0.4, ...]
        m = re.search(r'"w"\s*:\s*\[([^\]]+)\]', raw, flags=re.I)
    if not m:
        raise ValueError("No W=... recipe found in LLM response")
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", m.group(1))
    if len(nums) < 5:
        raise ValueError("W recipe must contain five numeric weights")
    w = [float(x) for x in nums[:5]]

    mr = re.search(r"\bR\s*[:=]\s*\[?\s*(\d+)", raw, flags=re.I)
    if not mr:
        mr = re.search(r'"r"\s*:\s*(\d+)', raw, flags=re.I)
    r = int(mr.group(1)) if mr else 1

    # Optional model-guided avoid list: A=none or A=4,9
    avoid: List[str] = []
    ma = re.search(r"\bA\s*[:=]\s*([^;]+)", raw, flags=re.I)
    if ma:
        a_raw = ma.group(1).strip()
        if a_raw.lower() not in {"none", "null", "empty", "-"}:
            avoid = [x.strip() for x in re.split(r"[,\s]+", a_raw) if x.strip()]
    return {"w": w, "r": r, "a": avoid, "raw_text": raw[:500]}


def _coerce_json_like(text: str) -> Dict[str, Any]:
    """Extract a JSON object from imperfect small-model output.

    This is kept as a fallback for models that still return JSON. Nested objects
    such as {"SA_p_hub": {"w": [...], "r": 2}} are unwrapped later.
    """
    raw = str(text or "").strip()
    raw = raw.replace("```json", "```").replace("```JSON", "```")
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if "{" in part and "}" in part:
                raw = part.strip()
                break
    if "{" in raw and "}" in raw:
        raw = raw[raw.find("{") : raw.rfind("}") + 1]
    try:
        return json.loads(raw)
    except Exception:
        fixed = raw.replace("'", '"')
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        fixed = re.sub(r"\bNone\b", "null", fixed)
        fixed = re.sub(r"\bTrue\b", "true", fixed)
        fixed = re.sub(r"\bFalse\b", "false", fixed)
        return json.loads(fixed)


def _unwrap_hh_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap common Phi-3 nested JSON mistakes."""
    if not isinstance(data, dict):
        return {}
    if "SA_p_hub" in data and isinstance(data["SA_p_hub"], dict):
        return data["SA_p_hub"]
    if "recipe" in data and isinstance(data["recipe"], dict):
        return data["recipe"]
    return data


def parse_hh_recipe(text: str, instance: PHubInstance) -> Dict[str, Any]:
    """Parse LLM output as a controlled hyper-heuristic recipe.

    Preferred Phi-3 schema is not JSON:
        W=0.45,0.25,0.20,0.05,0.05;R=1

    JSON is accepted only as a fallback.
    """
    data: Dict[str, Any]
    try:
        data = _parse_hh_dsl(text)
    except Exception:
        data = _unwrap_hh_json(_coerce_json_like(text))

    recipe: Dict[str, Any] = {"raw": data}

    default_weights = {
        "flow": 0.40,
        "centrality": 0.20,
        "dispersion": 0.25,
        "interaction": 0.10,
        "distance_penalty": 0.05,
    }

    # Compact schema: w=[flow,center,dispersion,interaction,dist_penalty]
    weights: Dict[str, float] = {}
    if isinstance(data.get("w"), list):
        arr = data.get("w", [])
        # Only numeric arrays are accepted. A list like ["flow","center",...] is a schema echo, not a recipe.
        numeric = []
        for x in arr:
            try:
                numeric.append(float(x))
            except Exception:
                pass
        if len(numeric) >= 5:
            keys = ["flow", "centrality", "dispersion", "interaction", "distance_penalty"]
            weights = {k: numeric[i] for i, k in enumerate(keys)}
    elif isinstance(data.get("weights"), dict):
        weights = data.get("weights", {})

    if not weights:
        # Last chance: parse a W=... line from raw text, then map it.
        try:
            dsl = _parse_hh_dsl(str(text))
            arr = dsl["w"]
            keys = ["flow", "centrality", "dispersion", "interaction", "distance_penalty"]
            weights = {k: arr[i] for i, k in enumerate(keys)}
            data["r"] = dsl.get("r", data.get("r", 1))
        except Exception:
            raise ValueError("LLM did not return a numeric HH recipe")

    parsed_weights: Dict[str, float] = {}
    for key, default in default_weights.items():
        alt_key = "center" if key == "centrality" else key
        try:
            val = float(weights.get(key, weights.get(alt_key, default))) if isinstance(weights, dict) else float(default)
        except Exception:
            val = default
        parsed_weights[key] = max(0.0, min(1.5, val))
    recipe["weights"] = parsed_weights

    r_raw = data.get("r", data.get("rcl_size", 1))
    if isinstance(r_raw, list) and r_raw:
        r_raw = r_raw[0]
    try:
        recipe["rcl_size"] = int(r_raw)
    except Exception:
        recipe["rcl_size"] = 1
    recipe["rcl_size"] = max(1, min(5, int(recipe["rcl_size"])))

    recipe["must_include"] = []

    avoid_raw = data.get("a", data.get("avoid", []))
    if isinstance(avoid_raw, str):
        if avoid_raw.strip().lower() in {"none", "null", "empty", "-", ""}:
            avoid_list: List[str] = []
        else:
            avoid_list = [x.strip() for x in re.split(r"[,\s]+", avoid_raw) if x.strip()]
    elif isinstance(avoid_raw, list):
        avoid_list = [str(x).strip() for x in avoid_raw if str(x).strip()]
    else:
        avoid_list = []
    # The model may suggest avoiding overused hubs, but never allow it to
    # remove so many candidates that a feasible p-hub set cannot be built.
    avoid_list = [h for h in avoid_list if h in instance.candidates]
    max_avoid = max(0, len(instance.candidates) - instance.p)
    recipe["avoid"] = avoid_list[:max_avoid]

    recipe["note"] = ""
    return recipe


def build_hh_json_retry_prompt(raw_response: str, instance: PHubInstance) -> str:
    """Tiny retry prompt. Still uses the DSL, not JSON."""
    return (
        "Your previous answer was invalid. Output only this one-line format:\n"
        "W=0.35,0.18,0.30,0.10,0.07;R=3;A=none\n"
        "Use five numeric decimal weights that sum about 1.0, R from 1 to 5, and A=none or A=hub ids. No JSON. No words.\n"
        "Answer:"
    )


def build_hh_duplicate_retry_prompt(instance: PHubInstance, memory: Dict[str, Any], repeated_hubs: List[str]) -> str:
    """Ask the LLM itself to redesign the recipe after a repeated hub set."""
    base = build_hh_prompt(instance, memory, temperature=0.5)
    return (
        base
        + "\nYour last recipe produced an already evaluated hub set: "
        + ",".join(map(str, repeated_hubs))
        + ".\nDo not reproduce this set. Redesign W logically: increase dispersion/R or use A for overused hubs, while keeping weights meaningful and nonrandom.\nAnswer:"
    )

def _hub_set_diversity(hub_sets: List[List[str]]) -> float:
    """Average pairwise Jaccard distance among proposed hub sets."""
    if len(hub_sets) < 2:
        return 0.0
    vals = []
    sets = [set(h) for h in hub_sets]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            inter = sets[i] & sets[j]
            vals.append(1.0 - len(inter) / max(1, len(union)))
    return float(sum(vals) / len(vals)) if vals else 0.0



def _recipe_score_context(instance: PHubInstance) -> Tuple[Dict[str, Dict[str, float]], Any, Dict[str, int], float, Any, float]:
    """Precompute deterministic objects used by the LLM-HH scoring rule."""
    metrics = candidate_metrics(instance)
    D = instance.dist_matrix
    idx = instance.node_index or {n: k for k, n in enumerate(instance.nodes)}
    max_dist = 1.0
    if D is not None and len(instance.nodes) > 0:
        max_dist = float(np.max(D)) or 1.0
    F = instance.flow_matrix
    max_interaction = 1.0
    if F is not None:
        max_interaction = float(np.max(F.sum(axis=0) + F.sum(axis=1))) or 1.0
    return metrics, D, idx, max_dist, F, max_interaction


def _candidate_score_from_recipe(
    instance: PHubInstance,
    h: str,
    selected: List[str],
    recipe: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    D: Any,
    idx: Dict[str, int],
    max_dist: float,
    F: Any,
    max_interaction: float,
    tabu_hub_sets: Optional[set] = None,
    anti_repeat_penalty: float = 0.0,
) -> float:
    """Compute the score of one candidate hub under the LLM-designed recipe.

    The anti-repeat term is not an extra objective component. It is only a
    diversification control used when the same hub set has already been tested.
    """
    W = recipe.get("weights", {})
    m = metrics[h]
    dispersion = 0.0
    interaction = 0.0
    if selected:
        if D is not None:
            hi = idx[h]
            dispersion = min(float(D[hi, idx[s]]) for s in selected) / max_dist
        if F is not None:
            hi = idx[h]
            interaction = sum(float(F[hi, idx[s]] + F[idx[s], hi]) for s in selected) / max_interaction

    score = (
        float(W.get("flow", 0.0)) * m["flow"]
        + float(W.get("centrality", 0.0)) * m["centrality"]
        + float(W.get("dispersion", 0.0)) * dispersion
        + float(W.get("interaction", 0.0)) * interaction
        - float(W.get("distance_penalty", 0.0)) * m["distance_penalty"]
    )

    if tabu_hub_sets and anti_repeat_penalty > 0:
        # Penalize hubs that repeatedly appear in already evaluated LLM-HH sets.
        hub_frequency = sum(1 for s in tabu_hub_sets if h in s) / max(1, len(tabu_hub_sets))
        score -= anti_repeat_penalty * hub_frequency
        # Stronger penalty if adding this hub would exactly reproduce a seen set.
        maybe_set = tuple(sorted(selected + [h]))
        if len(selected) + 1 == instance.p and maybe_set in tabu_hub_sets:
            score -= 2.0 * anti_repeat_penalty

    # Stable tie breaker by candidate index.
    score -= 1e-9 * instance.candidates.index(h)
    return float(score)


def construct_hubs_from_recipe(
    instance: PHubInstance,
    recipe: Dict[str, Any],
    rng: random.Random,
    tabu_hub_sets: Optional[set] = None,
    anti_repeat_penalty: float = 0.0,
) -> List[str]:
    """Instantiate the LLM-designed low-level heuristic.

    The scoring rule is deterministic except for optional RCL selection. This
    is a controlled hyper-heuristic: the LLM controls weights and RCL size;
    Python performs all calculations.

    When tabu_hub_sets is supplied, already evaluated hub sets are softly
    discouraged. This prevents the LLM-HH loop from spending all iterations on
    the same recipe-induced solution, while keeping the objective evaluation
    unchanged and fully deterministic.
    """
    # If model returns direct hubs, evaluate them but do not repair missing IDs here.
    if "direct_hubs" in recipe:
        hubs = []
        for h in recipe["direct_hubs"]:
            if h in instance.candidates and h not in hubs:
                hubs.append(h)
        if len(hubs) == instance.p:
            return hubs

    metrics, D, idx, max_dist, F, max_interaction = _recipe_score_context(instance)
    rcl_size = int(recipe.get("rcl_size", 1))
    avoid = set(recipe.get("avoid", []))

    selected: List[str] = []
    for h in recipe.get("must_include", []):
        if h in instance.candidates and h not in selected and h not in avoid and len(selected) < instance.p:
            selected.append(h)

    while len(selected) < instance.p:
        candidates = [h for h in instance.candidates if h not in selected and h not in avoid]
        if not candidates:
            candidates = [h for h in instance.candidates if h not in selected]
        scored: List[Tuple[float, str]] = []
        for h in candidates:
            score = _candidate_score_from_recipe(
                instance=instance,
                h=h,
                selected=selected,
                recipe=recipe,
                metrics=metrics,
                D=D,
                idx=idx,
                max_dist=max_dist,
                F=F,
                max_interaction=max_interaction,
                tabu_hub_sets=tabu_hub_sets,
                anti_repeat_penalty=anti_repeat_penalty,
            )
            scored.append((score, h))
        scored.sort(reverse=True)
        rcl = scored[: max(1, min(rcl_size, len(scored)))]
        # RCL adds controlled diversity; rcl_size=1 is deterministic.
        chosen = rcl[0][1] if len(rcl) == 1 else rng.choice(rcl)[1]
        selected.append(chosen)
    return selected


def repair_duplicate_hh_solution(
    instance: PHubInstance,
    recipe: Dict[str, Any],
    rng: random.Random,
    seen_hub_sets: set,
    apply_swap: bool = False,
    duplicate_repair_attempts: int = 6,
    anti_repeat_penalty: float = 0.25,
) -> Tuple[Solution, Dict[str, Any], int, bool]:
    """Try to turn a repeated LLM-HH hub set into a new evaluated candidate.

    First, the original recipe is evaluated. If it repeats a previously seen
    hub set, Python keeps the LLM weights but progressively increases R and
    applies a soft anti-repeat penalty. This is cheaper than calling the local
    model again and gives the paper a meaningful diversity-control mechanism.
    """
    best_duplicate: Optional[Solution] = None
    max_attempts = max(0, int(duplicate_repair_attempts))
    base_r = int(recipe.get("rcl_size", 1))
    for attempt in range(max_attempts + 1):
        trial_recipe = dict(recipe)
        trial_recipe["weights"] = dict(recipe.get("weights", {}))
        trial_recipe["rcl_size"] = max(1, min(5, base_r + attempt))
        penalty = 0.0 if attempt == 0 else float(anti_repeat_penalty) * attempt
        sol = evaluate_hh_recipe(
            instance,
            trial_recipe,
            rng,
            apply_swap=apply_swap,
            tabu_hub_sets=seen_hub_sets,
            anti_repeat_penalty=penalty,
        )
        key = tuple(sorted(sol.hubs))
        if not math.isfinite(sol.objective):
            continue
        if key not in seen_hub_sets:
            return sol, trial_recipe, attempt, False
        if best_duplicate is None or sol.objective < best_duplicate.objective:
            best_duplicate = sol
    if best_duplicate is None:
        best_duplicate = evaluate_hh_recipe(instance, recipe, rng, apply_swap=apply_swap)
    return best_duplicate, recipe, max_attempts, True


def evaluate_hh_recipe(
    instance: PHubInstance,
    recipe: Dict[str, Any],
    rng: random.Random,
    apply_swap: bool = False,
    tabu_hub_sets: Optional[set] = None,
    anti_repeat_penalty: float = 0.0,
) -> Solution:
    hubs = construct_hubs_from_recipe(
        instance,
        recipe,
        rng,
        tabu_hub_sets=tabu_hub_sets,
        anti_repeat_penalty=anti_repeat_penalty,
    )
    sol = canonical_solution(instance, hubs, None, source="LLM-HH+Evaluator")
    if apply_swap:
        sol, _ = single_swap_local_search(instance, sol)
        sol.source = "LLM-HH+Swap"
    return sol

def llm_hyperheuristic_search(
    instance: PHubInstance,
    iterations: int,
    model: str,
    llm_provider: str,
    seed: int,
    ollama_url: str,
    ollama_timeout: int,
    llm_max_tokens: int,
    print_prompt: bool = False,
    ollama_json_format: bool = False,
    apply_swap: bool = False,
    trace_path: Optional[str] = None,
    duplicate_repair_attempts: int = 0,
    anti_repeat_penalty: float = 0.25,
    llm_duplicate_retries: int = 2,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run an independent LLM-controlled hyper-heuristic.

    Independence guarantee:
    - It does not start from Greedy.
    - It does not read Greedy, Random, or reference optimum.
    - The memory given to the LLM contains only previous LLM-HH recipes.
    """
    rng = random.Random(seed)
    temperature = 0.25 if str(llm_provider).lower() == "ollama" else 0.7
    memory: Dict[str, Any] = {"valid_recipes": [], "recent_sets": [], "bad_hub_sets": []}
    trace_rows: List[Dict[str, Any]] = []
    best: Optional[Solution] = None
    seen_hub_sets = set()

    for it in range(1, max(1, iterations) + 1):
        prompt = build_hh_prompt(instance, memory, temperature)
        if print_prompt:
            print("\n--- LLM-HH PROMPT START ---")
            print(prompt)
            print("--- LLM-HH PROMPT END ---\n")
        start_call = time.perf_counter()
        raw = ""
        try:
            raw = call_llm(
                prompt,
                provider=llm_provider,
                model=model,
                temperature=temperature,
                max_output_tokens=llm_max_tokens,
                ollama_url=ollama_url,
                ollama_timeout=ollama_timeout,
                ollama_json_format=ollama_json_format,
            )
            try:
                recipe = parse_hh_recipe(raw, instance)
            except Exception as parse_exc:
                # Phi-3 sometimes answers with prose. Retry once with a tiny strict JSON prompt.
                print(f"LLM-HH iteration {it}/{iterations} returned non-JSON. Retrying once. Raw: {str(raw)[:160]!r}")
                retry_prompt = build_hh_json_retry_prompt(raw, instance)
                retry_raw = call_llm(
                    retry_prompt,
                    provider=llm_provider,
                    model=model,
                    temperature=0.05,
                    max_output_tokens=min(64, max(32, int(llm_max_tokens))),
                    ollama_url=ollama_url,
                    ollama_timeout=ollama_timeout,
                    ollama_json_format=False,
                )
                raw = (str(raw) + "\n---RETRY---\n" + str(retry_raw))[:1000]
                recipe = parse_hh_recipe(retry_raw, instance)
            used_recipe = recipe
            sol = evaluate_hh_recipe(instance, used_recipe, rng, apply_swap=apply_swap)
            key = tuple(sorted(sol.hubs))
            duplicate = math.isfinite(sol.objective) and key in seen_hub_sets
            llm_retry_used = 0
            repair_attempts_used = 0

            # Model-guided anti-repetition: if the recipe leads to a hub set that
            # has already been evaluated, ask the LLM itself to redesign W/R/A.
            # Python does not invent new weights here; it only evaluates the
            # model's new recipe.
            while duplicate and llm_retry_used < max(0, int(llm_duplicate_retries)):
                memory["bad_hub_sets"].insert(0, list(key))
                memory["bad_hub_sets"] = memory["bad_hub_sets"][:8]
                retry_prompt = build_hh_duplicate_retry_prompt(instance, memory, sol.hubs)
                if print_prompt:
                    print("\n--- LLM-HH DUPLICATE RETRY PROMPT START ---")
                    print(retry_prompt)
                    print("--- LLM-HH DUPLICATE RETRY PROMPT END ---\n")
                retry_raw = call_llm(
                    retry_prompt,
                    provider=llm_provider,
                    model=model,
                    temperature=min(0.9, temperature + 0.25),
                    max_output_tokens=llm_max_tokens,
                    ollama_url=ollama_url,
                    ollama_timeout=ollama_timeout,
                    ollama_json_format=False,
                )
                raw = (str(raw) + "\n---DUPLICATE_RETRY---\n" + str(retry_raw))[:1200]
                used_recipe = parse_hh_recipe(retry_raw, instance)
                sol = evaluate_hh_recipe(instance, used_recipe, rng, apply_swap=apply_swap)
                key = tuple(sorted(sol.hubs))
                duplicate = math.isfinite(sol.objective) and key in seen_hub_sets
                llm_retry_used += 1

            # Optional Python-side repair is now only a fallback and is disabled
            # by default. Use --hh-duplicate-repair-attempts if you want it.
            if duplicate and int(duplicate_repair_attempts) > 0:
                sol, used_recipe, repair_attempts_used, duplicate = repair_duplicate_hh_solution(
                    instance=instance,
                    recipe=used_recipe,
                    rng=rng,
                    seen_hub_sets=seen_hub_sets,
                    apply_swap=apply_swap,
                    duplicate_repair_attempts=duplicate_repair_attempts,
                    anti_repeat_penalty=anti_repeat_penalty,
                )
                key = tuple(sorted(sol.hubs))

            elapsed = time.perf_counter() - start_call
            seen_hub_sets.add(key)
            recipe_short = json.dumps(used_recipe.get("weights", {}), ensure_ascii=False, separators=(",", ":"))[:160]
            status = "ok" if math.isfinite(sol.objective) else "infeasible"
            if llm_retry_used > 0 and not duplicate and math.isfinite(sol.objective):
                status = "ok_model_diversified"
            elif repair_attempts_used > 0 and not duplicate and math.isfinite(sol.objective):
                status = "ok_python_repaired"
            elif duplicate:
                status = "duplicate_after_model_retry"
                memory["bad_hub_sets"].insert(0, list(key))
                memory["bad_hub_sets"] = memory["bad_hub_sets"][:8]
            if math.isfinite(sol.objective):
                row_mem = {"hubs": sol.hubs, "objective": sol.objective, "recipe_short": recipe_short}
                memory["recent_sets"].insert(0, row_mem)
                memory["recent_sets"] = memory["recent_sets"][:8]
                memory["valid_recipes"].insert(0, row_mem)
                memory["valid_recipes"] = sorted(memory["valid_recipes"], key=lambda x: x["objective"])[:5]
                if best is None or sol.objective < best.objective:
                    best = sol
            trace_rows.append({
                "iteration": it,
                "status": status,
                "objective": sol.objective if math.isfinite(sol.objective) else None,
                "hubs": _format_hubs(sol.hubs),
                "recipe_weights": recipe_short,
                "rcl_size": used_recipe.get("rcl_size"),
                "llm_duplicate_retries_used": llm_retry_used,
                "repair_attempts_used": repair_attempts_used,
                "anti_repeat_penalty": float(anti_repeat_penalty) * max(0, repair_attempts_used),
                "must_include": _format_hubs(used_recipe.get("must_include", [])),
                "avoid": _format_hubs(used_recipe.get("avoid", [])),
                "time_seconds": round(elapsed, 4),
                "raw_response": raw[:500],
            })
            print(f"LLM-HH iteration {it}/{iterations} | status={status} | objective={sol.objective:.6f} | hubs={sol.hubs}")
        except Exception as exc:
            elapsed = time.perf_counter() - start_call
            trace_rows.append({
                "iteration": it,
                "status": "failed",
                "objective": None,
                "hubs": "",
                "recipe_weights": "",
                "rcl_size": None,
                "llm_duplicate_retries_used": None,
                "repair_attempts_used": None,
                "anti_repeat_penalty": None,
                "must_include": "",
                "avoid": "",
                "time_seconds": round(elapsed, 4),
                "raw_response": raw[:500],
                "error": str(exc),
            })
            print(f"LLM-HH iteration {it}/{iterations} failed. Reason: {exc}")
        # Slightly increase diversity when recent recipes stagnate.
        temperature = min(0.8, temperature + 0.05)

    trace_df = pd.DataFrame(trace_rows)
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        trace_df.to_csv(trace_path, index=False)
        print(f"Saved LLM-HH trace: {trace_path}")

    valid = trace_df[trace_df["objective"].notna()] if not trace_df.empty else pd.DataFrame()
    hub_sets = []
    if not valid.empty:
        for val in valid["hubs"].tolist():
            hub_sets.append([x for x in str(val).split(",") if x])
    stats = {
        "evaluations": int(len(trace_df)),
        "valid_evaluations": int(len(valid)),
        "unique_hub_sets": int(len({tuple(sorted(h)) for h in hub_sets})),
        "diversity_jaccard": round(_hub_set_diversity(hub_sets), 4),
        "apply_swap": bool(apply_swap),
        "llm_duplicate_retries": int(llm_duplicate_retries),
        "duplicate_repair_attempts": int(duplicate_repair_attempts),
        "anti_repeat_penalty": float(anti_repeat_penalty),
    }

    if best is None:
        return {
            "llm_status": "failed_no_valid_recipe",
            "llm_failure_reason": "No valid LLM-HH recipe produced a feasible p-hub set.",
            "stats": stats,
        }, trace_df

    return {
        "llm_status": "ok",
        "instance": instance.name,
        "n": len(instance.nodes),
        "p": instance.p,
        "best_hubs": best.hubs,
        "allocation": best.allocation,
        "objective": best.objective,
        "source": best.source,
        "stats": stats,
    }, trace_df


def run_llm_evaluator_for_table(instance: PHubInstance, args: argparse.Namespace) -> Tuple[Optional[Solution], str, str, float]:
    """Run the independent LLM-HH+Evaluator method for the final table.

    Unlike the previous evolutionary variant, this method does not initialize
    from Greedy and does not use outputs of Random, or the
    reference optimum. The LLM controls a scoring heuristic; Python evaluates.
    """
    if args.no_llm:
        return None, "not_run_no_llm_flag", "LLM was disabled by --no-llm", 0.0

    provider = str(args.llm_provider).lower()
    if provider == "gemini":
        if genai is None or types is None:
            return None, "not_run_missing_google_genai", "Install google-genai to run Gemini LLM-HH+Evaluator", 0.0
        if not get_api_key():
            return None, "not_run_no_api_key", "Set GEMINI_API_KEY or GOOGLE_API_KEY to run Gemini LLM-HH+Evaluator", 0.0
    elif provider == "ollama":
        pass
    else:
        return None, "not_run_unknown_provider", f"Unknown LLM provider: {provider}", 0.0

    trace_path = None
    if hasattr(args, "_out_prefix_for_trace"):
        trace_path = f"{args._out_prefix_for_trace}_llm_hh_trace.csv"

    start = time.perf_counter()
    result, _trace_df = llm_hyperheuristic_search(
        instance=instance,
        iterations=args.iterations,
        model=args.model,
        llm_provider=provider,
        seed=args.seed,
        ollama_url=args.ollama_url,
        ollama_timeout=args.ollama_timeout,
        llm_max_tokens=args.llm_max_tokens,
        print_prompt=args.print_prompt,
        ollama_json_format=args.ollama_json_format,
        apply_swap=getattr(args, "llm_apply_swap", False),
        trace_path=trace_path,
        duplicate_repair_attempts=getattr(args, "hh_duplicate_repair_attempts", 0),
        anti_repeat_penalty=getattr(args, "hh_anti_repeat_penalty", 0.25),
        llm_duplicate_retries=getattr(args, "hh_llm_duplicate_retries", 2),
    )
    elapsed = time.perf_counter() - start
    status = str(result.get("llm_status", "failed"))
    if status != "ok":
        return None, status, str(result.get("llm_failure_reason", "LLM-HH failed")), elapsed

    sol = _solution_from_result(result, source=result.get("source", "LLM-HH+Evaluator"))
    stats = result.get("stats", {})
    note = (
        f"{sol.source}; independent HH; evaluations={stats.get('evaluations')}; "
        f"valid={stats.get('valid_evaluations')}; unique={stats.get('unique_hub_sets')}; "
        f"diversity={stats.get('diversity_jaccard')}"
    )
    sol.source = note
    return sol, "ok", "", elapsed


def run_final_article_table(instance: PHubInstance, args: argparse.Namespace, out_prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Solution]]:
    """Create the final output table requested for the article.

    Outputs:
      1) a wide one-row table with columns:
         Reference Optimum, Greedy, Random, LLM+Evaluator
      2) a long table with objective, gap, hubs, time, and status for each method.

    In this version, the classical baselines are intentionally reported without
    hub-swap local search:
      - Greedy: constructive greedy hub selection only, repeated for the common
        evaluation budget. Since it is deterministic, repetitions normally return
        the same answer.
      - Random: independent random multi-start hub selection only.
      - LLM+Evaluator: model-guided LLM-HH iterations.

    The common evaluation budget is args.iterations; default = 20.
    """
    method_iterations = max(1, int(getattr(args, "iterations", 20)))
    # Force the table methods to use exactly the same number of iterations/evaluations.
    args.iterations = method_iterations
    args.baseline_runs = method_iterations

    ref = lookup_known_optimum_details(args.known_optima, instance)
    known_opt = ref["known_optimum"]
    long_rows: List[Dict[str, Any]] = []
    solutions: Dict[str, Solution] = {}

    long_rows.append({
        "instance": instance.name,
        "n": len(instance.nodes),
        "p": instance.p,
        "alpha": instance.transfer_factor,
        "method": "Reference Optimum",
        "iterations": 0,
        "objective": known_opt,
        "gap_to_reference_percent": 0.0 if known_opt is not None else None,
        "hubs": "",
        "time_seconds": 0.0,
        "status": ref["status"],
        "reference": ref["reference"],
        "reference_table": ref["reference_table"],
        "note": ref["note"],
    })

    def record(method: str, iterations_used: int, func) -> None:
        start = time.perf_counter()
        sol = func()
        elapsed = time.perf_counter() - start
        solutions[method] = sol
        long_rows.append({
            "instance": instance.name,
            "n": len(instance.nodes),
            "p": instance.p,
            "alpha": instance.transfer_factor,
            "method": method,
            "iterations": iterations_used,
            "objective": sol.objective,
            "gap_to_reference_percent": _gap_percent(sol.objective, known_opt),
            "hubs": _format_hubs(sol.hubs),
            "time_seconds": round(elapsed, 4),
            "status": "ok",
            "reference": "",
            "reference_table": "",
            "note": sol.source,
        })
        gap_txt = "" if known_opt is None else f" | gap={_gap_percent(sol.objective, known_opt):.4f}%"
        print(f"{method:15s} | iter={iterations_used:02d} | objective={sol.objective:.6f}{gap_txt} | hubs={sol.hubs} | time={elapsed:.3f}s")

    print("\n================ REQUIRED OUTPUT TABLE RUNS ================")
    if known_opt is None:
        print("Reference Optimum | not found for this n, p, alpha")
    else:
        print(f"Reference Optimum | objective={known_opt:.6f} | status={ref['status']}")

    record("Greedy", method_iterations, lambda: greedy_repeated_search(instance, iterations=method_iterations))
    record("Random", method_iterations, lambda: random_multistart_search(instance, starts=method_iterations, seed=args.seed))

    llm_sol, llm_status, llm_note, llm_elapsed = run_llm_evaluator_for_table(instance, args)
    if llm_sol is not None:
        solutions["LLM+Evaluator"] = llm_sol
        long_rows.append({
            "instance": instance.name,
            "n": len(instance.nodes),
            "p": instance.p,
            "alpha": instance.transfer_factor,
            "method": "LLM+Evaluator",
            "iterations": method_iterations,
            "objective": llm_sol.objective,
            "gap_to_reference_percent": _gap_percent(llm_sol.objective, known_opt),
            "hubs": _format_hubs(llm_sol.hubs),
            "time_seconds": round(llm_elapsed, 4),
            "status": llm_status,
            "reference": "",
            "reference_table": "",
            "note": llm_sol.source,
        })
        gap_txt = "" if known_opt is None else f" | gap={_gap_percent(llm_sol.objective, known_opt):.4f}%"
        print(f"{'LLM+Evaluator':15s} | iter={method_iterations:02d} | objective={llm_sol.objective:.6f}{gap_txt} | hubs={llm_sol.hubs} | time={llm_elapsed:.3f}s")
    else:
        long_rows.append({
            "instance": instance.name,
            "n": len(instance.nodes),
            "p": instance.p,
            "alpha": instance.transfer_factor,
            "method": "LLM+Evaluator",
            "iterations": method_iterations,
            "objective": None,
            "gap_to_reference_percent": None,
            "hubs": "",
            "time_seconds": round(llm_elapsed, 4),
            "status": llm_status,
            "reference": "",
            "reference_table": "",
            "note": llm_note,
        })
        print(f"{'LLM+Evaluator':15s} | iter={method_iterations:02d} | {llm_status} | {llm_note}")

    long_df = pd.DataFrame(long_rows)

    wide: Dict[str, Any] = {
        "Instance": instance.name,
        "n": len(instance.nodes),
        "p": instance.p,
        "alpha": instance.transfer_factor,
        "Iterations per method": method_iterations,
        "Reference Optimum": known_opt,
    }
    for method in ["Greedy", "Random", "LLM+Evaluator"]:
        row = long_df[long_df["method"] == method].iloc[0]
        wide[method] = row["objective"]
        wide[f"{method} Gap (%)"] = row["gap_to_reference_percent"]
        wide[f"{method} Hubs"] = row["hubs"]
        wide[f"{method} Time (s)"] = row["time_seconds"]
        wide[f"{method} Status"] = row["status"]
    wide["Reference Source"] = ref["reference"]
    wide["Reference Table"] = ref["reference_table"]
    wide_df = pd.DataFrame([wide])

    wide_path = f"{out_prefix}_final_article_table.csv"
    long_path = f"{out_prefix}_final_article_table_long.csv"
    wide_df.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)

    compact_cols = ["Instance", "n", "p", "alpha", "Iterations per method", "Reference Optimum", "Greedy", "Random", "LLM+Evaluator"]
    compact_path = f"{out_prefix}_final_article_table_compact.csv"
    wide_df[compact_cols].to_csv(compact_path, index=False)

    if getattr(args, "save_method_solutions", True):
        for method, sol in solutions.items():
            safe = method.lower().replace("+", "_").replace(" ", "_")
            pd.DataFrame({"hub_id": sol.hubs}).to_csv(f"{out_prefix}_{safe}_hubs.csv", index=False)
            pd.DataFrame([{"node_id": node, "assigned_hub": hub} for node, hub in sol.allocation.items()]).to_csv(
                f"{out_prefix}_{safe}_allocation.csv", index=False
            )

    print(f"\nSaved final article table: {wide_path}")
    print(f"Saved final long table:    {long_path}")
    print(f"Saved compact table:       {compact_path}")
    return wide_df, long_df, solutions


def run_baseline_comparison(instance: PHubInstance, args: argparse.Namespace) -> pd.DataFrame:
    """Run the non-LLM baselines requested for the article table."""
    methods = []
    known_opt = lookup_known_optimum(args.known_optima, instance)

    def record(name: str, func) -> None:
        start = time.perf_counter()
        sol = func()
        elapsed = time.perf_counter() - start
        row = {
            "instance": instance.name,
            "n": len(instance.nodes),
            "p": instance.p,
            "alpha": instance.transfer_factor,
            "method": name,
            "objective": sol.objective,
            "known_optimum": known_opt,
            "gap_to_known_optimum_percent": None,
            "hubs": ",".join(sol.hubs),
            "time_seconds": round(elapsed, 4),
            "source": sol.source,
            "status": "ok",
        }
        if known_opt is not None and abs(known_opt) > 1e-12:
            row["gap_to_known_optimum_percent"] = 100.0 * (sol.objective - known_opt) / abs(known_opt)
        methods.append(row)
        gap_txt = "" if row["gap_to_known_optimum_percent"] is None else f" gap={row['gap_to_known_optimum_percent']:.3f}%"
        print(f"{name:18s} objective={sol.objective:.6f}{gap_txt} hubs={sol.hubs} time={elapsed:.3f}s")

    record("Greedy", method_iterations, lambda: greedy_repeated_search(instance, iterations=method_iterations))
    record("Random", method_iterations, lambda: random_multistart_search(instance, starts=method_iterations, seed=args.seed))

    df = pd.DataFrame(methods).sort_values("objective", ascending=True).reset_index(drop=True)
    best_obj = float(df.loc[0, "objective"])
    df["gap_to_best_found_percent"] = 100.0 * (df["objective"] - best_obj) / max(abs(best_obj), 1e-9)
    return df


def add_known_gap_to_result(result: Dict[str, Any], known_opt: Optional[float]) -> Dict[str, Any]:
    result["known_optimum"] = known_opt
    if known_opt is not None and abs(known_opt) > 1e-12:
        result["gap_to_known_optimum_percent"] = 100.0 * (float(result["objective"]) - known_opt) / abs(known_opt)
    else:
        result["gap_to_known_optimum_percent"] = None
    return result


def save_outputs(result: Dict[str, Any], out_prefix: str) -> None:
    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out_prefix}_solution.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    pd.DataFrame([{"node_id": node, "assigned_hub": hub} for node, hub in result["allocation"].items()]).to_csv(
        f"{out_prefix}_allocation.csv", index=False
    )
    pd.DataFrame({"hub_id": result["best_hubs"]}).to_csv(f"{out_prefix}_hubs.csv", index=False)
    print(f"\nSaved: {out_prefix}_solution.json, {out_prefix}_allocation.csv, {out_prefix}_hubs.csv")


def _is_default_dataset_dir(dataset_dir: Optional[str]) -> bool:
    """Return True when dataset_dir was not supplied by the user."""
    return dataset_dir is None or str(dataset_dir).strip() == ""


def build_paths(args: argparse.Namespace) -> Tuple[str, str, Optional[str], Optional[str], str, str]:
    """Build input/output paths.

    Priority for choosing the dataset folder:
      1) --dataset-dir, when explicitly provided.
      2) --n/--cab-size, mapped to CAB_instances/CAB{n}.
      3) BASE_DIR, for compatibility with custom CSVs next to the script.
    """
    if not _is_default_dataset_dir(args.dataset_dir):
        dataset_dir = Path(args.dataset_dir).resolve()
    elif args.cab_size is not None:
        dataset_dir = (BASE_DIR / "CAB_instances" / f"CAB{args.cab_size}").resolve()
    else:
        dataset_dir = BASE_DIR.resolve()

    nodes_path = Path(args.nodes).resolve() if args.nodes else dataset_dir / "nodes.csv"
    flows_path = Path(args.flows).resolve() if args.flows else dataset_dir / "flows.csv"
    candidates_path = Path(args.candidates).resolve() if args.candidates else dataset_dir / "candidate_hubs.csv"
    distances_path = Path(args.distances).resolve() if args.distances else dataset_dir / "distances.csv"
    candidates = str(candidates_path) if candidates_path.exists() else None
    distances = str(distances_path) if distances_path.exists() else None
    instance_name = args.instance_name or dataset_dir.name or "instance"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else dataset_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_prefix:
        out_prefix = str(Path(args.out_prefix).resolve())
    else:
        alpha_txt = str(args.transfer_factor).replace(".", "p")
        out_prefix = str(out_dir / f"{instance_name}_p{args.p}_a{alpha_txt}")
    return str(nodes_path), str(flows_path), candidates, distances, instance_name, out_prefix


def _prompt_int(message: str, default: Optional[int] = None, allowed: Optional[List[int]] = None) -> int:
    while True:
        suffix = ""
        if default is not None:
            suffix += f" [default: {default}]"
        if allowed:
            suffix += f" {allowed}"
        raw = input(f"{message}{suffix}: ").strip()
        if raw == "" and default is not None:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter an integer value.")
                continue
        if allowed is not None and value not in allowed:
            print(f"Allowed values are: {allowed}")
            continue
        return value


def _prompt_float(message: str, default: Optional[float] = None, allowed: Optional[List[float]] = None) -> float:
    while True:
        suffix = ""
        if default is not None:
            suffix += f" [default: {default}]"
        if allowed:
            suffix += f" {allowed}"
        raw = input(f"{message}{suffix}: ").strip()
        if raw == "" and default is not None:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Please enter a numeric value, e.g. 0.8")
                continue
        if allowed is not None and not any(abs(value - x) <= 1e-12 for x in allowed):
            print(f"Allowed values are: {allowed}")
            continue
        return value


def _prompt_yes_no(message: str, default: bool = False) -> bool:
    label = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{message} [{label}]: ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes", "1", "true", "بله", "آره"}:
            return True
        if raw in {"n", "no", "0", "false", "خیر", "نه"}:
            return False
        print("Please answer y or n.")


def apply_interactive_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """Ask the user for the three main experiment inputs: n, p, and alpha."""
    print("\nInteractive CAB input mode")
    print("--------------------------")
    cab_size = _prompt_int("Problem size n / CAB size", default=args.cab_size or 25, allowed=VALID_CAB_SIZES)
    max_p = cab_size
    p_value = _prompt_int("Number of hubs p", default=args.p, allowed=list(range(1, max_p + 1)))
    alpha_value = _prompt_float("Transfer discount alpha", default=args.transfer_factor)

    args.cab_size = cab_size
    args.p = p_value
    args.transfer_factor = alpha_value
    args.collection_factor = 1.0
    args.distribution_factor = 1.0

    # In interactive mode the CAB folder is selected automatically unless the
    # user has already provided a custom --dataset-dir.
    if _is_default_dataset_dir(args.dataset_dir):
        args.dataset_dir = None

    if _prompt_yes_no("Run LLM-free baseline comparison", default=args.compare_baselines):
        args.compare_baselines = True
    if _prompt_yes_no("Run without LLM", default=args.no_llm or True):
        args.no_llm = True
    return args

def main() -> None:
    parser = argparse.ArgumentParser(description="CAB-ready SA-p-hub solver with baselines and independent LLM-controlled hyper-heuristic.")
    parser.add_argument("--interactive", action="store_true", help="Ask for CAB size n, number of hubs p, and alpha at runtime")
    parser.add_argument("--n", "--cab-size", dest="cab_size", type=int, choices=VALID_CAB_SIZES, default=None, help="CAB problem size. Allowed: 5, 10, 15, 20, 25. This selects CAB_instances/CAB{n}")
    parser.add_argument("--dataset-dir", default=None, help="Folder containing nodes.csv, flows.csv, distances.csv, candidate_hubs.csv. Overrides --n if provided")
    parser.add_argument("--nodes", default=None, help="Override nodes CSV path")
    parser.add_argument("--flows", default=None, help="Override flows CSV path")
    parser.add_argument("--distances", default=None, help="Override distances CSV path. If omitted, dataset-dir/distances.csv is used if present")
    parser.add_argument("--candidates", default=None, help="Override candidate hubs CSV path")
    parser.add_argument("--instance-name", default=None)
    parser.add_argument("--p", type=int, default=DEFAULT_P, help="Number of hubs to locate")
    parser.add_argument("--iterations", type=int, default=20, help="Common number of iterations/evaluations for Greedy, Random, and LLM-HH in the final table")
    parser.add_argument("--llm-provider", choices=["ollama", "gemini"], default="ollama", help="LLM backend for LLM+Evaluator")
    parser.add_argument("--model", default=None, help="Model name. Defaults: phi3-cab for Ollama, gemini-2.5-flash-lite for Gemini")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Local Ollama API URL")
    parser.add_argument("--ollama-timeout", type=int, default=180, help="Seconds to wait for each local Ollama call")
    parser.add_argument("--llm-max-tokens", type=int, default=96, help="Maximum tokens generated by the LLM")
    parser.add_argument("--print-prompt", action="store_true", help="Print the exact LLM prompt for debugging")
    parser.add_argument("--ollama-json-format", action="store_true", help="Force Ollama JSON format grammar; can be slower for small local models")
    parser.add_argument("--llm-apply-swap", action="store_true", help="Optional ablation: apply classical swap after LLM-HH construction. Default is off, so LLM+Evaluator is independent of Greedy.")
    parser.add_argument("--hh-llm-duplicate-retries", type=int, default=2, help="When LLM-HH repeats an already evaluated hub set, ask the LLM itself to redesign W/R/A this many times")
    parser.add_argument("--hh-duplicate-repair-attempts", type=int, default=0, help="Optional Python-side fallback repairs after LLM duplicate retries. Default 0 keeps diversification model-guided.")
    parser.add_argument("--hh-anti-repeat-penalty", type=float, default=0.25, help="Soft penalty used only if optional Python-side duplicate repair is enabled")
    parser.add_argument("--collection-factor", type=float, default=1.0)
    parser.add_argument("--transfer-factor", type=float, default=0.75, help="Inter-hub transfer discount factor alpha")
    parser.add_argument("--alpha", type=float, default=None, help="Alias for --transfer-factor. If supplied, it overrides --transfer-factor")
    parser.add_argument("--distribution-factor", type=float, default=1.0)
    parser.add_argument("--normalize-flows", action="store_true", help="Normalize positive flows to sum to 1 after reading flows.csv")
    parser.add_argument("--distance-multiplier", type=float, default=1.0, help="Multiplier applied to all distances after reading")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--compare-baselines", action="store_true", help="Run LLM-free heuristics and save comparison CSV")
    parser.add_argument("--known-optima", default=str(BASE_DIR / "CAB_reference_optima.csv"), help="CSV with columns n,p,alpha,optimal_objective for gap calculations")
    parser.add_argument("--no-final-table", action="store_true", help="Do not create the requested final article table")
    parser.add_argument("--save-method-solutions", action="store_true", default=True, help="Save hub/allocation CSV files for each reported method")
    parser.add_argument("--baseline-runs", type=int, default=20, help="Legacy option. In the final table it is synchronized to --iterations so all methods use the same budget")
    parser.add_argument("--rcl-size", type=int, default=3, help="Legacy option; not used in the final table")
    parser.add_argument("--sa-iterations", type=int, default=150)
    parser.add_argument("--ga-population", type=int, default=16)
    parser.add_argument("--ga-generations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    if args.alpha is not None:
        args.transfer_factor = args.alpha

    if args.model is None:
        args.model = "phi3-cab-v2" if args.llm_provider == "ollama" else "gemini-2.5-flash-lite"

    # If the script is run directly with no command-line arguments in a terminal,
    # ask for the three main inputs instead of silently using defaults.
    if args.interactive or (len(sys.argv) == 1 and sys.stdin.isatty()):
        args = apply_interactive_inputs(args)

    nodes_path, flows_path, candidates_path, distances_path, instance_name, out_prefix = build_paths(args)

    instance = load_instance(
        nodes_path=nodes_path,
        flows_path=flows_path,
        candidates_path=candidates_path,
        distances_path=distances_path,
        p=args.p,
        collection_factor=args.collection_factor,
        transfer_factor=args.transfer_factor,
        distribution_factor=args.distribution_factor,
        normalize_flows=args.normalize_flows,
        distance_multiplier=args.distance_multiplier,
        instance_name=instance_name,
    )

    print("Loaded instance:", instance.name)
    print("Nodes:", len(instance.nodes), "Candidate hubs:", len(instance.candidates), "p:", instance.p)
    print("Transfer factor alpha:", instance.transfer_factor)
    print("Distance source:", distances_path if distances_path else "Euclidean coordinates")
    print("Total positive flow:", round(sum(instance.flows.values()), 10))

    if not args.no_final_table:
        args._out_prefix_for_trace = out_prefix
        wide_df, long_df, solutions = run_final_article_table(instance, args, out_prefix)
        print("\n================ FINAL ARTICLE TABLE ================")
        compact_cols = ["Instance", "n", "p", "alpha", "Iterations per method", "Reference Optimum", "Greedy", "Random", "LLM+Evaluator"]
        print(wide_df[compact_cols].to_string(index=False))

        # Save the best computed solution as the main solution file for compatibility.
        finite_solutions = [sol for sol in solutions.values() if math.isfinite(sol.objective)]
        if finite_solutions:
            best_sol = min(finite_solutions, key=lambda s: s.objective)
            result = {
                "instance": instance.name,
                "n": len(instance.nodes),
                "p": instance.p,
                "collection_factor": instance.collection_factor,
                "transfer_factor": instance.transfer_factor,
                "distribution_factor": instance.distribution_factor,
                "best_hubs": best_sol.hubs,
                "allocation": best_sol.allocation,
                "objective": best_sol.objective,
                "source": best_sol.source,
            }
            result = add_known_gap_to_result(result, lookup_known_optimum(args.known_optima, instance))
            save_outputs(result, out_prefix)
        return

    # Optional legacy behavior if the user explicitly disables the final table.
    if args.compare_baselines:
        print("\n================ LLM-FREE BASELINE COMPARISON ================")
        df = run_baseline_comparison(instance, args)
        comp_path = f"{out_prefix}_baseline_comparison.csv"
        df.to_csv(comp_path, index=False)
        print(f"Saved: {comp_path}")
        print(df.to_string(index=False))

    use_llm = not args.no_llm
    if use_llm and args.llm_provider == "gemini" and not get_api_key():
        print("No Gemini API key found. Use --no-llm or set GEMINI_API_KEY/GOOGLE_API_KEY. Falling back to heuristic mode.")
        use_llm = False

    result = solve_with_llm(instance, iterations=args.iterations, model=args.model, use_llm=use_llm, seed=args.seed)
    known_opt = lookup_known_optimum(args.known_optima, instance)
    result = add_known_gap_to_result(result, known_opt)
    print("\n================ FINAL RESULT ================")
    print("Selected hubs:", result["best_hubs"])
    print("Objective:", f"{result['objective']:.6f}")
    if known_opt is not None:
        print("Known optimum:", f"{known_opt:.6f}")
        print("Gap to known optimum (%):", f"{result['gap_to_known_optimum_percent']:.4f}")
    print("Source:", result["source"])
    save_outputs(result, out_prefix)

if __name__ == "__main__":
    main()
