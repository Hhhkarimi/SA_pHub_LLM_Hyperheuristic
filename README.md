# Language-Model-Based Hyper-Heuristic for the Single-Allocation p-Hub Location Problem

This repository contains the Python implementation and CAB benchmark data used for a language-model-based hyper-heuristic for the single-allocation p-hub location problem.

## Contents

- `sa_p_hub_llm_hyperheuristic.py`: final Python solver.
- `CAB_instances/`: CAB benchmark instances in CSV format for n = 5, 10, 15, 20, and 25.
- `CAB_reference_optima.csv`: reference optimum values used only for gap reporting.
- `Modelfile.phi3-cab.v2`: example Ollama model file for a local Phi-3 model.
- `requirements.txt`: Python dependencies.

## Method summary

The language model is not used as a direct solver. In each iteration, it proposes a compact recipe of the form:

```text
W=0.35,0.18,0.30,0.10,0.07;R=3;A=none
```

where `W` contains the weights for flow, centrality, dispersion, flow interaction, and distance penalty; `R` is the restricted candidate-list size; and `A` is an optional temporary avoidance list. The Python code constructs the hub set, repairs assignments, checks feasibility, and computes the objective value.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For local LLM execution, install Ollama and create a model from your local GGUF file, for example:

```bash
ollama create phi3-cab-v2 -f Modelfile.phi3-cab.v2
```

The GGUF model file is not included in this repository.

## Example runs

Run the proposed method on CAB10 with p = 2 and alpha = 0.8:

```bash
python sa_p_hub_llm_hyperheuristic.py --n 10 --p 2 --alpha 0.8 --model phi3-cab-v2 --llm-provider ollama
```

Print the prompt sent to the model:

```bash
python sa_p_hub_llm_hyperheuristic.py --n 10 --p 2 --alpha 0.8 --model phi3-cab-v2 --llm-provider ollama --print-prompt
```

Run without the language model:

```bash
python sa_p_hub_llm_hyperheuristic.py --n 10 --p 2 --alpha 0.8 --no-llm
```

By default, the final version uses 20 iterations/evaluations for the reported methods.

## Data format

Each CAB instance folder contains:

- `nodes.csv`: node identifiers and coordinates.
- `flows.csv`: normalized origin-destination flows.
- `flows_raw.csv`: original flow matrix values.
- `distances.csv`: benchmark distances.
- `candidate_hubs.csv`: candidate hub list.
- `metadata.json`: instance metadata.

## Reproducibility

Reference optimum values are read from `CAB_reference_optima.csv` only for reporting gaps. They are not provided to the language model during solution construction.
