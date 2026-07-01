# ANN Course Assignments

Implementations for the Artificial Neural Networks (ANN) course homework. Each folder (`HW1`–`HW4`) contains code, reports, and analysis for one assignment.

## HW1 — NumPy MLP on MNIST

Build a multi-layer perceptron from scratch in NumPy (no autograd). Implement forward/backward passes for activations (SELU, HardSwish, Mish), linear layers, and loss functions (softmax cross-entropy, Brier, hinge). Train and evaluate on MNIST with configurable architecture and hyperparameters.

**Key files:** `HW1/codes/` (`layers.py`, `loss.py`, `network.py`, `run_mlp.py`)

## HW2 — PyTorch MLP & CNN on CIFAR-10

Implement BatchNorm and Dropout in PyTorch, then train an MLP and a CNN on CIFAR-10. Compare training vs. inference behavior and study how normalization and regularization affect performance.

**Key files:** `HW2/codes/mlp/`, `HW2/codes/cnn/`

## HW3 — Transformer Language Model

Implement a GPT-style decoder-only Transformer: causal self-attention, layer norms (including RMSNorm), MLP blocks, and language-modeling loss. Train or fine-tune on a text corpus and experiment with decoding strategies (greedy, top-p, temperature sampling).

**Key files:** `HW3/codes/model_tfmr.py`, `HW3/codes/main.py`

## HW4 — LLM Prompting & Inference-Time Scaling

Use a hosted LLM API (OpenAI-compatible) for math reasoning. Implement zero-shot and few-shot prompting on `gsm8k` and `math500`, plus majority voting (self-consistency) with temperature sampling. Analyze when in-context examples and voting help.

**Key files:** `HW4/codes/prompt.py`, `HW4/codes/scale.py`, `HW4/codes/main.py`

**Note:** Pass your API key via the `OPENAI_API_KEY` environment variable; do not commit secrets.

## Repository Layout

```
HW1/   NumPy MLP
HW2/   PyTorch MLP & CNN
HW3/   Transformer LM
HW4/   LLM prompting
```

Each `HW*/codes/` directory holds the runnable implementation; `code_analyze/` (or `code_analysis/`) contains submission diff summaries.
