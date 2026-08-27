# DAY-6: Gen-AI Developer Classroom Notes — 08 Aug 2026

> **Source:** https://directai.blog/2026/08/08/gen-ai-developer-classroom-notes-08-aug-2026/
> **Unsloth Docs:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
> **Audience:** Gen-AI Developers (Beginner → Intermediate)

---

## Table of Contents

1. [Typical Model Training — The 3 Stages](#1-typical-model-training--the-3-stages)
2. [Unsloth for Fine-Tuning](#2-unsloth-for-fine-tuning)
3. [Datasets — The Foundation of Fine-Tuning](#3-datasets--the-foundation-of-fine-tuning)
4. [Hyperparameters — Deep Dive](#4-hyperparameters--deep-dive)
5. [Metrics to Observe During Training](#5-metrics-to-observe-during-training)
6. [Overfitting vs. Underfitting — Diagnosis & Fix](#6-overfitting-vs-underfitting--diagnosis--fix)
7. [Complete Unsloth Fine-Tuning Code Walkthrough](#7-complete-unsloth-fine-tuning-code-walkthrough)
8. [Spring AI Equivalent — Consuming a Fine-Tuned Model](#8-spring-ai-equivalent--consuming-a-fine-tuned-model)
9. [Key Takeaways & Cheat Sheet](#9-key-takeaways--cheat-sheet)

---

## 1. Typical Model Training — The 3 Stages

Modern LLMs (GPT-4, Claude, Llama 3, Gemini) are not trained in one shot. They go through three carefully designed stages, each building on the previous one.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  MODERN LLM TRAINING PIPELINE                            │
│                                                                          │
│  Raw Internet Text                                                        │
│  (Terabytes of data)                                                     │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE 1 — PRE-TRAINING (Base Model)                                │ │
│  │  Goal: Become knowledgeable                                         │ │
│  │  Method: Next-token prediction on massive corpus                    │ │
│  │  Output: A model that knows about the world but can't chat well     │ │
│  └───────────────────────────────┬─────────────────────────────────────┘ │
│                                  │                                       │
│  Instruction Pairs               ▼                                       │
│  (Q&A, task demos)    ┌─────────────────────────────────────────────────┐│
│                       │  STAGE 2 — SUPERVISED FINE-TUNING (SFT)         ││
│                       │  Goal: Learn how to interact                    ││
│                       │  Method: SFT on curated instruction datasets    ││
│                       │  Output: A helpful assistant that follows tasks ││
│                       └──────────────────┬──────────────────────────────┘│
│                                          │                               │
│  Human Preference                        ▼                               │
│  Comparisons          ┌─────────────────────────────────────────────────┐│
│                       │  STAGE 3 — RLHF                                 ││
│                       │  Goal: Answer right, refuse harmful requests    ││
│                       │  Method: Reward model + PPO or DPO              ││
│                       │  Output: Safe, aligned, well-calibrated model   ││
│                       └─────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────┘
```

---

### 1.1 Stage 1 — Pre-Training: "Become Knowledgeable" (Base Model)

#### What Happens

The model is exposed to a massive corpus — trillions of tokens from web pages, books, code, scientific papers, Wikipedia, etc. The training objective is simple: **predict the next token**.

```
Input:  "The Eiffel Tower is located in"
Target: "Paris"

Input:  "def quicksort(arr):\n    if len(arr) <="
Target: "1:"

Loss = Cross-Entropy(predicted token distribution, actual next token)
```

The model learns grammar, facts, reasoning patterns, and world knowledge purely by this self-supervised objective — no human labels required.

#### Scale of Pre-Training

| Model | Parameters | Training Tokens | Hardware | Time |
|-------|-----------|----------------|---------|------|
| GPT-3 | 175B | 300B tokens | ~10,000 A100s | Weeks |
| Llama 3 8B | 8B | 15 Trillion tokens | Thousands of GPUs | Months |
| Llama 3 70B | 70B | 15 Trillion tokens | Thousands of GPUs | Months |

#### Output: The Base Model

A base model (e.g., `meta-llama/Llama-3.1-8B` without "Instruct") can complete text brilliantly but behaves erratically in conversations — it might continue a question instead of answering it.

```
Prompt (to base model):  "What is the capital of France?"
Typical base model output: "What is the capital of Germany? What is the capital of Italy?..."
                            (it continues generating Q&A-style text, not answering!)
```

This is why Stage 2 is critical.

---

### 1.2 Stage 2 — Supervised Fine-Tuning (SFT): "Learn How to Interact" (Instruct Model)

#### What Happens

The base model is fine-tuned on a curated dataset of **instruction-response pairs** using standard supervised learning. A human or AI crafts examples that show the model what "good behavior" looks like.

```
Training Example (Alpaca format):
{
  "instruction": "Summarize the following text in one sentence.",
  "input": "The French Revolution was a period of radical political and societal change...",
  "output": "The French Revolution transformed France from a monarchy into a republic through widespread political and social upheaval."
}
```

#### Formats Used

| Format | Structure | Common Use |
|--------|----------|-----------|
| **Alpaca** | instruction + input + output | General instruction following |
| **ShareGPT / ChatML** | multi-turn user/assistant messages | Conversational fine-tuning |
| **ORPO / DPO** | chosen vs rejected response pairs | Preference optimization |

#### Why SFT Works

SFT teaches the model not what to know (that came from pre-training), but **how to express and apply** that knowledge in the format a user expects. It's like the difference between a professor who knows everything and a good teacher who knows how to explain it.

#### Real-World SFT Datasets

| Dataset | Size | Purpose |
|---------|------|---------|
| `tatsu-lab/alpaca` | 52K | General instruction following |
| `Open-Orca/OpenOrca` | 4.2M | Complex reasoning traces |
| `HuggingFaceH4/ultrachat_200k` | 200K | Multi-turn chat |
| `databricks/databricks-dolly-15k` | 15K | Diverse tasks by employees |

---

### 1.3 Stage 3 — RLHF: "Ensure You Answer Right Questions"

#### What RLHF Stands For

**Reinforcement Learning from Human Feedback** — combines:
1. **Human preferences**: Humans rank multiple model outputs (A is better than B)
2. **Reward Model (RM)**: A classifier trained on those rankings to score any response
3. **RL optimization (PPO)**: The policy (LLM) is updated to maximize the reward

```
┌──────────────────────────────────────────────────────────────┐
│                    RLHF PIPELINE                             │
│                                                              │
│  1. Prompt the SFT model → get multiple responses           │
│                                                              │
│  2. Human rankers compare: "Response A > Response B"        │
│     (helpful, safe, honest rankings)                        │
│                                                              │
│  3. Train a Reward Model (RM) on these preferences          │
│     RM outputs a scalar "goodness score" for any response   │
│                                                              │
│  4. Use PPO (Proximal Policy Optimization) to fine-tune     │
│     the SFT model to maximize RM score                      │
│     (while staying close to original SFT weights)           │
│                                                              │
│  Result: Model that is helpful + safe + aligned             │
└──────────────────────────────────────────────────────────────┘
```

#### RLHF Variants in 2024–2026

| Method | Full Name | Key Idea | Advantage |
|--------|----------|---------|-----------|
| **PPO** | Proximal Policy Optimization | Classic RL with reward model | Most studied, stable |
| **DPO** | Direct Preference Optimization | Eliminates RM, directly trains on pairs | Simpler, no RM needed |
| **GRPO** | Group Relative Policy Optimization | Group-based reward normalization | Used in DeepSeek-R1 |
| **ORPO** | Odds Ratio Preference Optimization | Combines SFT + preference in one loss | Single training pass |

#### What RLHF Achieves

- Model **refuses** harmful requests ("How do I make a bomb?")
- Model is **calibrated** — says "I don't know" when uncertain
- Model is **helpful** — gives complete, accurate, well-structured answers
- Model is **honest** — doesn't hallucinate facts confidently

#### Use Cases Per Stage

| You Want | Stage Needed |
|---------|-------------|
| Model knows your domain's vocabulary | Stage 1 (continued pre-training) |
| Model follows your task instructions | Stage 2 (SFT) |
| Model refuses off-topic/harmful queries | Stage 3 (RLHF/DPO) |
| Model matches a specific persona/tone | Stage 2 (SFT) |
| Model reasons step-by-step | Stage 2+3 (SFT + GRPO/RL) |

---

## 2. Unsloth for Fine-Tuning

### 2.1 What is Unsloth?

**Unsloth** is an open-source Python library that makes LoRA and QLoRA fine-tuning **2–5× faster** and uses **60–80% less VRAM** than standard HuggingFace PEFT. It achieves this through hand-optimized Triton kernels and custom CUDA implementations.

```
┌─────────────────────────────────────────────────────────┐
│              UNSLOTH vs STANDARD PEFT                   │
│                                                         │
│  Task: Fine-tune Llama 3 8B on 1 A100 GPU              │
│                                                         │
│  Standard PEFT + TRL:                                   │
│    Memory: ~22 GB VRAM                                  │
│    Speed:  ~0.8 samples/second                          │
│    Time:   ~8 hours for 10K examples                   │
│                                                         │
│  Unsloth QLoRA:                                         │
│    Memory: ~8 GB VRAM  (63% reduction!)                 │
│    Speed:  ~2.2 samples/second  (2.7× faster!)         │
│    Time:   ~3 hours for 10K examples                   │
│                                                         │
│  This means: Llama 3 70B fits in <48GB VRAM!           │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Why Use Unsloth?

| Feature | Standard PEFT/TRL | Unsloth |
|---------|------------------|---------|
| Speed | Baseline | 2–5× faster |
| VRAM usage | Baseline | 60–80% less |
| Gradient accumulation bug | Present | Fixed |
| Long context training | Limited | 4× longer context |
| Inference speed | Baseline | 2× faster |
| Supported models | All HF models | Llama, Mistral, Gemma, Qwen, Phi, Gemma, etc. |
| Free Colab support | Limited | Yes (3GB VRAM minimum) |

### 2.3 Installation

```bash
# ── For Colab / Linux / WSL ────────────────────────────────────────
pip install unsloth

# ── With specific CUDA version (recommended for local) ─────────────
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# ── Verify installation ────────────────────────────────────────────
python -c "import unsloth; print(unsloth.__version__)"
```

### 2.4 Supported Model Families

```
Unsloth natively supports (optimized kernels):
  ├── Llama family:   Llama 3, 3.1, 3.2, 3.3, 4, CodeLlama
  ├── Mistral family: Mistral, Mixtral (MoE)
  ├── Gemma family:   Gemma 2, 3, 3n
  ├── Qwen family:    Qwen2, Qwen2.5, Qwen3
  ├── Phi family:     Phi-3, Phi-4
  ├── DeepSeek:       DeepSeek-R1, DeepSeek-V3
  └── Others:         Falcon, Yi, Solar, Zephyr, ...
```

---

## 3. Datasets — The Foundation of Fine-Tuning

### 3.1 The Golden Rule

> **"Your fine-tuned model can only be as good as your dataset."**
> Garbage in → garbage out. High quality beats high quantity every time.

### 3.2 Dataset Formats

#### Alpaca Format (Single-Turn)

```json
{
  "instruction": "Classify the sentiment of this review.",
  "input": "The product broke after 2 days. Terrible quality.",
  "output": "NEGATIVE"
}
```

#### ShareGPT / ChatML Format (Multi-Turn)

```json
{
  "conversations": [
    {"role": "user",      "content": "What is LoRA?"},
    {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) is..."},
    {"role": "user",      "content": "How is it different from full fine-tuning?"},
    {"role": "assistant", "content": "Full fine-tuning updates all weights while LoRA..."}
  ]
}
```

#### Why Format Matters

The model learned during pre-training to expect specific token patterns. Using the wrong format causes the model to generate role labels literally rather than understanding them.

### 3.3 Dataset Size Guidelines

| Task Complexity | Minimum Examples | Ideal |
|----------------|-----------------|-------|
| Simple Q&A / style change | 500–1,000 | 5,000 |
| Domain adaptation (medical, legal) | 5,000 | 50,000 |
| New task type (code, math) | 10,000 | 100,000+ |
| Instruct model from base | 50,000 | 500,000+ |

### 3.4 Synthetic Dataset Generation

When you don't have enough real data, generate it with an LLM:

```python
from openai import OpenAI
import json

client = OpenAI()

def generate_qa_pair(document_text: str) -> dict:
    """Generate a Q&A training pair from a document using GPT-4o."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": """You are a dataset curator. Given a document, create ONE high-quality
                         Q&A pair. Return valid JSON: {"question": "...", "answer": "..."}"""
        }, {
            "role": "user",
            "content": f"Document:\n{document_text}"
        }],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# Example: Generate 100 pairs from your internal docs
documents = ["Your company policy text...", "Product manual excerpt..."]  # load your docs
dataset = [generate_qa_pair(doc) for doc in documents]
```

### 3.5 Training on Completions Only (Best Practice)

Training on both user inputs AND assistant responses can hurt accuracy. The QLoRA paper shows training only on **completions (assistant responses)** improves results by ~1–2%.

```
WITHOUT completion-only training (everything in green is trained on):
  USER:      ✅ "What is 2+2?"         ← model learns to predict this too (wasteful)
  ASSISTANT: ✅ "The answer is 4."     ← correct

WITH completion-only training:
  USER:      ❌ "What is 2+2?"         ← masked out (loss not computed here)
  ASSISTANT: ✅ "The answer is 4."     ← only this generates gradient updates
```

```python
# Enable in Unsloth — applies to Llama 3/3.1/3.2/3.3/4:
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part    = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

---

## 4. Hyperparameters — Deep Dive

### 4.1 The LoRA Config Parameters

The core LoRA adapter is controlled by 6 key parameters. Getting these right determines whether your fine-tune generalizes or overfits.

```python
# Full Unsloth LoRA config — every parameter explained:
model = FastLanguageModel.get_peft_model(
    model,
    r                   = 16,    # LoRA Rank
    lora_alpha          = 32,    # LoRA Alpha (= 2 × r → aggressive learning)
    target_modules      = [      # Which weight matrices get LoRA adapters
        "q_proj", "k_proj", "v_proj", "o_proj",     # Attention layers
        "gate_proj", "up_proj", "down_proj",         # MLP/FFN layers
    ],
    lora_dropout        = 0,     # Dropout (0 = faster, Unsloth-optimized)
    bias                = "none",# Don't train bias terms
    use_gradient_checkpointing  = "unsloth",  # Saves 30% extra VRAM
    random_state        = 3407,  # Fixed seed for reproducibility
    use_rslora          = False, # Rank-stabilized LoRA (advanced)
    loftq_config        = None,  # LoftQ initialization (advanced)
)
```

---

### 4.2 Rank (`r`) — "How Much Capacity?"

**What it controls:** The bottleneck dimension of the two LoRA matrices A and B.

```
W_new = W_frozen + (alpha/r) × B × A

A shape: [r × d_in]     e.g., [16 × 4096]
B shape: [d_out × r]    e.g., [4096 × 16]

Trainable params per layer = 2 × r × d
  r=8  → 2 × 8  × 4096 = 65,536  params
  r=16 → 2 × 16 × 4096 = 131,072 params
  r=64 → 2 × 64 × 4096 = 524,288 params
```

**Rules of thumb:**

| Rank | Use When | Risk |
|------|---------|------|
| `r = 8` | Simple task, small dataset, fast iteration | May underfit complex tasks |
| `r = 16` | Most common starting point (recommended) | Balanced |
| `r = 32` | Complex tasks, larger datasets | More VRAM needed |
| `r = 64–128` | Very complex tasks (reasoning, coding) | Risk of overfitting |

> **Blog note:** Rank must be a **multiple of 8** — this is a GPU memory alignment requirement. Non-multiples like `r=10` are valid but slower because GPU tensor cores are optimized for multiples of 8.

---

### 4.3 Alpha (`lora_alpha`) — "How Aggressively to Apply Updates?"

**What it controls:** A scaling factor applied to the LoRA output:

```
W_update = (alpha / r) × B × A

With r=16, alpha=16: scale = 16/16 = 1.0   (standard)
With r=16, alpha=32: scale = 32/16 = 2.0   (more aggressive — common heuristic)
```

**Best Practices (from Unsloth docs):**

| Alpha Setting | Effect | When to Use |
|--------------|--------|------------|
| `alpha = r` | Scale = 1.0 — conservative | Default baseline |
| `alpha = r × 2` | Scale = 2.0 — more aggressive | Most fine-tuning tasks (recommended) |
| RSLoRA (`use_rslora=True`) | Scale = alpha/√r | Large ranks (r ≥ 64), more stable |

> **Simple rule:** Start with `alpha = 2 × r`. So if `r=16`, set `alpha=32`.

---

### 4.4 Learning Rate — "How Big Are the Steps?"

**What it controls:** The magnitude of weight updates at each gradient step.

```
New_weight = Old_weight − (learning_rate × gradient)
```

```
┌────────────────────────────────────────────────────────────────┐
│                   LEARNING RATE EFFECT                         │
│                                                                │
│  Loss                                                          │
│   ▲                                                            │
│   │  Too high LR: loss oscillates / diverges                  │
│   │  ╭──────╮                                                  │
│   │  │      ╰╮ ╭╮                                             │
│   │          ╰╯  ╰╯   ← unstable                              │
│   │                                                            │
│   │  Good LR: smooth decline                                   │
│   │  ╭─╮                                                       │
│   │     ╰─╮                                                    │
│   │        ╰──────────  ← converging                          │
│   │                                                            │
│   │  Too low LR: barely moves                                  │
│   │  ╭───────────────── ← underfitting or very slow           │
│   └──────────────────────────────────────────────────► Steps  │
└────────────────────────────────────────────────────────────────┘
```

**Recommended Values:**

| Scenario | Learning Rate | Notes |
|---------|--------------|-------|
| LoRA / QLoRA Fine-tuning | `2e-4` | Best starting point |
| DPO / GRPO (RL) | `5e-6` | Much lower — RL is sensitive |
| Full Fine-tuning (FFT) | `1e-5` to `5e-5` | Lower rates for all-weight updates |
| If loss is unstable | `÷ 2` or `÷ 10` | Halve the LR and retry |
| If loss barely moves | `× 2` | Double the LR and retry |

---

### 4.5 Batch Size & Gradient Accumulation — "How Much Data Per Update?"

The **Effective Batch Size** is what the model actually "sees" before updating weights:

```
Effective Batch Size = batch_size × gradient_accumulation_steps
```

#### Why This Formula Exists

Loading 32 samples in one GPU forward pass (batch_size=32) requires 32× the VRAM. Instead, you can process 2 samples 16 times, then average the gradients before updating — identical results, 16× less peak VRAM.

```
Both achieve Effective Batch Size = 32:

Option A (VRAM heavy):
  batch_size=32, gradient_accumulation=1
  → 32 samples loaded at once → GPU OOM likely

Option B (VRAM efficient):
  batch_size=2, gradient_accumulation=16
  → 2 samples × 16 micro-batches → same gradient update
  → Only 2 samples ever in VRAM at once ✓
```

#### Configuration Guide

| Goal | `batch_size` | `grad_accum` | Effective | Notes |
|------|-------------|-------------|---------|-------|
| VRAM constrained (8GB) | 1 | 16 | 16 | Safest on small GPU |
| Balanced (16GB) | 2 | 8 | 16 | Recommended default |
| Fast (40GB A100) | 4 | 4 | 16 | Faster per epoch |
| Maximum stability | 2 | 16 | 32 | Better gradient estimates |

> **Unsloth Note:** Standard libraries had a bug where `batch_size=2, grad_accum=8` produced different results from `batch_size=16, grad_accum=1`. Unsloth fixed this — both are now truly equivalent.

---

### 4.6 Epochs — "How Many Times Does the Model See the Data?"

```
1 Epoch = model sees every training example exactly once

Total training steps = (dataset_size / effective_batch_size) × num_epochs
```

**Guidelines:**

| Epochs | Effect | Risk |
|--------|--------|------|
| < 1 (use `max_steps`) | Fast experimentation | Model undertrained |
| 1 | Light adaptation, good generalization | May miss complex patterns |
| 2–3 | Standard for instruction fine-tuning | Sweet spot for most tasks |
| > 3 | Deep domain specialization | High overfitting risk |

> **Rule:** For instruction-following datasets, **1–3 epochs** is the universal recommendation. More than 3 offers diminishing returns and risks overfitting.

---

### 4.7 Other Important Hyperparameters

| Parameter | Recommended | Effect |
|-----------|------------|--------|
| `lora_dropout` | 0 (default in Unsloth) | Regularization; 0 is fastest; use 0.05–0.1 if overfitting |
| `weight_decay` | 0.01 | Penalizes large weights — prevents overfitting |
| `warmup_steps` | 5–10% of total steps | Gradually increases LR from 0 → target at start |
| `lr_scheduler_type` | `"cosine"` or `"linear"` | `cosine` anneals smoothly; `linear` decays linearly |
| `random_state` | 3407 (any integer) | Ensures reproducibility across runs |
| `max_seq_length` | 2048 (start) | Max tokens per training example; increase for long docs |
| `target_modules` | all 7 attention+MLP | Apply LoRA to all layers for best performance |

#### LR Scheduler Visualization

```
Cosine Scheduler:                  Linear Scheduler:
LR                                 LR
▲  ╭───╮                           ▲  ╭─╮
│  │ warm│╲                        │  │ w│╲
│  │  up  ╰──╲__                   │  │  │ ╲
│  │            ╰──╮               │  │   ╲
│  │                ╰─             │  │    ╲──────
└──────────────────────► Steps    └──────────────────► Steps
   (smooth annealing)              (sharp drop)
```

---

### 4.8 Complete Hyperparameter Reference Table

| Hyperparameter | Blog Note | Recommended Value | What Happens if Too High | What Happens if Too Low |
|---------------|----------|------------------|------------------------|------------------------|
| `r` (rank) | Multiple of 8 | 16 or 32 | Overfitting, high VRAM | Underfitting, low capacity |
| `lora_alpha` | 2× rank | r × 2 | Too aggressive updates | Weak LoRA updates |
| `learning_rate` | — | 2e-4 (LoRA) | Loss diverges, instability | Very slow convergence |
| `batch_size` | — | 2 | OOM (out of memory) | More noise per update |
| `gradient_accumulation_steps` | mentioned | 8 | Slow training per epoch | Noisy gradient estimates |
| `num_train_epochs` | mentioned | 1–3 | Overfitting | Underfitting |
| `weight_decay` | — | 0.01 | Poor generalization | Overfitting |
| `lora_dropout` | — | 0 (default) | Training instability | Possible overfitting |
| `warmup_steps` | — | 5–10% of steps | Fast start → instability | Slow warm-up |

---

## 5. Metrics to Observe During Training

### 5.1 Training Loss vs Validation Loss

These two numbers tell you everything about how your fine-tune is progressing.

```
┌─────────────────────────────────────────────────────────────────┐
│                   LOSS CURVES DURING TRAINING                   │
│                                                                 │
│  Loss                                                           │
│   ▲                                                             │
│   │ ╭──╮  ← starting point (high loss = model confused)        │
│   │     ╰─╮                                                     │
│   │        ╰──╮    Training Loss ──────                        │
│   │            ╰────╮                                           │
│   │                  ╰────────────── ← 0.5–1.0 is healthy      │
│   │                                                             │
│   │ ╭───╮  Validation Loss ─ ─ ─ ─                            │
│   │      ╰──╮                                                   │
│   │          ╰──╮                                               │
│   │              ╰────────────────── ← should track train loss │
│   └─────────────────────────────────────────────────► Steps    │
└─────────────────────────────────────────────────────────────────┘
```

- **Training loss** = how well the model fits the training data
- **Validation loss** = how well the model generalizes to unseen data

### 5.2 Loss Behavior Diagnostic Table

The blog specifically asked: *"Express in tabular form how to respond to different behaviors of learning loss and validation loss during fine-tuning."*

Here is the comprehensive answer:

| Scenario | Training Loss | Validation Loss | Diagnosis | Action |
|----------|-------------|----------------|-----------|--------|
| **Healthy training** | Decreasing steadily | Decreasing steadily (close to train) | Model learning well | Continue training |
| **Overfitting** | Decreasing → very low (<0.2) | Decreasing then **increasing** | Memorizing training data | ↓ epochs, ↑ dropout, ↑ weight_decay, get more data |
| **Underfitting** | High, barely decreasing | High, barely decreasing | Model not learning enough | ↑ rank, ↑ LR, ↑ epochs, add more data |
| **Learning rate too high** | Oscillating / spiky / NaN | Chaotic / NaN | LR causes instability | ↓ LR by 10×, add warmup steps |
| **Learning rate too low** | Very slowly decreasing | Very slowly decreasing | Too cautious | ↑ LR by 2–5× |
| **Diverged training** | Increasing (goes up) | Increasing or NaN | Catastrophic failure | Stop, fix LR/batch size, restart |
| **Gap growing** | Decreasing | Plateau or increasing | Generalization gap widening | Early stopping, ↓ epochs |
| **Both losses plateau early** | Plateau at high value | Plateau at high value | Dataset too small or rank too low | More data, ↑ rank, ↑ epochs |
| **Train loss low, val loss high** | ~0.1–0.2 | 1.5+ | Severe overfitting | Much more data, strong regularization |
| **Train=val but both high** | 1.5+ | 1.5+ | Model too small / dataset wrong format | Check data format, ↑ rank, ↑ LR |
| **Sudden spike in loss** | Sharp upward spike | Sharp upward spike | Bad batch / corrupted data | Inspect dataset, add gradient clipping |
| **Loss stuck at same value** | Never changes | Never changes | Frozen gradients / bug | Check `requires_grad`, re-init optimizer |

### 5.3 Healthy Loss Ranges (Rule of Thumb)

| Loss Range | Interpretation |
|-----------|---------------|
| > 2.0 | Model barely learning (check data format) |
| 1.0–2.0 | Early stages of learning |
| **0.5–1.0** | **Healthy learning zone (target)** |
| 0.2–0.5 | Good convergence, watch validation loss |
| < 0.2 | Risk of overfitting — monitor carefully |
| ≈ 0.0 | Almost certainly overfitting (memorization) |

### 5.4 How to Enable Validation in Unsloth

```python
from datasets import load_dataset

# Split dataset into 80% train / 20% validation
dataset = load_dataset("your_dataset")
split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_data = split["train"]
val_data   = split["test"]

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = train_data,
    eval_dataset  = val_data,     # ← enable validation
    args = TrainingArguments(
        evaluation_strategy = "steps",  # evaluate every N steps
        eval_steps          = 100,      # evaluate every 100 steps
        save_strategy       = "steps",
        load_best_model_at_end = True,  # auto-restore best checkpoint
    ),
)
```

---

## 6. Overfitting vs. Underfitting — Diagnosis & Fix

### 6.1 Visual Mental Model

```
UNDERFITTING                GOOD FIT               OVERFITTING
(Too Generic)           (Generalizes Well)       (Too Specialized)

Training data: ●●●●●       ●●●●●                   ●●●●●
               ●●●●●       ●●●●●                   ●●●●●

Decision       ─────────   ─────────               ~~~~~
boundary:      (too flat)  (just right)            (too wiggly)

Unseen data:   ✗✗✗✗✗       ✓✓✓✓✓                   ✗✗✗✗✗
Result:        Fails both  Passes both             Passes train,
               train+val   train+val               fails val
```

### 6.2 Overfitting Solutions

```
Symptoms: train_loss < 0.2, val_loss rising, model gives memorized responses

Quick Fixes (try in order):
1. Reduce num_train_epochs → try 1 epoch instead of 3
2. Increase lora_dropout → set 0.05 or 0.1
3. Increase weight_decay → set 0.01 or 0.1
4. Increase gradient_accumulation_steps → smoother updates
5. Add more training data → merge with open-source datasets

Advanced:
6. LoRA alpha scaling → multiply alpha by 0.5 after training
7. Weight averaging → average base model + fine-tuned model weights
8. Enable early stopping → stop when val_loss increases 3× in a row
```

```python
# LoRA Alpha Scaling (post-training fix for overfitting)
# Reduces the influence of fine-tuning at inference time
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "./adapter")
for name, module in model.named_modules():
    if hasattr(module, 'scaling'):
        module.scaling = {k: v * 0.5 for k, v in module.scaling.items()}
        # This halves the alpha/rank ratio → weaker fine-tune influence
```

### 6.3 Underfitting Solutions

```
Symptoms: train_loss > 1.0 and plateau, model gives generic responses

Quick Fixes (try in order):
1. Increase num_train_epochs → train for 2–3 epochs
2. Increase learning_rate → try 2× current value
3. Increase rank r → from 8 → 16 → 32
4. Increase alpha → keep alpha = 2 × r
5. Add target_modules → ensure all attention+MLP layers are covered
6. Decrease batch_size → more frequent, noisier updates can help

Data Fixes:
7. Check data format → is it in the correct Alpaca/ShareGPT format?
8. Add more relevant training examples
9. Use higher quality examples (prefer quality over quantity)
```

---

## 7. Complete Unsloth Fine-Tuning Code Walkthrough

### 7.1 End-to-End Fine-Tuning Pipeline

```python
# pip install unsloth
# On Colab: use Unsloth's pre-built notebooks at https://unsloth.ai/docs

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ══════════════════════════════════════════════════════════════════
# STEP 1: Load Model (QLoRA — 4-bit quantized for VRAM efficiency)
# ══════════════════════════════════════════════════════════════════
MAX_SEQ_LENGTH = 2048   # controls max training context length
DTYPE = None            # auto-detect: bf16 on Ampere+, fp16 on older
LOAD_IN_4BIT = True     # QLoRA = True, LoRA 16-bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit",
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = DTYPE,
    load_in_4bit   = LOAD_IN_4BIT,
)

print(f"Model loaded. Trainable params before LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ══════════════════════════════════════════════════════════════════
# STEP 2: Attach LoRA Adapters
# ══════════════════════════════════════════════════════════════════
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,          # Rank — multiple of 8
    lora_alpha     = 32,          # Alpha — 2× rank (aggressive learning)
    target_modules = [            # Apply to ALL linear layers (best practice)
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
        "gate_proj", "up_proj", "down_proj",       # MLP/FFN
    ],
    lora_dropout             = 0,          # 0 = Unsloth-optimized fast path
    bias                     = "none",     # No bias training
    use_gradient_checkpointing = "unsloth",# Saves 30% extra VRAM
    random_state             = 3407,       # Reproducibility seed
    use_rslora               = False,      # Standard LoRA scaling
    loftq_config             = None,       # No LoftQ initialization
)

print(f"Trainable params after LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
# Expect ~0.3% of total parameters — e.g. ~24M out of 8B

# ══════════════════════════════════════════════════════════════════
# STEP 3: Prepare Dataset
# ══════════════════════════════════════════════════════════════════
# Using Alpaca format dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

ALPACA_TEMPLATE = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

def format_example(example):
    """Format a single Alpaca example into the prompt template."""
    text = ALPACA_TEMPLATE.format(
        instruction = example["instruction"],
        input       = example["input"] if example["input"] else "",
        output      = example["output"],
    )
    return {"text": text}

dataset = dataset.map(format_example, batched=False)
# Split 90/10 train/validation
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"Training examples: {len(train_dataset)}, Validation: {len(eval_dataset)}")

# ══════════════════════════════════════════════════════════════════
# STEP 4: Configure Training
# ══════════════════════════════════════════════════════════════════
training_args = TrainingArguments(
    output_dir                  = "./unsloth-lora-output",
    num_train_epochs            = 2,        # 1–3 recommended; watch val loss
    per_device_train_batch_size = 2,        # Low to save VRAM
    gradient_accumulation_steps = 8,        # Effective batch = 2 × 8 = 16
    learning_rate               = 2e-4,     # Unsloth recommended starting point
    weight_decay                = 0.01,     # L2 regularization
    lr_scheduler_type           = "cosine", # Smooth LR annealing
    warmup_ratio                = 0.05,     # 5% steps for LR warmup
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    logging_steps               = 10,       # Log train loss every 10 steps
    evaluation_strategy         = "steps",
    eval_steps                  = 100,      # Compute val loss every 100 steps
    save_strategy               = "steps",
    save_steps                  = 100,
    load_best_model_at_end      = True,     # Restore best checkpoint after training
    metric_for_best_model       = "eval_loss",
    report_to                   = "none",   # or "wandb" for tracking
    seed                        = 3407,
)

trainer = SFTTrainer(
    model            = model,
    tokenizer        = tokenizer,
    train_dataset    = train_dataset,
    eval_dataset     = eval_dataset,
    dataset_text_field = "text",
    max_seq_length   = MAX_SEQ_LENGTH,
    args             = training_args,
)

# ── Optional: Train on completions only ────────────────────────────
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part    = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# ══════════════════════════════════════════════════════════════════
# STEP 5: Train!
# ══════════════════════════════════════════════════════════════════
print("Starting training...")
trainer_stats = trainer.train()

print(f"\nTraining complete!")
print(f"  Time:  {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"  Steps: {trainer_stats.metrics['train_steps_per_second']:.2f} steps/sec")

# ══════════════════════════════════════════════════════════════════
# STEP 6: Test the Fine-Tuned Model
# ══════════════════════════════════════════════════════════════════
FastLanguageModel.for_inference(model)  # Enables Unsloth's 2× faster inference

test_prompt = ALPACA_TEMPLATE.format(
    instruction = "Summarize the key benefits of QLoRA fine-tuning.",
    input       = "",
    output      = "",  # leave blank — model fills this in
)

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens = 256,
    temperature    = 0.7,
    top_p          = 0.9,
    use_cache      = True,
)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"\nModel response:\n{response}")

# ══════════════════════════════════════════════════════════════════
# STEP 7: Save the LoRA Adapter
# ══════════════════════════════════════════════════════════════════
model.save_pretrained("./my-lora-adapter")        # Saves ~50–200MB adapter files
tokenizer.save_pretrained("./my-lora-adapter")

# ── Push to Hugging Face Hub ──────────────────────────────────────
# model.push_to_hub("your-hf-username/my-lora-adapter", token="hf_...")
# tokenizer.push_to_hub("your-hf-username/my-lora-adapter", token="hf_...")

# ══════════════════════════════════════════════════════════════════
# STEP 8: Export to GGUF for Ollama / llama.cpp
# ══════════════════════════════════════════════════════════════════
model.save_pretrained_gguf(
    "my-model-gguf",
    tokenizer,
    quantization_method = "q4_k_m"  # 4-bit quantization — good quality/size balance
)
# Creates: my-model-gguf/unsloth.Q4_K_M.gguf
# Use with Ollama: ollama create my-model -f Modelfile
```

### 7.2 Monitoring Metrics with Weights & Biases

```python
import wandb

# Initialize W&B project
wandb.init(project="my-llm-finetune", name="llama3-lora-run1")

training_args = TrainingArguments(
    ...
    report_to     = "wandb",          # ← enable W&B logging
    logging_steps = 1,                # log every step for detailed curves
)

# After training, access at: https://wandb.ai/your-username/my-llm-finetune
# Key charts to watch:
#   train/loss          → should decrease steadily
#   eval/loss           → should track train/loss
#   train/learning_rate → should follow cosine/linear schedule
```

---

## 8. Spring AI Equivalent — Consuming a Fine-Tuned Model

> Fine-tuning is always done in Python. Once the model is saved and deployed, Spring AI consumes it identically to any other model — just a configuration change.

### 8.1 Using an OpenAI Fine-Tuned Model

After completing OpenAI fine-tuning via `openai api fine_tuning.jobs.create`:

```properties
# application.properties
spring.ai.openai.api-key=${OPENAI_API_KEY}
# Replace with your actual fine-tuned model ID from OpenAI dashboard
spring.ai.openai.chat.options.model=ft:gpt-4o-mini:acme-corp:support-bot:abc12345
```

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.stereotype.Service;

@Service
public class FineTunedChatService {

    private final ChatClient chatClient;

    public FineTunedChatService(ChatClient.Builder builder) {
        this.chatClient = builder
            .defaultSystem("""
                You are a specialized assistant fine-tuned on Acme Corp's
                customer support data. Answer only product-related queries.
                Be concise and helpful.
                """)
            .defaultOptions(ChatOptions.builder()
                .temperature(0.3)          // lower temp for factual support
                .maxTokens(512)
                .build())
            .build();
    }

    public String answer(String customerQuery) {
        return chatClient.prompt()
            .user(customerQuery)
            .call()
            .content();
    }
}
```

### 8.2 Using Ollama with a GGUF Fine-Tuned Model

After running `unsloth.save_pretrained_gguf(...)` and importing into Ollama:

```bash
# Create Ollama Modelfile
cat > Modelfile << 'EOF'
FROM ./my-model-gguf/unsloth.Q4_K_M.gguf

SYSTEM """You are a specialized assistant fine-tuned for customer support."""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
EOF

ollama create acme-support-bot -f Modelfile
ollama run acme-support-bot  # test locally
```

```xml
<!-- pom.xml -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-ollama-spring-boot-starter</artifactId>
</dependency>
```

```properties
# application.properties
spring.ai.ollama.base-url=http://localhost:11434
spring.ai.ollama.chat.options.model=acme-support-bot
spring.ai.ollama.chat.options.temperature=0.3
```

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.*;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import java.util.List;
import java.util.ArrayList;

@Service
public class OllamaFineTunedService {

    private final ChatClient chatClient;

    // Maintain conversation history for multi-turn chat
    private final List<Message> history = new ArrayList<>();

    public OllamaFineTunedService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    // Single-turn query
    public String query(String prompt) {
        return chatClient.prompt()
            .user(prompt)
            .call()
            .content();
    }

    // Multi-turn chat with history
    public String chat(String userMessage) {
        history.add(new UserMessage(userMessage));

        String response = chatClient.prompt()
            .messages(history)
            .call()
            .content();

        history.add(new AssistantMessage(response));
        return response;
    }

    // Streaming response (token-by-token)
    public Flux<String> streamQuery(String prompt) {
        return chatClient.prompt()
            .user(prompt)
            .stream()
            .content();
    }
}
```

### 8.3 Full Spring Boot Application Example

```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
import org.springframework.http.MediaType;

@SpringBootApplication
public class FineTunedModelApp {
    public static void main(String[] args) {
        SpringApplication.run(FineTunedModelApp.class, args);
    }
}

@RestController
@RequestMapping("/api/support")
class SupportController {

    private final OllamaFineTunedService service;

    SupportController(OllamaFineTunedService service) {
        this.service = service;
    }

    // POST /api/support/query  body: "How do I reset my password?"
    @PostMapping(consumes = MediaType.TEXT_PLAIN_VALUE)
    public String query(@RequestBody String question) {
        return service.query(question);
    }

    // POST /api/support/chat  body: "What is your return policy?"
    @PostMapping(value = "/chat", consumes = MediaType.TEXT_PLAIN_VALUE)
    public String chat(@RequestBody String message) {
        return service.chat(message);  // maintains conversation history
    }

    // POST /api/support/stream  — Server-Sent Events for real-time output
    @PostMapping(value = "/stream",
                 produces = MediaType.TEXT_EVENT_STREAM_VALUE,
                 consumes = MediaType.TEXT_PLAIN_VALUE)
    public Flux<String> stream(@RequestBody String question) {
        return service.streamQuery(question);
    }
}
```

### 8.4 A/B Testing Base vs Fine-Tuned Model

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.stereotype.Service;

@Service
public class ModelComparisonService {

    private final ChatClient.Builder builder;

    public ModelComparisonService(ChatClient.Builder builder) {
        this.builder = builder;
    }

    public record ComparisonResult(
        String prompt,
        String baseModelResponse,
        String fineTunedResponse
    ) {}

    public ComparisonResult compare(String prompt) {
        // Base model response
        String baseResponse = builder.build().prompt()
            .options(ChatOptions.builder().model("gpt-4o-mini").build())
            .user(prompt).call().content();

        // Fine-tuned model response
        String ftResponse = builder.build().prompt()
            .options(ChatOptions.builder()
                .model("ft:gpt-4o-mini:acme-corp:support:abc123").build())
            .user(prompt).call().content();

        return new ComparisonResult(prompt, baseResponse, ftResponse);
    }
}
```

> **Python → Spring AI mapping for this session:**
> | Python (Unsloth) | Spring AI (Java) |
> |-----------------|-----------------|
> | `FastLanguageModel.from_pretrained(...)` | `spring.ai.*.chat.options.model=...` in properties |
> | `FastLanguageModel.for_inference(model)` | No equivalent — Spring AI is always in inference mode |
> | `model.generate(**inputs, max_new_tokens=256)` | `chatClient.prompt().user(...).call().content()` |
> | Streaming with `TextIteratorStreamer` | `chatClient.prompt().user(...).stream().content()` → `Flux<String>` |
> | `model.save_pretrained_gguf(...)` → Ollama | `spring.ai.ollama.chat.options.model=my-model` |
> | `trainer.train()` | N/A — training is Python-only |

---

## 9. Key Takeaways & Cheat Sheet

### 9.1 The 3-Stage Training Mental Model

| Stage | Goal | Method | Analogy |
|-------|------|--------|---------|
| Pre-training | Knowledge | Next-token prediction | Reading every book in a library |
| SFT | Interaction | Supervised on Q&A pairs | Taking a communication course |
| RLHF | Alignment | Reward model + RL | Passing a professional ethics exam |

### 9.2 Unsloth Fine-Tuning Checklist

```
□ 1. Load model with FastLanguageModel.from_pretrained()
□ 2. Attach LoRA with get_peft_model() — target ALL layers
□ 3. Prepare dataset in Alpaca or ChatML format
□ 4. Enable train_on_responses_only for better accuracy
□ 5. Set effective_batch_size = batch_size × grad_accum = 16
□ 6. Start with r=16, alpha=32, lr=2e-4, epochs=1–3
□ 7. Enable validation split — watch val_loss closely
□ 8. Watch: if val_loss rises while train_loss drops → STOP (overfitting)
□ 9. Save adapter and test inference
□ 10. Export to GGUF for Ollama or push to HuggingFace Hub
```

### 9.3 Hyperparameter Quick-Start Defaults

```python
# Copy-paste safe starting point for most fine-tuning tasks:
r              = 16
lora_alpha     = 32          # = r × 2
lora_dropout   = 0
learning_rate  = 2e-4
batch_size     = 2
grad_accum     = 8           # effective = 2 × 8 = 16
num_epochs     = 2
weight_decay   = 0.01
scheduler      = "cosine"
warmup_ratio   = 0.05
max_seq_length = 2048
seed           = 3407
```

### 9.4 Loss Diagnostic Quick Reference

| You See | You Do |
|--------|--------|
| `train_loss` > 1.5 after 100 steps | Check data format, increase LR |
| Both losses plateauing at high value | More data, increase rank |
| `val_loss` rising while `train_loss` falls | Stop early, reduce epochs |
| Loss oscillating / spiky | Decrease LR by 10× |
| Loss = NaN | LR too high or data contains NaN tokens |
| `train_loss` < 0.2 | Risk of overfitting — check val_loss |

### 9.5 Key URLs

| Resource | Link |
|---------|------|
| Unsloth Docs | https://unsloth.ai/docs |
| Fine-tuning Guide | https://unsloth.ai/docs/get-started/fine-tuning-llms-guide |
| Hyperparameters Guide | https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide |
| Unsloth Notebooks (Colab) | https://docs.unsloth.ai/get-started/unsloth-notebooks |
| Spring AI Docs | https://docs.spring.io/spring-ai/reference/ |

---

> **Next Session Preview:** Gen-AI Developer classroom notes for 10 Aug 2026 — likely covering model deployment, GGUF/vLLM serving, or RAG with fine-tuning.

---

*Notes compiled with deep-dive explanations, examples, and use cases based on:*
*https://directai.blog/2026/08/08/gen-ai-developer-classroom-notes-08-aug-2026/*
