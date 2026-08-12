# DAY-4: Gen-AI Developer Classroom Notes — 03 Aug 2026

> **Source:** https://directai.blog/2026/08/03/gen-ai-developer-classroom-notes-03-aug-2026/
> **Audience:** Gen-AI Developers (Beginner → Intermediate)

---

## Table of Contents

1. [Running a Small LLM Locally / Colab](#1-running-a-small-llm-locally--colab)
2. [Running a Translator (Seq2Seq Transformer)](#2-running-a-translator-seq2seq-transformer)
3. [What Problems Do Transformers Solve?](#3-what-problems-do-transformers-solve)
4. [Transformers in Plain English — Step-by-Step](#4-transformers-in-plain-english--step-by-step)
5. [Self-Attention — The Math Behind the Magic](#5-self-attention--the-math-behind-the-magic)
6. [Multi-Head Attention](#6-multi-head-attention)
7. [Softmax — The Final Probability Step](#7-softmax--the-final-probability-step)
8. [Full Transformer Pipeline (End-to-End)](#8-full-transformer-pipeline-end-to-end)
9. [Java (Spring AI) Equivalents](#9-java-spring-ai-equivalents)
10. [Key Takeaways & Cheat Sheet](#10-key-takeaways--cheat-sheet)

---

## 1. Running a Small LLM Locally / Colab

### Why Qwen3-0.6B?

`Qwen/Qwen3-0.6B` is a **0.6 billion parameter** instruct model by Alibaba — small enough to run on a free Google Colab GPU (T4) or even a modern CPU. It demonstrates the full transformer pipeline in a single runnable script.

### Step-by-Step Breakdown of the Code

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Step 1: Choose a model ─────────────────────────────────────────
model_name = "Qwen/Qwen3-0.6B"

# ── Step 2: Load tokenizer ─────────────────────────────────────────
# The tokenizer converts text → token IDs and back
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ── Step 3: Load model weights ─────────────────────────────────────
# torch_dtype="auto" → uses bf16/fp16 on GPU, fp32 on CPU (saves memory)
# device_map="auto"  → automatically puts layers on GPU/CPU as available
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# ── Step 4: Create prompt in chat format ───────────────────────────
prompt = "Explain Democracy in simple words."
messages = [
    {"role": "user", "content": prompt}
]

# apply_chat_template wraps the prompt in the model's expected format:
# e.g., "<|im_start|>user\nExplain Democracy...<|im_end|>\n<|im_start|>assistant\n"
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,            # return string, not token IDs yet
    add_generation_prompt=True # add the "assistant:" trigger at the end
)

# ── Step 5: Tokenize the formatted text ────────────────────────────
# Returns a dict with input_ids (token IDs) and attention_mask
inputs = tokenizer(text, return_tensors="pt").to(model.device)
# input_ids shape: [1, sequence_length]

# ── Step 6: Generate tokens (the actual inference) ─────────────────
outputs = model.generate(
    **inputs,
    max_new_tokens=200,   # generate up to 200 new tokens
    temperature=0.7,      # balanced creativity
    top_p=0.9,            # nucleus sampling: top 90% probability mass
    do_sample=True        # enable sampling (False = greedy)
)
# outputs shape: [1, original_length + generated_length]

# ── Step 7: Decode only the NEW tokens (skip the input) ────────────
# outputs[0][inputs.input_ids.shape[1]:]  → slice off input tokens
response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True  # remove <EOS>, <PAD>, etc.
)

print(response)
```

### What Each Parameter Does

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `torch_dtype="auto"` | bf16 on GPU | Halves memory usage vs fp32 |
| `device_map="auto"` | GPU if available | Auto-distributes layers across devices |
| `max_new_tokens=200` | 200 | Hard cap on generated output length |
| `temperature=0.7` | 0.7 | Balanced — not too creative, not too rigid |
| `top_p=0.9` | 0.9 | Nucleus sampling — ignore tokens below 10th percentile |
| `do_sample=True` | True | Enable probabilistic sampling (vs greedy) |
| `skip_special_tokens=True` | True | Strip `<EOS>`, `<PAD>` from human-readable output |

### Why `apply_chat_template`?

Different instruct models expect different prompt formats. The template handles this automatically:

```
Qwen format:
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Explain Democracy in simple words.<|im_end|>
<|im_start|>assistant
← generation starts here

Llama format:
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Explain Democracy in simple words.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

ChatML format (used by many):
<|im_start|>user
Explain Democracy in simple words.<|im_end|>
<|im_start|>assistant
```

> **Key Insight:** `apply_chat_template` is the bridge between a human-readable conversation and the raw token sequence the model expects. Without it, instruct models give poor or incoherent responses.

### Memory Requirements

| Model Size | GPU RAM Needed (fp16) | Example Hardware |
|-----------|----------------------|-----------------|
| 0.6B | ~1.2 GB | Free Colab T4, laptop GPU |
| 3B | ~6 GB | RTX 3060, Colab T4 |
| 7B | ~14 GB | RTX 3090, A10 |
| 13B | ~26 GB | A100 (40GB) |
| 70B | ~140 GB | 2× A100 80GB |

### Extended Version — Multi-Turn Conversation

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Multi-turn: add system message + conversation history
messages = [
    {"role": "system", "content": "You are a concise and clear teacher."},
    {"role": "user",   "content": "What is Democracy?"},
    # You would append model response and next user message here for multi-turn
]

def generate_response(messages, max_new_tokens=200):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():   # disable gradient tracking — saves memory during inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

response = generate_response(messages)
print(response)
# Append to conversation
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "Give me one real-world example."})
print(generate_response(messages))
```

---

## 2. Running a Translator (Seq2Seq Transformer)

### What Is Seq2Seq?

A **Sequence-to-Sequence** model is a Transformer with both an **Encoder** and a **Decoder**:

```
Encoder: reads and understands the full input sequence
Decoder: generates the output sequence token by token

English Input: "Hello, how are you?"
      │
      ▼
┌─────────────┐
│   Encoder   │  → contextual representation of full sentence
└──────┬──────┘
       │ (cross-attention)
┌──────▼──────┐
│   Decoder   │  → generates Hindi tokens one by one
└─────────────┘
      │
      ▼
Hindi Output: "नमस्ते, आप कैसे हैं?"
```

> **Unlike decoder-only LLMs** (GPT, Llama, Gemini) which just predict the next token, Seq2Seq models can generate a **completely different sequence** in a different language, length, or structure.

### The Translation Code — Explained

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Helsinki-NLP models cover 1,000+ language pairs
# Naming convention: opus-mt-{source}-{target}
# en-hi = English to Hindi
# en-fr = English to French
# de-en = German to English
model_name = "Helsinki-NLP/opus-mt-en-hi"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Seq2Seq, NOT CausalLM

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

# generate() for Seq2Seq uses beam search by default (not sampling)
# Beam search explores multiple parallel paths → better translation quality
outputs = model.generate(**inputs)

translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translation)
# "नमस्ते, आप कैसे हैं?"
```

### Causal LM vs Seq2Seq LM

| Aspect | `AutoModelForCausalLM` | `AutoModelForSeq2SeqLM` |
|--------|----------------------|------------------------|
| **Architecture** | Decoder-only | Encoder + Decoder |
| **Use case** | Text generation, chatbots | Translation, summarization |
| **Generation** | Predicts next token in same sequence | Generates a new, separate sequence |
| **Attention** | Masked self-attention | Cross-attention between encoder & decoder |
| **Examples** | GPT-4, Llama, Qwen | Helsinki opus-mt, BART, T5 |

### More Language Pairs

```python
from transformers import pipeline

# Easy one-liner translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
result = translator("Artificial Intelligence is changing the world.")
print(result[0]["translation_text"])
# "L'intelligence artificielle change le monde."

# Summarization (also Seq2Seq — BART model)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = """
    Transformers are a type of neural network architecture that was introduced
    in the paper 'Attention Is All You Need' by Vaswani et al. in 2017.
    They have since become the dominant architecture for natural language
    processing tasks and have enabled the development of large language models
    like GPT-4, Claude, and Gemini.
"""
summary = summarizer(text, max_length=50, min_length=20)
print(summary[0]["summary_text"])
```

---

## 3. What Problems Do Transformers Solve?

Transformers are the **general-purpose engine** behind almost every modern AI task. The same architecture — with different training — solves all of these:

### Problem 1: Text Generation

**What:** Generate coherent, contextually relevant text from a prompt.

```
Input:  "Once upon a time in a kingdom far away,"
Output: "...there lived a young engineer who discovered the power of transformers
         and used them to build intelligent systems that helped millions of people."
```

**Use Cases:** Chatbots, story writing, code completion, email drafting, documentation generation

**Models:** GPT-4o, Llama 3, Qwen, Gemini

---

### Problem 2: Summarization

**What:** Compress long documents into short, accurate summaries.

```
Input:  10-page research paper on climate change
Output: 3-sentence executive summary
```

**Use Cases:** News summarization, legal document review, meeting transcription → action items, research paper digests

**Models:** BART, T5, Pegasus, GPT-4

---

### Problem 3: Sentiment Analysis

**What:** Classify text as Positive / Negative / Neutral (or more fine-grained emotions).

```
Input:  "The product broke after 2 days. Terrible quality."
Output: NEGATIVE (confidence: 0.99)
```

**Use Cases:** Product review monitoring, social media brand tracking, customer support ticket routing, stock market sentiment

**Models:** BERT, DistilBERT, RoBERTa, GPT-4 (few-shot)

---

### Problem 4: Translation

**What:** Convert text from one language to another while preserving meaning and tone.

```
Input (English):  "The transformer architecture revolutionized NLP."
Output (Spanish): "La arquitectura transformer revolucionó el NLP."
```

**Use Cases:** Multilingual apps, global customer support, document localization, real-time interpreter apps

**Models:** Helsinki-NLP opus-mt, NLLB-200 (Meta), mBART, GPT-4

---

### Problem 5: Image Captioning

**What:** Generate a natural language description of an image.

```
Input:  [Image of a dog running on a beach]
Output: "A golden retriever runs joyfully along a sandy beach at sunset."
```

**How It Works:**
```
Image
  │
  ▼
Vision Encoder (ViT / CNN)    ← extracts visual features
  │
  ▼
Cross-Attention
  │
  ▼
Language Decoder (Transformer) ← generates caption token by token
  │
  ▼
Caption text
```

**Use Cases:** Accessibility (alt-text generation), image search indexing, social media auto-tagging, medical image reports

**Models:** BLIP-2, LLaVA, Gemini Vision, GPT-4V

### Complete Transformer Task Map

```
                    TRANSFORMER
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   Text Only      Text + Image    Text + Audio
        │               │               │
   ┌────┴────┐      ┌───┴───┐      ┌───┴───┐
   │Generate │      │Caption│      │Transcr│
   │Summarize│      │VQA    │      │ibe    │
   │Classify │      │OCR    │      │Dub    │
   │Translate│      │       │      │       │
   │QA / NER │      │       │      │       │
   └─────────┘      └───────┘      └───────┘
```

---

## 4. Transformers in Plain English — Step-by-Step

> 🔗 **Interactive visualization:** [transformer-explainer](https://poloclub.github.io/transformer-explainer/)

**The core contract:**
- **Input:** A piece of text (a prompt)
- **Output:** A probability distribution over the entire vocabulary

Here is what happens at every stage — in plain English, then with the math.

---

### Stage 1: Tokenization

**Plain English:** Chop the text into small word-pieces (tokens) and give each a number.

```
Input text:  "Democracy is rule by the people."

After tokenization:
Word         Token      Token ID
"Democracy"  "Dem"      →  39608
             "ocracy"   →  16193
"is"         "is"       →  318
"rule"       "rule"     →  3896
"by"         "by"       →  416
"the"        "the"      →  262
"people"     "people"   →  661
"."          "."        →  13

Token IDs: [39608, 16193, 318, 3896, 416, 262, 661, 13]
```

**Why sub-words?** "Democracy" → ["Dem", "ocracy"] means the model never sees an unknown word — it always knows the parts.

---

### Stage 2: Embedding

**Plain English:** Replace each token ID with a long list of floats (a vector) that captures meaning.

```
Token ID 39608 ("Dem")    → [0.21, -0.45, 0.88, ..., 0.12]   (768 floats)
Token ID 16193 ("ocracy") → [0.33,  0.67, -0.22, ..., 0.44]  (768 floats)
Token ID 318   ("is")     → [0.01, -0.01,  0.03, ..., -0.02]  (768 floats)
...

Result: a matrix of shape [sequence_length × embedding_dim]
        e.g., [8 tokens × 768 floats] = 6,144 numbers total
```

**Positional Encoding is added here:**
```
final_embedding = token_embedding + positional_encoding
```
This tells the model "this 'the' is at position 5, that 'the' is at position 0" — same word, different roles.

---

### Stage 3: Self-Attention

**Plain English:** Every token looks at every other token and decides how much to "pay attention" to it when computing its own meaning.

```
"The animal didn't cross the street because it was too tired."

When computing the meaning of "it":
  Attention to "animal" → 0.72  ← HIGH — "it" refers to "animal"
  Attention to "street" → 0.11
  Attention to "cross"  → 0.08
  Attention to "tired"  → 0.09
```

This is what makes Transformers understand **long-range dependencies** — connections across the entire sentence, not just adjacent words.

---

### Stage 4: Multi-Head Attention

**Plain English:** Run self-attention MULTIPLE TIMES in parallel, each looking for different types of relationships.

```
Head 1 → focuses on subject-verb agreement ("animal ... was tired")
Head 2 → focuses on pronoun resolution ("it" → "animal")
Head 3 → focuses on negation ("didn't cross")
Head 4 → focuses on causal relationships ("because")
...
Head N → each learns something different automatically during training

All heads concatenated → full rich contextual representation
```

---

### Stage 5: Feed-Forward Network (FFN)

**Plain English:** After attention collects context from all positions, a small neural network processes each token's representation independently.

```
For each token position:
  hidden = ReLU(W1 × attention_output + b1)
  output = W2 × hidden + b2

This is where the model "thinks" about what it has learned from attention
and transforms it into a richer representation.
```

Stages 3–5 form one **Transformer Block**. Modern LLMs stack this block many times:
- BERT-base: 12 blocks
- GPT-3: 96 blocks
- GPT-4: estimated 120+ blocks

---

### Stage 6: Output — Probability Distribution via Softmax

**Plain English:** After all the layers, the final vector is projected onto the vocabulary and converted into probabilities — a score for every possible next word.

---

## 5. Self-Attention — The Math Behind the Magic

This is the most important formula in all of modern AI.

### Q, K, V — What They Are

Every token's embedding is projected into **three vectors** using learned weight matrices:

```
For each token embedding x:

Q (Query)  = x × W_Q     ← "What am I looking for?"
K (Key)    = x × W_K     ← "What do I contain?"
V (Value)  = x × W_V     ← "What information do I pass on?"
```

**Analogy — Library search:**
```
You (Q) search for "books about democracy"
Each book has a Title tag (K) describing its content
Attention score = how well your query matches each book's title
V = the actual content of the book you get to read
```

### The Self-Attention Formula

```
                    Q × K^T
Attention(Q,K,V) = Softmax(─────────) × V
                    √d_k

Where:
  Q  = Query matrix  [seq_len × d_k]
  K  = Key matrix    [seq_len × d_k]
  V  = Value matrix  [seq_len × d_v]
  d_k = dimension of keys (e.g., 64)
  √d_k = scaling factor to prevent vanishingly small gradients
```

### Step-by-Step Calculation

```
Step 1: Compute raw attention scores
  scores = Q × K^T          shape: [seq_len × seq_len]
  Each cell (i,j) = how much token i attends to token j

  Example (4 tokens):
        "The"  "cat"  "sat"  "on"
  "The" [ 1.2,  0.3,  0.1,  0.2 ]
  "cat" [ 0.4,  2.1,  0.8,  0.1 ]
  "sat" [ 0.2,  1.5,  1.8,  0.3 ]
  "on"  [ 0.3,  0.2,  0.4,  1.1 ]

Step 2: Scale by √d_k (e.g., √64 = 8)
  scaled_scores = scores / 8

Step 3: Apply Softmax (converts to probabilities, row-wise)
  attention_weights = Softmax(scaled_scores)
  Each row now sums to 1.0

  "cat" row: [0.08, 0.55, 0.28, 0.09]
              ↑     ↑     ↑
              The  cat   sat
              (cat attends mostly to itself and "sat")

Step 4: Weighted sum of Values
  output = attention_weights × V
  Each token gets a new vector = weighted combination of all value vectors
```

### 🐍 Python — Self-Attention from Scratch
```python
import torch
import torch.nn.functional as F

def self_attention(Q, K, V):
    """
    Q: [batch, seq_len, d_k]
    K: [batch, seq_len, d_k]
    V: [batch, seq_len, d_v]
    """
    d_k = Q.shape[-1]

    # Step 1 & 2: Scaled dot-product scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    # scores shape: [batch, seq_len, seq_len]

    # Step 3: Softmax → attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4: Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

# Demo with 4 tokens, d_k = 8
batch, seq_len, d_k = 1, 4, 8
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_k)

output, weights = self_attention(Q, K, V)
print(f"Output shape: {output.shape}")   # [1, 4, 8]
print(f"Attention weights (token 0):\n{weights[0, 0]}")
# Each value = how much token 0 attends to each other token
```

---

## 6. Multi-Head Attention

### Why Multiple Heads?

A single attention head can only look for **one type of relationship** at a time. Multiple heads run in parallel, each initialized differently, learning to capture different aspects simultaneously.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

Where each head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)
```

### Visualized

```
Input Embeddings
      │
      ├─────────────────────────────────────────┐
      │                                         │
   Head 1 (W_Q1, W_K1, W_V1)    ...   Head H (W_QH, W_KH, W_VH)
   Attention output_1                  Attention output_H
      │                                         │
      └────────────────┬────────────────────────┘
                       │
                  Concatenate
                       │
                  Linear(W_O)
                       │
                Multi-Head Output
```

### 🐍 Python — Multi-Head Attention (PyTorch built-in)
```python
import torch
import torch.nn as nn

# PyTorch's built-in MultiheadAttention
# embed_dim=512, num_heads=8 → each head works in 512/8=64 dims
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

# [batch=1, seq_len=10, embed_dim=512]
x = torch.randn(1, 10, 512)

# Self-attention: Q=K=V=x
output, attention_weights = mha(x, x, x)

print(f"Output shape:           {output.shape}")          # [1, 10, 512]
print(f"Attention weights shape: {attention_weights.shape}") # [1, 10, 10]
# attention_weights[0, i, j] = how much token i attends to token j
```

---

## 7. Softmax — The Final Probability Step

### What Softmax Does

Softmax converts a vector of raw scores (logits) into a valid **probability distribution** where all values are between 0 and 1 and sum to exactly 1.

### The Formula

```
         e^(x_i)
softmax(x_i) = ─────────────
               Σ_j e^(x_j)

Example:
Raw logits (model's unnormalized scores for next token):
  "sat"   = 2.1
  "ran"   = 1.4
  "ate"   = 1.8
  "was"   = 0.5

After softmax:
  e^2.1 = 8.17,  e^1.4 = 4.06,  e^1.8 = 6.05,  e^0.5 = 1.65
  Sum = 19.93

  P("sat") = 8.17 / 19.93 = 0.41  ← 41% chance
  P("ran") = 4.06 / 19.93 = 0.20
  P("ate") = 6.05 / 19.93 = 0.30
  P("was") = 1.65 / 19.93 = 0.08
  Sum      =                 0.99  ≈ 1.0 ✅
```

### Softmax in Two Places in a Transformer

| Where | What It Does |
|-------|-------------|
| **Inside Self-Attention** | Converts attention scores to weights (rows sum to 1) |
| **Final Output Layer** | Converts logits over vocabulary to token probabilities |

### 🐍 Python — Softmax Explained
```python
import torch
import torch.nn.functional as F

# Raw logit scores from the model's final layer
# (one score per token in vocabulary — here simplified to 5 tokens)
logits = torch.tensor([2.1, 1.4, 1.8, 0.5, -0.3])

# Apply softmax → probabilities
probs = F.softmax(logits, dim=0)
print("Probabilities:", probs)
print("Sum:", probs.sum().item())  # always 1.0

# Effect of temperature on softmax
def softmax_with_temperature(logits, temperature):
    scaled = logits / temperature
    return F.softmax(scaled, dim=0)

tokens = ["sat", "ran", "ate", "was", "jumped"]
for temp in [0.1, 0.7, 1.0, 2.0]:
    probs = softmax_with_temperature(logits, temp)
    print(f"\nT={temp}:")
    for token, p in zip(tokens, probs):
        bar = "█" * int(p * 30)
        print(f"  {token:8s}: {p:.3f} {bar}")

# T=0.1: sat=0.997 (near certain)
# T=2.0: sat=0.289 (more uniform — creative)
```

---

## 8. Full Transformer Pipeline (End-to-End)

```
┌──────────────────────────────────────────────────────────────────┐
│              COMPLETE TRANSFORMER FORWARD PASS                   │
│                                                                  │
│  Input Text: "Explain Democracy in simple words."                │
│                        │                                         │
│                        ▼                                         │
│  ① TOKENIZATION                                                  │
│     Text → Token IDs: [39608, 16193, 318, ...]                  │
│                        │                                         │
│                        ▼                                         │
│  ② EMBEDDING + POSITIONAL ENCODING                               │
│     Token IDs → Float vectors (e.g., 768-dim each)              │
│     + Position signals added                                     │
│     Shape: [seq_len × 768]                                       │
│                        │                                         │
│      ┌─────────────────┘                                         │
│      │   Repeat N times (N = number of layers)                   │
│      ▼                                                           │
│  ③ MULTI-HEAD SELF-ATTENTION (Q, K, V)                          │
│     Every token looks at every other token                       │
│     Compute: Softmax(QK^T / √d_k) × V                          │
│     Output: context-enriched vectors                             │
│                        │                                         │
│                        ▼                                         │
│  ④ FEED-FORWARD NETWORK (per token)                             │
│     Apply 2-layer MLP to each position                           │
│                        │                                         │
│      └─────────────────┘ (back to ③ for next layer)             │
│                        │                                         │
│                        ▼                                         │
│  ⑤ FINAL LINEAR PROJECTION                                       │
│     Map last hidden state → logits over vocabulary               │
│     Shape: [seq_len × vocab_size] (e.g., × 100,000)             │
│                        │                                         │
│                        ▼                                         │
│  ⑥ SOFTMAX → PROBABILITY DISTRIBUTION                            │
│     Logits → probabilities for each next token                   │
│                        │                                         │
│                        ▼                                         │
│  ⑦ SAMPLING (influenced by temperature)                          │
│     Pick next token → append → loop back to ①                   │
│     Until: <EOS> token OR max_new_tokens reached                 │
│                                                                  │
│  Output: "Democracy is a system of government where citizens..." │
└──────────────────────────────────────────────────────────────────┘
```

### The Inference Loop Summarized

```python
# Conceptual pseudocode of what model.generate() does internally
generated_tokens = input_token_ids[:]

for _ in range(max_new_tokens):
    # Forward pass through all transformer layers
    logits = transformer_forward(generated_tokens)       # shape: [vocab_size]

    # Apply temperature scaling
    logits = logits / temperature

    # Convert to probabilities
    probs = softmax(logits)

    # Sample next token
    next_token = sample_from(probs, top_p=0.9)          # nucleus sampling

    # Stop if EOS
    if next_token == EOS_TOKEN_ID:
        break

    # Append and continue
    generated_tokens.append(next_token)

# Decode all generated token IDs back to text
output_text = tokenizer.decode(generated_tokens[len(input_token_ids):])
```

---

## 9. Java (Spring AI) Equivalents

### Text Generation (same as running Qwen locally — but via API)

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatOptions;
import org.springframework.stereotype.Service;

@Service
public class TransformerDemoService {

    private final ChatClient chatClient;

    public TransformerDemoService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    // Equivalent to running Qwen locally — same transformer pipeline,
    // just executed on Google's TPUs instead of your GPU
    public String explainConcept(String concept) {
        return chatClient.prompt()
                .system("You are a concise and clear teacher.")
                .user("Explain " + concept + " in simple words.")
                .options(VertexAiGeminiChatOptions.builder()
                        .temperature(0.7)
                        .topP(0.9)
                        .maxOutputTokens(200)
                        .build())
                .call()
                .content();
    }
}
```

### Translation via Spring AI (LLM-based)

```java
@Service
public class TranslationService {

    private final ChatClient chatClient;

    public TranslationService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public String translate(String text, String targetLanguage) {
        return chatClient.prompt()
                .system("You are a professional translator. Translate accurately and naturally.")
                .user("Translate the following text to " + targetLanguage
                        + ". Return only the translation, nothing else.\n\nText: " + text)
                .options(VertexAiGeminiChatOptions.builder()
                        .temperature(0.2)   // low temp → consistent, accurate translation
                        .build())
                .call()
                .content();
    }

    public String summarize(String text) {
        return chatClient.prompt()
                .user("Summarize the following text in 2-3 sentences:\n\n" + text)
                .options(VertexAiGeminiChatOptions.builder()
                        .temperature(0.3)
                        .maxOutputTokens(150)
                        .build())
                .call()
                .content();
    }
}
```

### Sentiment Analysis via Spring AI (Structured Output)

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

@Service
public class SentimentService {

    private final ChatClient chatClient;

    // Inner record for structured response
    record SentimentResult(String label, double confidence, String reasoning) {}

    public SentimentService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public SentimentResult analyze(String text) {
        // Spring AI can map response directly to a Java record
        return chatClient.prompt()
                .user("""
                    Analyze the sentiment of this text.
                    Respond in JSON format:
                    {"label": "POSITIVE|NEGATIVE|NEUTRAL", "confidence": 0.0-1.0, "reasoning": "..."}

                    Text: %s
                    """.formatted(text))
                .call()
                .entity(SentimentResult.class);  // auto-deserialize JSON → Java record
    }
}
```

---

## 10. Key Takeaways & Cheat Sheet

### The 7 Stages of a Transformer

| Stage | Input | Output | Key Concept |
|-------|-------|--------|-------------|
| **① Tokenization** | Raw text | Token IDs | BPE sub-words, never OOV |
| **② Embedding + Position** | Token IDs | Float vectors | Semantic meaning + position |
| **③ Self-Attention** | Embeddings | Context vectors | Q·K^T/√d_k → Softmax → ×V |
| **④ Multi-Head Attention** | Embeddings | Richer context | H heads × different relations |
| **⑤ Feed-Forward** | Attention output | Transformed vectors | Per-position MLP |
| **⑥ Final Projection** | Last hidden state | Logits (vocab-size) | Linear layer |
| **⑦ Softmax + Sampling** | Logits | Next token | Temperature controls creativity |

### Formula Reference Card

```
Self-Attention:    Attention(Q,K,V) = Softmax( QK^T / √d_k ) × V

Softmax:           softmax(x_i) = e^x_i / Σ_j e^x_j

Multi-Head:        MultiHead(Q,K,V) = Concat(head_1,...,head_h) × W_O

Temperature:       scaled_logits = logits / T
                   T→0: deterministic  |  T→∞: uniform random
```

### Model Type Decision

```
Need to generate text?        → AutoModelForCausalLM   (Decoder-only)
Need to translate/summarize?  → AutoModelForSeq2SeqLM  (Encoder-Decoder)
Need embeddings/classify?     → AutoModelForMaskedLM   (Encoder-only / BERT)
Just calling an API?          → ChatClient (Spring AI) / genai (Python)
```

### Problems Transformers Solve — Quick Map

| Problem | Python Model | Java (Spring AI) |
|---------|-------------|-----------------|
| Text Generation | `AutoModelForCausalLM` / `pipeline("text-generation")` | `ChatClient.prompt().user().call()` |
| Summarization | `pipeline("summarization", model="facebook/bart-large-cnn")` | `ChatClient` with summarize prompt |
| Sentiment Analysis | `pipeline("sentiment-analysis")` | `ChatClient` + `.entity(SentimentResult.class)` |
| Translation | `pipeline("translation", model="Helsinki-NLP/opus-mt-en-*")` | `ChatClient` with translation prompt |
| Image Captioning | `pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")` | `ChatClient` + `.media(image)` |

---

## References & Further Reading

- [Transformer Explainer (Interactive)](https://poloclub.github.io/transformer-explainer/) — Must-visit visual demo
- [Attention Is All You Need (Paper)](https://arxiv.org/abs/1706.03762) — Original 2017 paper
- [Qwen3 on HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B) — The model used in class
- [Helsinki-NLP Translation Models](https://huggingface.co/Helsinki-NLP) — 1,000+ language pairs
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers) — Full API reference
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Best blog explanation
- [Spring AI Docs](https://docs.spring.io/spring-ai/reference/) — Java AI integration

---

*Notes compiled and expanded from Direct AI Blog — Classroom session 03 August 2026*
*Enhanced with deep insights, formulas, code examples (Python + Java Spring AI), and architecture diagrams*
