# DAY-3: Gen-AI Developer Classroom Notes — 30 Jul 2026

> **Source:** https://directai.blog/2026/07/30/gen-ai-developer-classroom-notes-30-jul-2026/
> **Audience:** Gen-AI Developers (Beginner → Intermediate)

---

## Table of Contents

1. [Embeddings — Deep Dive](#1-embeddings--deep-dive)
2. [Tokens — The Currency of LLMs](#2-tokens--the-currency-of-llms)
3. [LLM Pricing — Token-Based Cost Model](#3-llm-pricing--token-based-cost-model)
4. [Transformers — Full Pipeline](#4-transformers--full-pipeline)
5. [Temperature & Sampling Strategies](#5-temperature--sampling-strategies)
6. [Well-Known Training Datasets](#6-well-known-training-datasets)
7. [Pre-training vs Distillation](#7-pre-training-vs-distillation)
8. [Base Models vs Instruct Models](#8-base-models-vs-instruct-models)
9. [Exercises](#9-exercises)
10. [Key Takeaways & Cheat Sheet](#10-key-takeaways--cheat-sheet)

---

## 1. Embeddings — Deep Dive

### What Is an Embedding?

An embedding converts a **token** (piece of text) into a **vector — an array of floating-point numbers** that captures its meaning in a high-dimensional space.

```
Token  →  Embedding Vector (e.g., 768 floats)

"cat"  →  [0.21, -0.45, 0.88, 0.03, -0.67, ..., 0.12]
                                              ↑ 768 numbers
```

> Every dimension captures some abstract feature of meaning. No single dimension maps to a human-readable concept, but **together** they encode semantic relationships.

### How Embeddings Are Structured

```
Embedding Matrix (lookup table):

Vocabulary size: 50,257 tokens (GPT-4 uses ~100K)
Embedding dimension: 768 (BERT) / 1536 (GPT-3) / 4096 (Llama 3)

         dim0   dim1   dim2  ...  dim767
token_0 [0.21, -0.45,  0.88, ..., 0.12]   ← "the"
token_1 [0.88,  0.33, -0.21, ..., 0.77]   ← "cat"
token_2 [0.91,  0.31, -0.19, ..., 0.73]   ← "dog"   ← similar to "cat"
token_3 [-0.45, 0.92,  0.55, ..., -0.34]  ← "bank"
  ...
```

### Why Floating-Point Vectors?

| Why Not...? | Problem | Floats Solve It By... |
|-------------|---------|----------------------|
| Plain words | Computers can't do math on strings | Enabling distance/similarity calculations |
| One-hot integers | Sparse, no semantic relationship | Being dense — similar meanings = close vectors |
| Fixed integers (1,2,3) | Arbitrary ordering, no meaning | Capturing multidimensional semantic space |

### Cosine Similarity — Measuring "Closeness"

```
Similar words → vectors point in the same direction → high cosine similarity

cos(θ) = (A · B) / (|A| × |B|)

cat  ↔  dog    →  0.91  (very similar — both animals)
cat  ↔  table  →  0.23  (not similar)
king ↔  queen  →  0.89  (similar — royalty)
```

#### 🐍 Python — Inspect Embeddings
```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

words = ["cat", "dog", "king", "queen", "Paris", "France"]
embeddings = model.encode(words)

print(f"Embedding shape: {embeddings.shape}")
# (6, 384) → 6 words, each 384-dimensional vector

# Show first 5 floats of "cat" embedding
print(f"'cat' first 5 dims: {embeddings[0][:5]}")
# [ 0.0234 -0.1872  0.3451  0.0091 -0.2234]

# Cosine similarity between cat and dog
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"cat ↔ dog   : {cosine_sim(embeddings[0], embeddings[1]):.3f}")  # ~0.80
print(f"cat ↔ Paris : {cosine_sim(embeddings[0], embeddings[4]):.3f}")  # ~0.18
```

#### ☕ Java (Spring AI — Get Embedding Vectors)
```java
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;

@Service
public class EmbeddingInspector {

    private final EmbeddingClient embeddingClient;

    public EmbeddingInspector(EmbeddingClient embeddingClient) {
        this.embeddingClient = embeddingClient;
    }

    public void inspectEmbedding(String word) {
        EmbeddingResponse response = embeddingClient
                .embedForResponse(List.of(word));

        float[] vector = response.getResults().get(0).getOutput();
        System.out.printf("'%s' → %d-dim vector%n", word, vector.length);
        System.out.printf("First 5 values: %s%n",
                Arrays.toString(Arrays.copyOf(vector, 5)));
    }

    public double cosineSimilarity(String wordA, String wordB) {
        float[] a = embeddingClient.embedForResponse(List.of(wordA))
                .getResults().get(0).getOutput();
        float[] b = embeddingClient.embedForResponse(List.of(wordB))
                .getResults().get(0).getOutput();

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot   += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
```

---

## 2. Tokens — The Currency of LLMs

### Words ≠ Tokens

LLMs do NOT process words. They process **tokens** — sub-word units produced by a tokenizer algorithm (usually BPE — Byte Pair Encoding).

```
"unhappiness"  →  ["un", "happ", "iness"]   = 3 tokens
"cat"          →  ["cat"]                    = 1 token
"Hello, world!" → ["Hello", ",", " world", "!"] = 4 tokens
```

> **Why sub-words?** It lets the model handle **rare or new words** by breaking them into known parts, while keeping common words as single tokens for efficiency.

### Tokenization Process

```
Raw text: "The cat sat on the mat."
    │
    ▼
Tokenizer (BPE / SentencePiece)
    │
    ▼
Token IDs: [464, 3797, 3332, 319, 262, 2603, 13]
    │
    ▼
Embedding lookup → floating-point vectors
    │
    ▼
Fed into Transformer layers
```

### Token Count Intuition

| Rule of Thumb | Value |
|---------------|-------|
| 1 token | ≈ 4 characters |
| 1 token | ≈ 0.75 words |
| 100 tokens | ≈ 75 words |
| 1 page of text | ≈ 500–700 tokens |
| 1 novel (80K words) | ≈ 107K tokens |

#### 🐍 Python — Tokenize Text with tiktoken (OpenAI's tokenizer)
```python
import tiktoken

# GPT-4 uses the "cl100k_base" encoding
enc = tiktoken.get_encoding("cl100k_base")

text = "The cat sat on the mat. Embeddings are fascinating!"

tokens = enc.encode(text)
print(f"Text: '{text}'")
print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)}")
print(f"Decoded tokens: {[enc.decode([t]) for t in tokens]}")

# Token count: 11
# Decoded: ['The', ' cat', ' sat', ' on', ' the', ' mat', '.', 
#            ' Embed', 'dings', ' are', ' fascinating', '!']
```

```python
# Estimate cost before sending to API
def estimate_cost(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    token_count = len(enc.encode(text))
    # GPT-4o pricing: $5 per 1M input tokens
    cost_usd = (token_count / 1_000_000) * 5.0
    print(f"Tokens: {token_count} | Estimated cost: ${cost_usd:.6f}")
    return token_count

estimate_cost("Explain quantum computing in simple terms.")
# Tokens: 7 | Estimated cost: $0.000035
```

#### ☕ Java (Spring AI — Token-Aware Prompting)
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.stereotype.Service;

@Service
public class TokenAwareService {

    private final ChatClient chatClient;

    public TokenAwareService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public void callWithTokenTracking(String prompt) {
        // Spring AI returns token usage metadata with each response
        ChatResponse response = chatClient.prompt()
                .user(prompt)
                .call()
                .chatResponse();

        String content = response.getResult().getOutput().getContent();
        Usage usage = response.getMetadata().getUsage();

        System.out.println("Response: " + content);
        System.out.println("Input tokens  : " + usage.getPromptTokens());
        System.out.println("Output tokens : " + usage.getGenerationTokens());
        System.out.println("Total tokens  : " + usage.getTotalTokens());
    }
}
```

---

## 3. LLM Pricing — Token-Based Cost Model

All hosted LLM vendors charge **per token** — separately for input (prompt) and output (completion).

### How to Check Current Pricing

Use this prompt in any LLM:
```
Give me model pricing of all popular LLMs for API usage in a tabular form
```

### Approximate Pricing Reference (as of mid-2026)

| Model | Provider | Input (per 1M tokens) | Output (per 1M tokens) | Context Window |
|-------|----------|----------------------|----------------------|----------------|
| **GPT-4o** | OpenAI | $5.00 | $15.00 | 128K |
| **GPT-4o mini** | OpenAI | $0.15 | $0.60 | 128K |
| **o3** | OpenAI | $10.00 | $40.00 | 200K |
| **Claude 3.5 Sonnet** | Anthropic | $3.00 | $15.00 | 200K |
| **Claude 3 Haiku** | Anthropic | $0.25 | $1.25 | 200K |
| **Gemini 1.5 Pro** | Google | $3.50 | $10.50 | 1M |
| **Gemini 1.5 Flash** | Google | $0.075 | $0.30 | 1M |
| **Gemini 2.0 Flash** | Google | $0.10 | $0.40 | 1M |
| **Llama 3.1 70B** | Groq | $0.59 | $0.79 | 128K |

> **⚠️ Always verify pricing at the official vendor site** — it changes frequently.

### Cost Optimization Tips for Developers

| Strategy | How | Saving |
|----------|-----|--------|
| **Choose smaller models** | Use Flash/mini for simple tasks | 10–50x cheaper |
| **Compress prompts** | Remove filler words, be concise | 20–40% fewer input tokens |
| **Limit output tokens** | Set `max_tokens` appropriately | Avoid runaway completions |
| **Cache responses** | Same prompt → return cached result | 100% saving on repeats |
| **Batch requests** | Process multiple items in one call | Fewer API round trips |

#### 🐍 Python — Cost-Aware API Call
```python
import os
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")  # cheapest option

response = model.generate_content(
    "Summarize the benefits of embeddings in Gen-AI.",
    generation_config=genai.types.GenerationConfig(
        max_output_tokens=150  # limit output to control cost
    )
)

# Check token usage
usage = response.usage_metadata
print(f"Input tokens  : {usage.prompt_token_count}")
print(f"Output tokens : {usage.candidates_token_count}")
print(f"Total tokens  : {usage.total_token_count}")

# Estimate cost (Gemini 1.5 Flash: $0.075 input / $0.30 output per 1M)
input_cost  = (usage.prompt_token_count     / 1_000_000) * 0.075
output_cost = (usage.candidates_token_count / 1_000_000) * 0.30
print(f"Estimated cost: ${input_cost + output_cost:.8f}")
```

#### ☕ Java (Spring AI — Model Selection for Cost Control)
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatOptions;
import org.springframework.stereotype.Service;

@Service
public class CostOptimizedService {

    private final ChatClient chatClient;

    public CostOptimizedService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public String cheapSummarize(String text) {
        // Use Flash model + limit output tokens = cheapest option
        return chatClient.prompt()
                .user("Summarize in 2 sentences: " + text)
                .options(VertexAiGeminiChatOptions.builder()
                        .model("gemini-2.0-flash")   // cheapest Gemini model
                        .maxOutputTokens(100)         // cap output tokens
                        .build())
                .call()
                .content();
    }

    public String premiumAnalyze(String text) {
        // Use Pro model for complex reasoning tasks
        return chatClient.prompt()
                .user("Perform deep analysis: " + text)
                .options(VertexAiGeminiChatOptions.builder()
                        .model("gemini-1.5-pro")     // best quality
                        .maxOutputTokens(2048)
                        .build())
                .call()
                .content();
    }
}
```

---

## 4. Transformers — Full Pipeline

### The Complete Flow: Text → Token → Embedding → Prediction

```
Step 1: Tokenization
─────────────────────
Input: "The cat sat"
         │
         ▼
Tokenizer (BPE)
         │
         ▼
Token IDs: [464, 3797, 3332]  ← integers from vocabulary

Step 2: Positional Encoding
──────────────────────────────
Token IDs + Position Tags
[464/pos=0, 3797/pos=1, 3332/pos=2]
         │
         ▼
"the" at position 0 ≠ "the" at position 5
(Transformers process all tokens in parallel,
 so position must be explicitly encoded)

Step 3: Embedding Lookup
─────────────────────────
Each token ID → lookup in embedding matrix → float vector
[464]  →  [0.21, -0.45, 0.88, ...]  (768 floats)
[3797] →  [0.88,  0.33, -0.21, ...]  (768 floats)
[3332] →  [0.91,  0.31, -0.19, ...]  (768 floats)

Step 4: Transformer Layers (×N stacked)
─────────────────────────────────────────
Multi-Head Self-Attention
    ↓
Feed-Forward Network
    ↓
(Repeat for each layer — GPT-4 has 96 layers)

Step 5: Output → Probability Distribution
──────────────────────────────────────────
Final layer → logits for every token in vocabulary

Vocabulary: 100,000 tokens
P("is")   = 0.0023
P("was")  = 0.0041
P("sat")  = 0.4512  ← highest
P("ran")  = 0.1834
P("ate")  = 0.2108
...

Step 6: Sample Next Token
──────────────────────────
Pick token based on temperature:
Temperature = 0.0 → always pick "sat" (greedy)
Temperature = 1.0 → sample proportionally

Append "sat" → repeat from Step 1 with new sequence
```

### Positional Encoding — Why It Matters

Transformers process all tokens **simultaneously** (unlike RNNs which go one by one). Without position info, `"cat bit dog"` and `"dog bit cat"` would look identical.

```
"cat bit dog"  →  same tokens, different positions → different meaning
"dog bit cat"  →  same tokens, different positions → different meaning

Positional encoding adds a unique signal to each position:
pos=0: [sin(0/10000^0), cos(0/10000^0), sin(0/10000^2/512), ...]
pos=1: [sin(1/10000^0), cos(1/10000^0), sin(1/10000^2/512), ...]
```

### 🔗 Visualization Resource

> **Highly recommended:** [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — interactive, visual walkthrough of the entire transformer pipeline in your browser.

### Max Token Limits of Popular LLMs

Use this prompt to get latest limits:
```
Give me max input and output tokens of all popular LLMs in a tabular form
```

**Reference Table:**

| Model | Max Input Tokens | Max Output Tokens | Notes |
|-------|-----------------|-------------------|-------|
| **GPT-4o** | 128,000 | 16,384 | ~96,000 words input |
| **GPT-4o mini** | 128,000 | 16,384 | Same window, cheaper |
| **Claude 3.5 Sonnet** | 200,000 | 8,192 | ~150,000 words |
| **Gemini 1.5 Pro** | 1,000,000 | 8,192 | ~750,000 words! |
| **Gemini 2.0 Flash** | 1,048,576 | 8,192 | 1M token context |
| **Llama 3.1 70B** | 128,000 | 4,096 | Open source |
| **Qwen2.5 72B** | 131,072 | 8,192 | Strong multilingual |

#### 🐍 Python — Check Context Window Programmatically
```python
import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# List all available models and their token limits
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"{model.name}")
        print(f"  Input limit  : {model.input_token_limit:,} tokens")
        print(f"  Output limit : {model.output_token_limit:,} tokens")
        print()
```

#### ☕ Java (Spring AI — Prompt with context window awareness)
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatOptions;
import org.springframework.stereotype.Service;

@Service
public class LongContextService {

    private final ChatClient chatClient;

    public LongContextService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    // Gemini 1.5 Pro can handle a 1M token context window
    // — you can send an entire book as context!
    public String analyzeDocument(String largeDocument, String question) {
        return chatClient.prompt()
                .system("You are a document analysis assistant.")
                .user("""
                    Document:
                    %s

                    Question: %s
                    """.formatted(largeDocument, question))
                .options(VertexAiGeminiChatOptions.builder()
                        .model("gemini-1.5-pro")   // 1M token context
                        .build())
                .call()
                .content();
    }
}
```

---

## 5. Temperature & Sampling Strategies

### What Is Temperature?

Temperature controls **how deterministic vs creative** the token selection is. It reshapes the probability distribution before sampling.

```
Raw probability distribution after softmax:
Token    Prob
"sat"  = 0.45
"ate"  = 0.21
"ran"  = 0.18
"was"  = 0.16

Temperature = 0.0 (greedy — always pick max):
→ Always picks "sat"

Temperature = 0.7 (balanced):
→ Probabilities slightly smoothed
→ Mostly "sat", occasionally others

Temperature = 2.0 (very creative — nearly uniform):
→ All tokens roughly equal chance
→ Unexpected, creative, sometimes incoherent
```

### Temperature Visual Effect

```
T = 0.1    [████████████████░░░░]  "sat" dominates
T = 0.5    [████████░░░░░░░░░░░░]  some variety
T = 1.0    [██████░░░░░░░░░░░░░░]  balanced
T = 2.0    [████░░░░░░░░░░░░░░░░]  near-random
```

### Sampling Strategies

| Strategy | Description | Use Case |
|----------|-------------|---------|
| **Greedy** (`temp=0`) | Always pick highest probability token | SQL, JSON, code — deterministic output |
| **Top-K** | Sample from top K most probable tokens | Balanced creativity — chatbots |
| **Top-P (nucleus)** | Sample from smallest set whose cumulative prob ≥ P | Better than Top-K, adapts to distribution |
| **Temperature** | Scale logits before softmax | General creativity control |

> **Best Practice:** Use `temperature=0.2` + `top_p=0.95` for code generation; `temperature=0.9` for creative writing.

#### 🐍 Python — Temperature Comparison
```python
import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

prompt = "Complete this sentence: The robot opened its eyes and"
temperatures = [0.0, 0.5, 1.0, 1.8]

for temp in temperatures:
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temp,
            max_output_tokens=30
        )
    )
    print(f"T={temp}: {response.text.strip()}")

# T=0.0: ...saw a world of data and logic.
# T=0.5: ...began to process its surroundings slowly.
# T=1.0: ...felt a strange warmth in its circuits.
# T=1.8: ...danced with purple starlight through neon fog.
```

#### ☕ Java (Spring AI — Temperature per Request)
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatOptions;
import org.springframework.stereotype.Service;

@Service
public class SamplingStrategyService {

    private final ChatClient chatClient;

    public SamplingStrategyService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    // For structured/factual output — deterministic
    public String generateSQL(String request) {
        return chatClient.prompt()
                .user("Generate SQL query: " + request)
                .options(VertexAiGeminiChatOptions.builder()
                        .temperature(0.0)    // always same output
                        .topP(1.0)
                        .build())
                .call()
                .content();
    }

    // For creative content — high randomness
    public String generatePoem(String topic) {
        return chatClient.prompt()
                .user("Write a short creative poem about: " + topic)
                .options(VertexAiGeminiChatOptions.builder()
                        .temperature(1.5)   // high creativity
                        .topP(0.95)
                        .topK(40)
                        .build())
                .call()
                .content();
    }
}
```

---

## 6. Well-Known Training Datasets

LLMs learn from enormous text corpora collected from the internet and curated sources.

### Major Datasets

| Dataset | Source | Size | What It Contributes |
|---------|--------|------|---------------------|
| **[Common Crawl](https://commoncrawl.org/)** | Web crawl of the entire internet | ~1 PB (petabyte) raw | General world knowledge, diverse text styles |
| **Wikipedia** | Wikipedia articles (all languages) | ~21 GB text | Factual, structured, encyclopedic knowledge |
| **Books (Gutenberg, Books3)** | Published books | ~100 GB | Long-form reasoning, narrative, literary style |
| **GitHub (open source code)** | Public code repositories | ~1 TB | Programming knowledge, all languages/frameworks |
| **Arxiv** | Scientific papers (math, CS, physics) | ~270 GB | Reasoning, scientific notation, research patterns |
| **StackExchange** | Q&A forums (Stack Overflow etc.) | ~80 GB | Technical problem-solving, instruction-following |
| **OpenWebText** | Filtered Reddit-linked pages | ~40 GB | Conversational, opinionated, diverse topics |

### Data Quality Filtering Pipeline

```
Raw Internet Data (Common Crawl — 1 PB)
         │
         ▼
Language Detection  → keep English (or target language)
         │
         ▼
Deduplication       → remove exact and near-duplicate pages
         │
         ▼
Quality Filtering   → remove spam, gibberish, adult content
         │
         ▼
PII Removal         → strip personal information
         │
         ▼
Tokenization        → convert to token IDs
         │
         ▼
Final Training Set  (~500B–10T tokens depending on model)
```

> **Key Insight:** Data quality matters MORE than quantity. A smaller, well-curated dataset often outperforms a larger, noisy one. This is why Phi-3 (3.8B params) outperforms much larger models on reasoning tasks — it was trained on "textbook quality" data.

### Why GitHub Matters

```python
# The model learned from millions of GitHub repos
# This is why LLMs can write code in 100+ languages,
# understand API docs, and debug programs

# Example: GPT-4 can write this Spring AI code
# because it was trained on Spring documentation + 
# millions of Spring Boot projects on GitHub
```

---

## 7. Pre-training vs Distillation

### Two Ways to Build a Model

#### Option A: Pre-training from Scratch

```
Massive Raw Dataset (trillions of tokens)
         │
         ▼
Random weight initialization
         │
         ▼
Train on next-token prediction (self-supervised)
         │  (weeks/months on thousands of GPUs)
         ▼
Base Model (knows language, facts, reasoning)
         │
         ▼
Fine-tuning (instruction tuning, RLHF)
         │
         ▼
Final Model (GPT-4, Gemini, Llama)
```

**Cost:** $10M–$100M+ in compute for frontier models

#### Option B: Distillation

```
Large Teacher Model (e.g., GPT-4, 70B params)
         │
         ▼
Generate high-quality responses / soft labels
         │
         ▼
Small Student Model (e.g., 7B params) trained to
mimic the teacher's output distribution
         │
         ▼
Distilled Model
(smaller, faster, cheaper — nearly as capable)
```

**Cost:** 10–100x cheaper than training from scratch

### Comparison

| Aspect | Pre-training from Scratch | Distillation |
|--------|--------------------------|--------------|
| **Cost** | Extremely high ($10M+) | Much lower |
| **Data needed** | Trillions of raw tokens | Millions of teacher-generated examples |
| **Result** | Fully original model | Compressed version of teacher |
| **Performance** | State-of-the-art | Near-teacher quality at smaller size |
| **Examples** | GPT-4, Gemini 1.5 | Phi-3, Gemini Flash, Llama-distill |

> **Real Example:** Gemini 1.5 Flash is a **distilled** version of Gemini 1.5 Pro — 10x cheaper, nearly as capable for most tasks.

---

## 8. Base Models vs Instruct Models

### The Two Stages of Every LLM

```
Stage 1: Pre-training → BASE MODEL
──────────────────────────────────
Trained on raw text (predict next token)
Knows language, facts, code
Does NOT know how to follow instructions

Example:
  Prompt:  "What is the capital of France?"
  Base:    "What is the capital of Germany? What is the..."
           (continues text pattern — does not answer!)

Stage 2: Instruction Tuning → INSTRUCT / CHAT MODEL
─────────────────────────────────────────────────────
SFT (Supervised Fine-Tuning) on (instruction, response) pairs
+ RLHF (Reinforcement Learning from Human Feedback)
Learns to: answer questions, follow instructions, be helpful & safe

Example:
  Prompt:  "What is the capital of France?"
  Instruct: "The capital of France is Paris."  ✅
```

### SFT → RLHF Pipeline

```
Base Model
    │
    ▼
SFT (Supervised Fine-Tuning)
    ├── Dataset: thousands of (prompt → ideal response) pairs
    ├── Written by human annotators
    └── Model learns instruction-following behavior
    │
    ▼
Reward Model Training
    ├── Humans rank multiple model outputs (A > B > C)
    └── Train a reward model to predict human preference
    │
    ▼
RLHF (PPO)
    ├── Model generates responses
    ├── Reward model scores them
    └── Model is updated to maximize reward
    │
    ▼
Final Instruct / Chat Model
```

### Base vs Instruct — When to Use Which

| Scenario | Use | Reason |
|----------|-----|--------|
| Chatbot, Q&A, assistant | **Instruct** | Follows instructions correctly |
| Fine-tuning on your data | **Base** | Start from raw capability, add your own SFT |
| Text completion (creative writing) | **Base** | Continues text naturally without instruction framing |
| API/Production application | **Instruct** | Predictable, safe, structured responses |

### Popular Models in Base & Instruct Form

Use this prompt:
```
Give me most popular models in instruct and base form to run on my hardware
```

**Reference:**

| Model Family | Base Model | Instruct/Chat Model |
|-------------|-----------|---------------------|
| **Llama 3.1** | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` |
| **Qwen 2.5** | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-7B-Instruct` |
| **Mistral** | `mistralai/Mistral-7B-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` |
| **Gemma 2** | `google/gemma-2-9b` | `google/gemma-2-9b-it` |
| **Phi-3** | `microsoft/Phi-3-mini-4k` | `microsoft/Phi-3-mini-4k-instruct` |

#### 🐍 Python — Run Instruct Model Locally (Ollama)
```python
import ollama

# Pull model first: ollama pull llama3.1:8b-instruct
response = ollama.chat(
    model="llama3.1:8b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user",   "content": "What is the capital of France?"}
    ]
)
print(response["message"]["content"])
# "The capital of France is Paris."

# Base model — notice different behavior
response_base = ollama.generate(
    model="llama3.1:8b",   # base model
    prompt="What is the capital of France?"
)
print(response_base["response"])
# "What is the capital of Germany? What is the capital of Italy?..."
# (continues text, does not answer the question)
```

#### ☕ Java (Spring AI — Ollama with local instruct model)
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.ollama.OllamaChatOptions;
import org.springframework.stereotype.Service;

@Service
public class LocalModelService {

    private final ChatClient chatClient;

    public LocalModelService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    // Run Llama 3.1 Instruct locally via Ollama (no API key needed!)
    public String askLocal(String question) {
        return chatClient.prompt()
                .system("You are a helpful AI assistant.")
                .user(question)
                .options(OllamaChatOptions.builder()
                        .model("llama3.1:8b-instruct")  // local instruct model
                        .temperature(0.7)
                        .build())
                .call()
                .content();
    }
}
```
```properties
# application.properties — point to local Ollama server
spring.ai.ollama.base-url=http://localhost:11434
spring.ai.ollama.chat.options.model=llama3.1:8b-instruct
```

---

## 9. Exercises

### Exercise A: Frameworks to Run a Transformer on Your System

| Framework | Language | Best For | Install |
|-----------|----------|---------|---------|
| **PyTorch** | Python | Research, training, full control | `pip install torch` |
| **TensorFlow / Keras** | Python | Production deployment, mobile | `pip install tensorflow` |
| **HuggingFace Transformers** | Python | Easiest — 100K+ pretrained models | `pip install transformers` |
| **Ollama** | CLI / API | Run LLMs locally, zero config | `winget install Ollama.Ollama` |
| **LM Studio** | GUI | User-friendly local LLM runner | Download from lmstudio.ai |
| **ONNX Runtime** | Multi-lang | Fast inference (export from PyTorch) | `pip install onnxruntime` |
| **Deep Java Library (DJL)** | Java | Run HuggingFace models from Java | Maven: `ai.djl:api:0.27.0` |

#### 🐍 Python — Run Any Transformer in 3 Lines (HuggingFace)
```python
from transformers import pipeline

# Text classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
print(classifier("I love building AI applications with Spring!"))
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Text generation (GPT-2 — a Transformer, not a modern LLM)
generator = pipeline("text-generation", model="gpt2")
result = generator("The future of AI is", max_new_tokens=30)
print(result[0]["generated_text"])
```

#### ☕ Java (Deep Java Library — Run HuggingFace Transformer)
```java
// Maven dependency:
// ai.djl:api:0.27.0
// ai.djl.huggingface:tokenizers:0.27.0

import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Classifications;

// Load and run a sentiment analysis transformer from HuggingFace
Criteria<String, Classifications> criteria = Criteria.builder()
        .optApplication(Application.NLP.SENTIMENT_ANALYSIS)
        .setTypes(String.class, Classifications.class)
        .optEngine("PyTorch")
        .optModelUrls("djl://ai.djl.huggingface.pytorch/distilbert-base-uncased-finetuned-sst-2-english")
        .build();

try (ZooModel<String, Classifications> model = criteria.loadModel();
     Predictor<String, Classifications> predictor = model.newPredictor()) {

    Classifications result = predictor.predict("Spring AI makes Java developers love Gen-AI!");
    System.out.println(result.best().getClassName());  // POSITIVE
    System.out.printf("Confidence: %.2f%%%n", result.best().getProbability() * 100);
}
```

---

### Exercise B: 5 Popular Open-Source Transformers That Are NOT LLMs

LLMs are not the only Transformer-based models. Here are 5 powerful non-LLM transformers:

| # | Model | Task | Modality | Use Case |
|---|-------|------|----------|---------|
| 1 | **Whisper** (OpenAI) | Speech-to-Text | Audio → Text | Transcribe meetings, podcasts, videos |
| 2 | **ViT** — Vision Transformer (Google) | Image Classification | Image → Label | Classify product photos, medical images |
| 3 | **CLIP** (OpenAI) | Image-Text Matching | Image + Text → Score | Semantic image search, content moderation |
| 4 | **BERT** (Google) | Text Understanding | Text → Embeddings | Semantic search, NER, classification |
| 5 | **Wav2Vec 2.0** (Meta) | Speech Representation | Audio → Embeddings | Low-resource speech recognition, speaker ID |

#### 🐍 Python — Run Whisper (Speech-to-Text Transformer)
```python
import whisper

# Load Whisper model (base, small, medium, large)
model = whisper.load_model("base")

# Transcribe audio file
result = model.transcribe("meeting_recording.mp3")
print(result["text"])
# "The capital of Spain is Madrid and they just won the FIFA World Cup..."

# Detect language automatically
print(f"Detected language: {result['language']}")  # en
```

#### 🐍 Python — Run CLIP (Image-Text Matching)
```python
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

model = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Load image
image = Image.open("selfie.jpg")

# Match image against text descriptions
texts = ["a passport photo", "a selfie", "a landscape", "a dog"]
inputs = model(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = processor(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob:.3f}")
# a selfie: 0.821
# a passport photo: 0.094
```

---

## 10. Key Takeaways & Cheat Sheet

### Today's Core Concepts

```
Tokens   = sub-word units; everything in LLMs is measured in tokens
            └── pricing, context window, input/output limits

Embedding = token → float array; similar tokens → close vectors
            └── foundation of semantic understanding

Transformer pipeline:
  Text → Tokenize → Positional Encode → Embed → Attention layers
  → Probability distribution → Sample (temperature) → Next token

Training datasets:
  Common Crawl + Wikipedia + Books + GitHub + Arxiv
  → Pre-train base model → Instruction-tune → Instruct model

Temperature:
  0.0  = deterministic (code, SQL, JSON)
  0.7  = balanced (chatbots, Q&A)
  1.5+ = creative (poetry, music, brainstorming)
```

### When to Use What

| Task | Model Type | Temperature | Python Tool | Java Tool |
|------|-----------|-------------|-------------|-----------|
| Chatbot | Instruct | 0.5–0.8 | `google-generativeai` | `ChatClient` |
| Code generation | Instruct | 0.0–0.2 | `openai` | `ChatClient` |
| Semantic search | Embedding model | N/A | `sentence-transformers` | `EmbeddingClient` |
| Speech transcription | Whisper | N/A | `openai-whisper` | DJL + Whisper |
| Image classification | ViT / CLIP | N/A | `transformers` | DJL |
| Local / private | Instruct (Ollama) | any | `ollama` | `spring-ai-ollama` |

### Token Cost Quick Math

```
1,000 tokens  ≈  750 words  ≈  1.5 pages
$1 of GPT-4o input  =  200,000 tokens  =  150,000 words  =  600 pages
$1 of Gemini Flash  =  13,333,333 tokens  → extremely cheap for large volumes
```

---

## References & Further Reading

- [Transformer Explainer (Interactive)](https://poloclub.github.io/transformer-explainer/) — Visual transformer pipeline
- [Common Crawl](https://commoncrawl.org/) — The dataset that powers most LLMs
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — Visualize tokenization
- [Tiktoken (GitHub)](https://github.com/openai/tiktoken) — OpenAI's tokenizer library
- [HuggingFace Model Hub](https://huggingface.co/models) — Browse base & instruct models
- [Ollama](https://ollama.ai) — Run LLMs locally
- [Spring AI Docs](https://docs.spring.io/spring-ai/reference/) — Spring AI reference
- [OpenAI Pricing](https://openai.com/api/pricing/) — Latest token pricing
- [Google AI Pricing](https://ai.google.dev/pricing) — Latest Gemini pricing

---

*Notes compiled and expanded from Direct AI Blog — Classroom session 30 July 2026*
*Enhanced with deep insights, code examples (Python + Java Spring AI), architecture diagrams, and use cases*
