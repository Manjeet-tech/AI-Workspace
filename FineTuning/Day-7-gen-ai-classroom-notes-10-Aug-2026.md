# DAY-7: Gen-AI Developer Classroom Notes — 10 Aug 2026

> **Source:** https://directai.blog/2026/08/10/gen-ai-developer-classroom-notes-10-aug-2026/
> **Audience:** Gen-AI Developers (Beginner → Intermediate)

---

## Table of Contents

1. [The Core Principle: Fine-Tuning Is for Behavior, Not Facts](#1-the-core-principle-fine-tuning-is-for-behavior-not-facts)
2. [RAG vs Fine-Tuning — How to Decide](#2-rag-vs-fine-tuning--how-to-decide)
3. [Fine-Tuning Use Cases with Examples](#3-fine-tuning-use-cases-with-examples)
4. [Datasets — The Practical Details](#4-datasets--the-practical-details)
5. [Can You Fine-Tune Closed-Source Models?](#5-can-you-fine-tune-closed-source-models)
6. [Running Open-Source Models on Your Own Hardware](#6-running-open-source-models-on-your-own-hardware)
7. [Spring AI Examples for Each Deployment Path](#7-spring-ai-examples-for-each-deployment-path)
8. [Key Takeaways & Cheat Sheet](#8-key-takeaways--cheat-sheet)

---

## 1. The Core Principle: Fine-Tuning Is for Behavior, Not Facts

The single most important lesson from this session is:

> **Fine-tuning changes HOW the model behaves. RAG gives the model WHAT to know.**

```
┌─────────────────────────────────────────────────────────────────┐
│              FINE-TUNING vs RAG MENTAL MODEL                    │
│                                                                 │
│  FINE-TUNING                                                    │
│  ├── Teaches the model a tone/persona                           │
│  ├── Teaches a response format                                  │
│  ├── Teaches how to reason over a task type                     │
│  └── Encodes skill through weight updates                       │
│       (permanent, baked into the model)                         │
│                                                                 │
│  RAG                                                            │
│  ├── Provides specific up-to-date facts                         │
│  ├── Uses external knowledge base / vector DB                   │
│  └── Facts are NOT in the model weights                         │
│       (fetched at runtime, always current)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Distinction Matters

Trying to teach facts through fine-tuning is like trying to update a textbook by retraining the author. The author (model) will eventually confuse old and new facts. Instead:

- Give the author (model) **skills** through fine-tuning
- Give the author a **library** (vector DB) for facts through RAG
- Or do **both** for the best applications

### Example from the Real World

| You Need | Fine-Tuning? | RAG? | Reason |
|---------|-------------|------|--------|
| Customer support agent that sounds like your brand | YES | YES | Tone via FT; product docs via RAG |
| Latest stock prices | NO | YES | Prices change every second — keep them external |
| Code reviewer that follows your style guide | YES | NO/Optional | Style guide can be encoded in weights |
| Medical coding assistant for ICD-10 rules | YES | YES | Rules in RAG; reasoning style via FT |
| Chatbot that knows your CEO's biography | NO | YES | Facts change; keep in docs |
| Summarizer that outputs structured JSON | YES | NO | Output format is a behavior |

---

## 2. RAG vs Fine-Tuning — How to Decide

### 2.1 The Decision Matrix

```
START: What problem are you solving?
    │
    ▼
Does the answer depend on knowledge that changes frequently?
    │
    ├── YES  ───────────────────────────────────────────► Use RAG
    │                                                        (examples: stock prices, news, policies)
    │
    └── NO
         │
         ▼
Does the answer require a specific tone, format, or reasoning style?
    │
    ├── YES  ───────────────────────────────────────────► Use FINE-TUNING
    │                                                        (examples: brand voice, code style)
    │
    └── NO
         │
         ▼
Do you need both up-to-date facts AND a specific style?
    │
    ├── YES  ───────────────────────────────────────────► Use BOTH RAG + FINE-TUNING
    │                                                        (examples: support bots, legal assistants)
    │
    └── NO
         │
         ▼
You probably just need prompt engineering or a better base model.
```

### 2.2 The Knowledge / Skill 2×2

|  | Static Knowledge | Dynamic Knowledge |
|---|---|---|
| **Specific Style** | **Fine-Tune + maybe RAG** | **Fine-Tune + RAG** |
| **Generic Style** | **RAG** or base model | **RAG** |

- **Static + Style → Fine-tune** (learn skills + static facts)
- **Dynamic + Style → Fine-tune + RAG** (learn style, fetch facts at runtime)
- **Any fact-heavy but no style → RAG**

---

## 3. Fine-Tuning Use Cases with Examples

### 3.1 Blog Example 1: Organization Policies Chatbot

**Requirement:** Build an application that retrieves the *latest* information about organization policies.

**Solution:** **RAG** ✅

Why:
- Policies change frequently (new versions, amendments)
- Employees need the *current* version, not what was true last year
- You don't want to retrain the model every time HR updates a PDF

```
User: "What is the current work-from-home policy?"

┌────────────────────────────────────────────────────────────────┐
│                          RAG PIPELINE                          │
│                                                                │
│  1. User query ──► embed query ──► vector similarity search │
│  2. Retrieve top-3 policy PDF chunks                          │
│  3. Inject chunks into LLM prompt                             │
│  4. Generate answer based on retrieved context                │
│                                                                │
│  When policy PDFs change: just re-index the new files.        │
│  No model retraining needed.                                   │
└────────────────────────────────────────────────────────────────┘
```

#### Python RAG Code Example

```python
# pip install langchain chromadb sentence-transformers openai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# 1. Load policy PDFs
loaders = [PyPDFLoader(f"policies/{doc}") for doc in ["wfh.pdf", "leave.pdf"]]
documents = []
for loader in loaders:
    documents.extend(loader.load_and_split())

# 2. Embed and store in ChromaDB
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embedding, persist_directory="./policy-db")

# 3. Build QA chain
llm = ChatOpenAI(model="gpt-4o-mini")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 3}))

# 4. Ask a question
question = "What is the current work-from-home policy?"
answer = qa_chain.invoke(question)
print(answer)
```

---

### 3.2 Blog Example 2: Insurance Customer Support Chatbot

**Requirement:** Build a customer support chatbot that responds as a trained customer support representative of an insurance company.

**Solution:** **FINE-TUNE + RAG** ✅

Why:
- **Behavior**: It must sound like your company's trained support agent (tone, empathy, procedural phrasing)
- **Facts**: It needs access to up-to-date policy documents, claim statuses, product details

```
User: "My car claim was rejected. Can you help?"

Fine-tuned model:           +  RAG:
├── Empathetic tone         │   ├── Claim status from CRM
├── Insurance vocabulary    │   ├── Policy clauses
├── Refund/escalation       │   └── Rejection reasons
  procedure               │
└── Structured response     │
     format                │
                            ▼
              "I'm really sorry to hear your claim was declined.
               Based on your policy (section 4.2), damage caused by
               driving under the influence is excluded. However, I
               can escalate this for a manual review if you'd like."
```

#### Architecture for RAG + Fine-Tuned Model

```
┌─────────────────────────────────────────────────────────────────┐
│                 INSURANCE SUPPORT CHATBOT                       │
│                                                                 │
│  User Input                                                     │
│     │                                                           │
│     ▼                                                           │
│  ┌──────────────────┐                                          │
│  │  Query Router    │  ── "Need policy facts?" ──┬─YES─► RAG  │
│  │                  │                            │             │
│  │                  │  ── "Just conversational?" ─┴─NO──►      │
│  └──────────────────┘                                          │
│     │                                                           │
│     ▼                                                           │
│  ┌──────────────────┐                                          │
│  │  Fine-Tuned LLM  │  (tone: empathetic insurance agent)      │
│  └──────────────────┘                                          │
│     │                                                           │
│     ▼                                                           │
│  Final Response                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Spring AI RAG + Fine-Tuned Model Example

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class InsuranceSupportService {

    private final ChatClient chatClient;    // points to fine-tuned model endpoint
    private final VectorStore vectorStore;  // Chroma / PGVector / Redis

    public InsuranceSupportService(ChatClient chatClient, VectorStore vectorStore) {
        this.chatClient = chatClient;
        this.vectorStore = vectorStore;
    }

    public String handleQuery(String customerQuestion) {
        // 1. Retrieve relevant policy documents
        List<Document> relevantDocs = vectorStore.similaritySearch(
            SearchRequest.query(customerQuestion).withTopK(3)
        );

        String context = relevantDocs.stream()
            .map(Document::getContent)
            .collect(Collectors.joining("\n---\n"));

        // 2. Generate empathetic, accurate response using fine-tuned model
        return chatClient.prompt()
            .system("""
                You are a trained customer support agent for Acme Insurance.
                Be empathetic, concise, and always ground your answers in the
                provided policy context. If you cannot answer, offer to escalate.
                """)
            .user("""
                Customer Question: %s

                Relevant Policy Context:
                %s
                """.formatted(customerQuestion, context))
            .call()
            .content();
    }
}
```

---

## 4. Datasets — The Practical Details

### 4.1 Why `jsonl` Is the Most Popular Format

**JSONL = JSON Lines.** Every line is a self-contained JSON object. This is the de-facto format for fine-tuning because:

1. **Streaming-friendly** — read one example at a time, no need to load a 10GB file into memory
2. **Appendable** — easily add more data without rewriting the whole file
3. **Tool-friendly** — OpenAI, HuggingFace, Unsloth, llama.cpp, vLLM all accept it
4. **Git-friendly** — line-by-line diffs for version control

```
train.jsonl:
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
...
```

#### JSONL vs JSON vs CSV / Parquet

| Format | Pros | Cons | Best For |
|--------|------|------|----------|
| **JSONL** | Streaming, appendable, line-by-line parsing | No schema validation per file | Fine-tuning datasets, logs |
| **JSON** | Single object, human-readable | Must load entire file | Small config files, single examples |
| **CSV** | Simple tabular view | Doesn't handle nested messages | Non-conversational datasets |
| **Parquet** | Compressed, fast column access | Binary, not human-readable | Large-scale preprocessing |
| **Arrow** | Zero-copy, fast | Less universal | Huge dataset streaming |

### 4.2 Conversational Format for Instruct Models

Most fine-tuning is done on **instruct models** (not base models), so the dataset is a sequence of messages. The exact syntax depends on the model family.

#### OpenAI / ChatGPT Format (universal)

```jsonl
{"messages": [{"role": "system", "content": "You are Acme Insurance support."}, {"role": "user", "content": "My claim was rejected."}, {"role": "assistant", "content": "I'm sorry to hear that. Could you provide your claim number?"}]}
{"messages": [{"role": "system", "content": "You are Acme Insurance support."}, {"role": "user", "content": "What is the deductible for comprehensive coverage?"}, {"role": "assistant", "content": "For comprehensive coverage, the deductible is $500 per claim."}]}
```

#### Llama 3 / 3.1 ChatML Format

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "Write a Python function to reverse a string."}, {"role": "assistant", "content": "```python\ndef reverse(s):\n    return s[::-1]\n```"}]}
```

The tokenizer's `apply_chat_template` will internally convert these to:

```
<|start_header_id|>system<|end_header_id|>
You are a helpful coding assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Write a Python function to reverse a string.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```python
def reverse(s):
    return s[::-1]
```<|eot_id|>
```

#### Mistral Format

```
<s> [INST] Write a Python function to reverse a string. [/INST] ```python
def reverse(s):
    return s[::-1]
```</s>
```

#### Format Depends on the Model — Quick Reference

| Model Family | Format | Special Tokens |
|-------------|--------|----------------|
| **GPT-4 / OpenAI** | `messages` array with roles | None explicit (API handles it) |
| **Llama 3/3.1/3.2** | ChatML with header IDs | `<|start_header_id|>`, `<|eot_id|>` |
| **Mistral 7B / Mixtral** | `[INST] ... [/INST]` | `<s>`, `</s>` |
| **Gemma 2/3** | `user\n` + `model\n` | `<start_of_turn>`, `<end_of_turn>` |
| **Qwen 2/2.5/3** | ChatML variant | `<|im_start|>`, `<|im_end|>` |
| **Phi-3/4** | ChatML-like | `<|user|>`, `<|assistant|>` |

> **Critical:** Using the wrong chat template produces garbage outputs because the model was trained to recognize specific control tokens. Always use `tokenizer.apply_chat_template()` to generate the correct string.

### 4.3 Non-Conversational Formats

For classification or extraction tasks, you can also use simpler formats:

#### Alpaca Format

```jsonl
{"instruction": "Classify the sentiment.", "input": "This product is amazing!", "output": "POSITIVE"}
{"instruction": "Classify the sentiment.", "input": "Very disappointing.", "output": "NEGATIVE"}
```

#### Completion-Only Format

```jsonl
{"text": "### Text: This product is amazing!\n### Sentiment: POSITIVE"}
{"text": "### Text: Very disappointing.\n### Sentiment: NEGATIVE"}
```

### 4.4 Dataset Validation Checklist

```
Before training:
□ Every line is valid JSON
□ No trailing commas
□ No empty assistant messages
□ System prompt is consistent if used
□ User/assistant roles alternate correctly
□ Special tokens are NOT added manually (let tokenizer handle them)
□ Train/validation split is 80/20 or 90/10
□ Validation set is truly unseen — no overlap with train
□ Token lengths are under max_seq_length
□ File size is under platform limits (OpenAI: 1GB per file, HuggingFace: no hard limit)
```

### 4.5 Counting Tokens Before Training

```python
from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def count_tokens(jsonl_line: str) -> int:
    data = json.loads(jsonl_line)
    formatted = tokenizer.apply_chat_template(data["messages"], tokenize=True)
    return len(formatted)

with open("train.jsonl") as f:
    for i, line in enumerate(f, 1):
        n = count_tokens(line)
        if n > 2048:
            print(f"Line {i} is too long: {n} tokens")
```

---

## 5. Can You Fine-Tune Closed-Source Models?

### 5.1 The Short Answer: Yes, but Through Managed APIs

You cannot download Claude, Gemini, or GPT weights. Instead, you upload a `jsonl` dataset to the vendor's API, and they fine-tune a custom copy for you.

```
Closed-Source Model Fine-Tuning:

You ──► Upload JSONL dataset ──► OpenAI / Google / Anthropic cloud
                                They train a private model copy
                                Return a model ID like ft:gpt-4o-mini:...
                                You call it via the same API, just different model= string
```

### 5.2 Vendor Comparison

| Vendor | Fine-Tuning Available | Format | Pricing Model | Notes |
|--------|---------------------|--------|--------------|-------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-3.5-Turbo | JSONL chat completions | Per training token + per inference token | Most mature platform |
| **Google** | Gemini 1.5 Flash, Gemini 1.5 Pro | JSONL (vertex/gemini) | Per training hour + inference | Enterprise via Vertex AI |
| **Anthropic** | Claude 3.5 Haiku (limited) | Custom via console | Contact sales | Most restrictive access |
| **Cohere** | Command R/R+ | JSONL | Token-based | Good for RAG-style fine-tuning |
| **Mistral AI** | Mistral Small / Medium (API) | JSONL | Token-based | Also offers open-source weights |

### 5.3 OpenAI Fine-Tuning Example

```bash
# 1. Prepare data as JSONL (OpenAI chat completions format)
# Each line: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

# 2. Upload the file
curl https://api.openai.com/v1/files \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F purpose="fine-tune" \
  -F file=@"insurance_support_train.jsonl"

# Response includes file ID: file-xxxxxxxx

# 3. Create fine-tuning job
curl https://api.openai.com/v1/fine_tuning/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "training_file": "file-xxxxxxxx",
    "model": "gpt-4o-mini-2024-07-18",
    "hyperparameters": {
      "n_epochs": 2,
      "batch_size": "auto",
      "learning_rate_multiplier": "auto"
    }
  }'

# 4. Wait for email completion. You receive ft:MODEL-NAME:JOB-ID
# 5. Use it like any other OpenAI model
```

### 5.4 OpenAI Fine-Tuning via Python SDK

```python
from openai import OpenAI
import json

client = OpenAI()

# Upload training file
with open("insurance_support_train.jsonl", "rb") as f:
    file = client.files.create(file=f, purpose="fine-tune")

print(f"Uploaded file ID: {file.id}")

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "n_epochs": 2,
        "batch_size": "auto",
        "learning_rate_multiplier": "auto"
    },
    suffix="insurance-support"
)

print(f"Job ID: {job.id}")
print(f"Status: {job.status}")

# Poll until complete
import time
while True:
    job = client.fine_tuning.jobs.retrieve(job.id)
    print(f"Status: {job.status}")
    if job.status in ["succeeded", "failed", "cancelled"]:
        break
    time.sleep(60)

# Retrieve fine-tuned model ID
fine_tuned_model = job.fine_tuned_model
print(f"Fine-tuned model: {fine_tuned_model}")
# Example output: ft:gpt-4o-mini:my-org:insurance-support:abc123

# Use the fine-tuned model
response = client.chat.completions.create(
    model=fine_tuned_model,
    messages=[
        {"role": "system", "content": "You are Acme Insurance support."},
        {"role": "user", "content": "My car claim was rejected. Can you help?"}
    ]
)
print(response.choices[0].message.content)
```

### 5.5 Spring AI Using OpenAI Fine-Tuned Model

```properties
# application.properties
spring.ai.openai.api-key=${OPENAI_API_KEY}
# Use the exact fine-tuned model ID from the API dashboard or job response
spring.ai.openai.chat.options.model=ft:gpt-4o-mini:my-org:insurance-support:abc123
```

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

@Service
public class FineTunedInsuranceService {

    private final ChatClient chatClient;

    public FineTunedInsuranceService(ChatClient.Builder builder) {
        this.chatClient = builder
            .defaultSystem("You are Acme Insurance customer support. Be empathetic.")
            .build();
    }

    public String support(String question) {
        return chatClient.prompt()
            .user(question)
            .call()
            .content();
    }
}
```

> **Key point:** Fine-tuning a closed-source model is just an API call. You prepare `jsonl`, upload it, wait, then change the `model` string. The rest of your application code stays the same.

---

## 6. Running Open-Source Models on Your Own Hardware

### 6.1 The Deployment Spectrum

| Tool | Ease | Speed | Best For | Hardware |
|------|------|-------|---------|----------|
| **Hugging Face Transformers** | Easy (Python) | Baseline | Prototyping, research | Any GPU with enough VRAM |
| **llama.cpp** | Hard (C++ build) | Very fast CPU inference | Edge devices, Macs, no GPU | CPU (ARM/x86), GPU optional |
| **Ollama** | Very easy | Good | Local development, easy sharing | Consumer GPU (8GB+), CPU fallback |
| **vLLM** | Medium (server setup) | Fastest GPU serving | Production, high throughput | NVIDIA datacenter GPUs |
| **TGI (HuggingFace)** | Medium | Fast | Production with HF ecosystem | NVIDIA GPUs |
| **Triton + TensorRT-LLM** | Hard | Fastest | Enterprise NVIDIA deployment | A100/H100 |

### 6.2 Option 1: Hugging Face Transformers (Python Native)

**When to use:** Prototyping, research, fine-tuning workflows, when you have a GPU with enough VRAM.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {"role": "system", "content": "You are Acme Insurance support."},
    {"role": "user", "content": "What is my deductible?"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

**Pros:** Full control, can fine-tune, largest ecosystem
**Cons:** Requires manual memory management, not optimized for concurrent requests

### 6.3 Option 2: llama.cpp (C++ Runtime)

**When to use:** Running models on CPU, edge devices, Raspberry Pi, mobile, Apple Silicon.

```bash
# 1. Clone and build

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make -j 4

# 2. Download or convert a GGUF model
# Many GGUF files are already on HuggingFace
wget https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF/resolve/main/llama-3.1-8b-instruct.Q4_K_M.gguf

# 3. Run inference
./main \
  -m llama-3.1-8b-instruct.Q4_K_M.gguf \
  -p "You are Acme Insurance support.\nUser: What is my deductible?\nAssistant:" \
  -n 200 \
  -t 4 \
  --temp 0.7
```

**Pros:** Runs on CPU, extremely memory efficient, many quantization options
**Cons:** Lower-level, must manage prompts manually, slower than GPU servers

### 6.4 Option 3: Ollama (Easiest Local Deployment)

**When to use:** Local development, quick demos, small team sharing, Docker deployment.

```bash
# 1. Install (macOS/Linux/Windows WSL)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model
ollama pull llama3.1:8b

# 3. Run it interactively
ollama run llama3.1:8b

# 4. Run in API server mode
ollama serve
```

#### Creating a Custom Modelfile

```dockerfile
# Modelfile for insurance support bot (uses fine-tuned GGUF or base model)
FROM llama3.1:8b

SYSTEM """You are a trained customer support agent for Acme Insurance.
Be empathetic, concise, and ground answers in company policy."""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

# Optionally point to a local GGUF file
# FROM ./my-finetuned-model.gguf
```

```bash
ollama create acme-insurance -f Modelfile
ollama run acme-insurance
```

**Pros:** One-line install, Modelfile system prompts, OpenAI-compatible API, Docker images
**Cons:** Single-machine, not designed for massive scale

### 6.5 Option 4: vLLM (Production GPU Serving)

**When to use:** High-throughput production APIs, concurrent users, Kubernetes deployments.

```bash
# 1. Install
pip install vllm

# 2. Start server with an OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --port 8000

# 3. Query it like OpenAI
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are Acme Insurance support."},
      {"role": "user", "content": "What is my deductible?"}
    ],
    "temperature": 0.7
  }'
```

**Pros:** PagedAttention for high throughput, continuous batching, OpenAI-compatible API, tensor/pipeline parallelism, quantization support
**Cons:** Requires NVIDIA GPU, more setup than Ollama

### 6.6 Docker for Ollama

```dockerfile
# Dockerfile for Ollama deployment
FROM ollama/ollama:latest

# Pull the model at build time (optional; increases image size)
RUN ollama serve & \
    sleep 5 && \
    ollama pull llama3.1:8b

# Copy custom Modelfile and build it
COPY Modelfile /Modelfile
RUN ollama serve & \
    sleep 5 && \
    ollama create acme-insurance -f /Modelfile

EXPOSE 11434

ENTRYPOINT ["ollama"]
CMD ["serve"]
```

```bash
docker build -t acme-ollama .
docker run -p 11434:11434 --gpus all acme-ollama
```

---

## 7. Spring AI Examples for Each Deployment Path

### 7.1 Hugging Face via Local Python Server

Wrap your HuggingFace inference in a small FastAPI server, then call it from Spring AI using a custom `RestClient` or the OpenAI-compatible bridge.

```python
# local_llm_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

class ChatRequest(BaseModel):
    messages: list
    max_new_tokens: int = 200

def generate(messages, max_new_tokens):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    content = generate(req.messages, req.max_new_tokens)
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}

# Run: uvicorn local_llm_server:app --host 0.0.0.0 --port 8000
```

```properties
# Spring Boot points to local server as if it were OpenAI
spring.ai.openai.base-url=http://localhost:8000/v1
spring.ai.openai.api-key=not-needed
spring.ai.openai.chat.options.model=local-llama-3.1
```

```java
@Service
public class LocalHfService {
    private final ChatClient chatClient;
    public LocalHfService(ChatClient.Builder builder) { this.chatClient = builder.build(); }
    public String ask(String prompt) {
        return chatClient.prompt().user(prompt).call().content();
    }
}
```

### 7.2 llama.cpp Server

llama.cpp has a built-in `--server` mode that exposes an OpenAI-compatible API.

```bash
# Start llama.cpp server
./server \
  -m llama-3.1-8b-instruct.Q4_K_M.gguf \
  -c 4096 \
  --host 0.0.0.0 \
  --port 8080
```

```properties
# Spring Boot
spring.ai.openai.base-url=http://localhost:8080/v1
spring.ai.openai.api-key=not-needed
spring.ai.openai.chat.options.model=llama-3.1-8b
```

### 7.3 Ollama (Spring AI Native Support)

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
spring.ai.ollama.chat.options.model=acme-insurance
spring.ai.ollama.chat.options.temperature=0.3
```

```java
@Service
public class OllamaInsuranceService {

    private final ChatClient chatClient;

    public OllamaInsuranceService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public String ask(String question) {
        return chatClient.prompt()
            .user(question)
            .call()
            .content();
    }
}
```

### 7.4 vLLM (Production)

```properties
# application.properties — vLLM exposes OpenAI-compatible /v1 endpoint
spring.ai.openai.base-url=http://gpu-server:8000/v1
spring.ai.openai.api-key=internal-token
spring.ai.openai.chat.options.model=meta-llama/Llama-3.1-8B-Instruct
```

```java
@RestController
@RequestMapping("/api/insurance")
public class InsuranceController {

    private final ChatClient chatClient;

    public InsuranceController(ChatClient.Builder builder) {
        this.chatClient = builder
            .defaultSystem("You are Acme Insurance support. Be empathetic.")
            .build();
    }

    @PostMapping(consumes = "text/plain")
    public String ask(@RequestBody String question) {
        return chatClient.prompt().user(question).call().content();
    }
}
```

### 7.5 Deployment Path Selection in Spring AI

```
┌─────────────────────────────────────────────────────────────────┐
│           CHOOSING A BACKEND IN SPRING AI                      │
│                                                                 │
│  Local dev / no GPU        ──► Ollama (spring-ai-ollama)      │
│  Edge / CPU-only device    ──► llama.cpp server               │
│  Research / fine-tuning    ──► HuggingFace Transformers       │
│  Production / high traffic ──► vLLM (OpenAI-compatible)       │
│  Closed-source fine-tuned  ──► OpenAI / Gemini / Anthropic    │
│                                                                 │
│  Spring AI abstracts all of them behind ChatClient/ImageClient │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Key Takeaways & Cheat Sheet

### 8.1 One-Sentence Lessons

1. **Fine-tuning = behavior.** Use it for tone, format, style, and reasoning patterns.
2. **RAG = facts.** Use it for anything that changes, is specific, or must be sourced.
3. **Best apps often combine both** — fine-tuned model + RAG retriever.
4. **JSONL** is the universal fine-tuning dataset format because it's streamable and tool-friendly.
5. **The chat format depends on the model** — always use the model's correct chat template.
6. **Closed-source models can be fine-tuned** via vendor APIs — upload JSONL, get a model ID, change one config string.
7. **Open-source models run on hardware** through HuggingFace, llama.cpp, Ollama, or vLLM — choose based on ease vs scale.

### 8.2 Fine-Tune vs RAG Quick Reference

| Need | Fine-Tune | RAG |
|------|-----------|-----|
| Specific tone/voice | ✅ | ❌ |
| Output format (JSON, markdown tables) | ✅ | ❌ |
| Task reasoning style | ✅ | ❌ |
| Up-to-date facts | ❌ | ✅ |
| Source attribution | ❌ | ✅ |
| Large changing document corpus | ❌ | ✅ |
| Sensitive data privacy (on-prem) | ✅ (open-source) | ✅ (self-hosted vector DB) |
| Cost to update | High (retrain) | Low (re-index) |

### 8.3 JSONL Format Checklist

```
Valid fine-tuning JSONL:
□ One JSON object per line
□ Each object has a "messages" array for instruct models
□ Messages have "role" and "content"
□ Roles: system, user, assistant (in OpenAI format)
□ No blank assistant responses
□ No trailing commas
□ System prompt consistent across examples
□ Train/validation split created
```

### 8.4 Hardware / Deployment Decision Tree

```
Do you have a GPU?
    │
    ├── NO ──► llama.cpp (CPU, edge) or Ollama (CPU fallback)
    │
    └── YES
         │
         ▼
    Is this for production / many users?
         │
         ├── NO ──► Ollama (easiest) or HuggingFace (research)
         │
         └── YES ──► vLLM (best throughput) or TGI (HF ecosystem)
```

### 8.5 Glossary

| Term | Definition |
|------|-----------|
| **Behavior** | How a model responds — tone, format, style, reasoning |
| **RAG** | Retrieval-Augmented Generation — fetch relevant docs at runtime |
| **JSONL** | JSON Lines — one JSON object per line |
| **Chat template** | The model-specific token formatting for conversations |
| **Fine-tuning** | Adjusting model weights to change behavior |
| **Instruct model** | A pre-trained model already trained to follow instructions |
| **GGUF** | llama.cpp's quantized model file format |
| **vLLM** | High-throughput GPU inference engine for LLMs |
| **Ollama** | User-friendly wrapper around llama.cpp for local models |
| **Closed-source fine-tuning** | Training a hosted model (OpenAI/Gemini/Claude) via API |
| **Open-source fine-tuning** | Training weights locally using PEFT/Unsloth/TRL |

---

> **Next Session Preview:** The blog listed PEFT, LoRA, and QLoRA as upcoming concepts — these were already covered in Day-5 and Day-6 classroom notes.

---

*Notes compiled with deep-dive explanations, examples, and use cases based on:*
*https://directai.blog/2026/08/10/gen-ai-developer-classroom-notes-10-aug-2026/*
