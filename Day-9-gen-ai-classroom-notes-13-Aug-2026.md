# DAY-9: Gen-AI Developer Classroom Notes — 13 Aug 2026

> **Source:** https://directai.blog/2026/08/13/gen-ai-developer-classroom-notes-13-aug-2026/
> **Colab Notebook:** https://github.com/GenAIDevelopment/agenticai/blob/main/aug26/finetune/Qwen3_PizzaPalace_finetune.ipynb
> **Audience:** Gen-AI Developers (Beginner → Intermediate)

---

## Table of Contents

1. [Today's Goal: Execute the Fine-Tune End-to-End](#1-todays-goal-execute-the-fine-tune-end-to-end)
2. [Colab Setup — Open the Notebook](#2-colab-setup--open-the-notebook)
3. [Notebook Walkthrough: Step by Step](#3-notebook-walkthrough-step-by-step)
4. [Loading Data — `pizza_palace_400_train.jsonl` & `_val.jsonl`](#4-loading-data--pizza_palace_400_trainjsonl--_valjsonl)
5. [Qwen3 Chat Template & `train_on_responses_only`](#5-qwen3-chat-template--train_on_responses_only)
6. [Training, Eval Loss & Overfitting Detection](#6-training-eval-loss--overfitting-detection)
7. [Quick Sanity Test](#7-quick-sanity-test)
8. [Export Formats — GGUF vs Merged 16-bit](#8-export-formats--gguf-vs-merged-16-bit)
9. [Ollama Deployment](#9-ollama-deployment)
10. [vLLM Deployment](#10-vllm-deployment)
11. [Spring AI Integration](#11-spring-ai-integration)
12. [Key Takeaways & Cheat Sheet](#12-key-takeaways--cheat-sheet)

---

## 1. Today's Goal: Execute the Fine-Tune End-to-End

Today's class is the **hands-on execution** of the Pizza Palace fine-tune we designed in Day-8. Instead of re-explaining theory, we:

1. Open a ready-made **Unsloth Colab notebook**
2. Upload our **train/validation JSONL files**
3. Fine-tune **Qwen3-4B** using **QLoRA**
4. Watch **train vs. validation loss**
5. Export to **GGUF for Ollama** and **merged 16-bit for vLLM**

```
Dataset (320 train + 80 val)
        │
        ▼
   ┌──────────────┐
   │  Google Colab │ ← T4 GPU (free), 15 GB VRAM
   │  Qwen3-4B     │
   │  QLoRA r=16   │
   │  3 epochs     │
   └──────┬───────┘
          │
          ▼
   Three artifacts:
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │  LoRA adapter    │  │  GGUF (q4_k_m)   │  │  Merged 16-bit   │
   │  (few MB)        │  │  for Ollama      │  │  for vLLM        │
   │  reusable        │  │  edge/laptop     │  │  production GPU  │
   └──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 2. Colab Setup — Open the Notebook

### 2.1 Steps from the Blog

1. Go to https://colab.research.google.com
2. **File → Open Notebook**
3. Switch to the **GitHub** tab
4. Paste this URL:
   ```
   https://github.com/GenAIDevelopment/agenticai/blob/main/aug26/finetune/Qwen3_PizzaPalace_finetune.ipynb
   ```
5. Click the search result to open it

### 2.2 Set GPU Runtime

```
Runtime → Change runtime type → Hardware accelerator: T4 GPU
```

A T4 has:
- ~15 GB VRAM
- Enough for Qwen3-4B in 4-bit + LoRA adapters + small dataset

### 2.3 Upload JSONL Files

In the Colab left sidebar:

```
📁 Files (folder icon)
   ├── Right-click → Upload
   ├── Select pizza_palace_400_train.jsonl
   └── Select pizza_palace_400_val.jsonl
```

Or, from Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/pizza_palace_400_train.jsonl /content/
!cp /content/drive/MyDrive/pizza_palace_400_val.jsonl /content/
```

---

## 3. Notebook Walkthrough: Step by Step

### 3.1 Cell 1 — Install Unsloth

```python
# %%capture
!pip install -U unsloth unsloth_zoo

# Optional: pin versions for reproducibility
# !pip install "unsloth==2026.5.*" "unsloth_zoo==2026.5.*"
```

> **Why `unsloth_zoo`?** This helper package downloads pre-compiled Triton/CUDA kernels that make Unsloth's training 2–5× faster than standard PEFT. If you see "kernel not found", upgrade or reinstall.

---

### 3.2 Cell 2 — Load Qwen3-4B in 4-bit

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024      # enough for system menu + short chat
dtype = None               # auto: bf16 on Ampere, fp16 on T4
load_in_4bit = True        # Q in QLoRA

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "unsloth/Qwen3-4B",    # 4B params, ~2.5 GB in 4-bit
    max_seq_length = max_seq_length,
    dtype          = dtype,
    load_in_4bit   = load_in_4bit,
)
```

#### Why Qwen3-4B?

| Model | Params | 4-bit VRAM | Notes |
|-------|--------|-----------|-------|
| Qwen3-4B | 4B | ~2.5 GB | Fits easily on T4, good for this demo |
| Qwen3-7B | 7B | ~4.5 GB | Also fits on T4, slightly slower |
| Llama 3.1 8B | 8B | ~5 GB | Fits, but less headroom |
| Qwen3-32B | 32B | ~20 GB | **Will not fit** on free T4 |

Qwen3 is an **Instruct model** and uses **ChatML** internally, which matches our `messages` JSONL format.

---

### 3.3 Cell 3 — Attach LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",        # Attention
        "gate_proj", "up_proj", "down_proj"             # MLP/FFN
    ],
    lora_alpha = 16,
    lora_dropout = 0,                 # 0 is optimized in Unsloth
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # 30% extra VRAM savings
    random_state = 3407,
    use_rslora = False,               # try True later for higher ranks
    loftq_config = None,
)
```

#### Decoding the LoRA Config

```
Base model (Qwen3-4B):      ~4 billion parameters
LoRA adapters (r=16):       ~12 million parameters  (only 0.3% trainable!)

For this notebook, the team chose:
  r=16      → enough capacity for pizza persona + "use the menu" behavior
  alpha=16  → alpha/r ratio = 1.0  (conservative scaling)

Note: In Day-8 we recommended alpha=32 for r=16 (ratio = 2.0).
      Both are valid — the notebook uses the more conservative alpha=16.
```

---

### 3.4 Cell 4 — Load Train & Validation Datasets

```python
from datasets import load_dataset

train_dataset = load_dataset("json", data_files="pizza_palace_400_train.jsonl", split="train")
val_dataset   = load_dataset("json", data_files="pizza_palace_400_val.jsonl",   split="train")

print(train_dataset)
print("example:", train_dataset[0]["messages"][1])
```

Expected output:

```
Dataset({
    features: ['messages'],
    num_rows: 320
})
example: {'role': 'user', 'content': 'do u deliver after 11pm'}
```

---

### 3.5 Cell 5 — Apply Qwen3's Chat Template

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="qwen3")

def format_chat(examples):
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        for convo in examples["messages"]
    ]
    return {"text": texts}

train_dataset = train_dataset.map(format_chat, batched=True)
val_dataset   = val_dataset.map(format_chat,   batched=True)

print(train_dataset[0]["text"][:600])
```

#### What `apply_chat_template` Produces

For Qwen3, the raw text becomes:

```
<|im_start|>system
You are a warm, upbeat customer support agent for Pizza Palace. Today's menu:
Margherita ₹210
Paneer Supreme ₹330
...
Use ONLY this menu...<|im_end|>
<|im_start|>user
do u deliver after 11pm<|im_end|>
<|im_start|>assistant
Hey, thanks for stopping by! Great question! ...<|im_end|>
```

> **You never write these tokens by hand.** The tokenizer wraps your `messages` list into the exact string the model was trained on.

#### What Is `enable_thinking=False`?

Qwen3 has a **reasoning/thinking mode** that emits an internal thought block before answering:

```
<|im_start|>user
What veg pizza should I get?<|im_end|>
<|im_start|>assistant
` / ` utesentinel)` |
| **Llama 3** | ` breathtaking` | ` coarse` | ` conversational` | `<|eot_id|>` |
| **Mistral** | `[INST]` wrapper | `[INST]...[/INST]` | Direct after `[/INST]` | `</s>` |
| **Gemma** | `user\n` / `model\n` | `<start_of_turn>user\n` | `<start_of_turn>model\n` | `<end_of_turn>` |

For Qwen3, the notebook uses:
```python
instruction_part = "<|im_start|>user\n"
response_part    = "<|im_start|>assistant\n"
```

This tells Unsloth: **"mask everything between user start and assistant start, then train from assistant start onwards."**

### 5.2 Full ChatML String Anatomy

```
<|im_start|>system
You are a Pizza Palace agent.<|im_end|>
<|im_start|>user
do you have veg pizza?<|im_end|>
<|im_start|>assistant
Yes! Our Margherita and Paneer Supreme are great veg picks...<|im_end|>

Token boundaries (what the model actually sees):
  im_start      = special control token
  system/user/assistant = role markers
  im_end        = turn terminator
```

---

## 6. Training, Eval Loss & Overfitting Detection

### 6.1 What the Numbers Mean

| You See | Interpretation |
|---------|---------------|
| `training loss` decreases, `eval loss` decreases | Healthy training |
| `training loss` decreases, `eval loss` flat | Slight overfitting, consider stopping |
| `training loss` decreases, `eval loss` rises | Overfitting — stop immediately |
| `training loss` flat around 1.5+ | Underfitting — increase LR or epochs |
| `training loss` spikes / NaN | LR too high or data issue |

### 6.2 Colab Memory Tips

```
If you hit OOM on T4:
1. Reduce per_device_train_batch_size to 1
2. Increase gradient_accumulation_steps to 8
3. Reduce max_seq_length to 512
4. Enable use_gradient_checkpointing="unsloth" (already on)
5. Use Qwen3-1.5B or Qwen3-0.6B for even smaller footprint
```

---

## 7. Quick Sanity Test

### 7.1 Test Prompts to Try

```python
test_prompts = [
    "hey do u have anything veg",
    "what's the cheapest pizza",
    "app charged me twice",
    "recommend something",
    "do you have a calzone",
    "3 pizzas for office party, budget around 1000",
]
```

For each one, check:
- ✅ Warm, upbeat tone
- ✅ Only recommends from the menu
- ✅ Prices match the system prompt
- ✅ Handles complaints with apology + action
- ✅ Suggests a relevant add-on when appropriate

### 7.2 Red Flags to Watch

| Output | Problem | Fix |
|--------|---------|-----|
| Recommends "Pepperoni" when menu has none | Hallucination | More training examples, longer training, or stronger system prompt |
| Gives wrong price | Memorized old menu | Update system prompt in inference, not model weights |
| Sounds cold/robotic | Underfitting on style | More epochs or higher r |
| Repeats the same phrase every answer | Overfitting / memorization | Reduce epochs, more diverse data |
| Starts with thinking tokens | `enable_thinking=False` not passed | Add the flag to generate call |

---

## 8. Export Formats — GGUF vs Merged 16-bit

### 8.1 Which Artifact for What?

| Target | Method | Output Format | Size (Qwen3-4B) | Best For |
|--------|--------|--------------|-----------------|----------|
| **Ollama / laptop / edge** | `save_pretrained_gguf` | `.gguf` | ~2.5 GB (q4_k_m) | Local chat, CPU inference |
| **vLLM / TGI / HF** | `save_pretrained_merged` | Safetensors folder | ~8 GB (16-bit) | Production GPU serving |
| **Re-merge later** | `save_pretrained` | LoRA adapter (~50 MB) | ~50 MB | Version control, experimentation |

### 8.2 Save LoRA Adapters (Lightweight)

```python
model.save_pretrained("pizza_lora")
tokenizer.save_pretrained("pizza_lora")

# Optional: push to HuggingFace Hub
# model.push_to_hub("your-username/pizza-qwen3-lora", token="hf_...")
```

These are just the A and B matrices — tiny and perfect for sharing or versioning.

### 8.3 Export to GGUF for Ollama

```python
model.save_pretrained_gguf(
    "pizza_gguf",
    tokenizer,
    quantization_method = "q4_k_m",   # ~2.5 GB
    # quantization_method = "q8_0",   # ~4.5 GB, higher quality
)

!ls -lh pizza_gguf
```

#### GGUF Quantization Options

| Method | Size | Quality | Use Case |
|--------|------|---------|----------|
| `q4_k_m` | ~2.5 GB | Good | Laptops, Ollama, edge |
| `q4_k_s` | ~2.3 GB | Okay | Very constrained devices |
| `q8_0` | ~4.5 GB | Very good | Higher-fidelity local use |
| `f16` | ~8 GB | Best | GPU with plenty of VRAM |

### 8.4 Export to Merged 16-bit for vLLM

```python
model.save_pretrained_merged(
    "pizza_merged_16bit",
    tokenizer,
    save_method = "merged_16bit",
)

!ls -lh pizza_merged_16bit

# Optional: push merged model to HuggingFace Hub
# model.push_to_hub_merged("your-username/pizza-qwen3", tokenizer, save_method="merged_16bit", token="hf_...")
```

> **Teaching point:** Training format (4-bit) and serving format (GGUF or 16-bit) are separate decisions. You train once, then export to multiple artifacts.

---

## 9. Ollama Deployment

### 9.1 Create Modelfile and Import

Unsloth auto-generates a `Modelfile` next to the GGUF. You can use it directly or customize it:

```dockerfile
# pizza_gguf/Modelfile (auto-generated by Unsloth)
FROM ./pizza_gguf/unsloth.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}chat
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}chat
{{ end }}<|im_start|>assistant
{{ .Response }}chat
"""

PARAMETER stop "chat"
PARAMETER temperature 0.7
```

> Note: The `imates` sentinel in the template is actually `im_end` control token displayed with odd characters due to escaping. In your real `Modelfile` it will look like `<|im_end|>` or the actual token.

Run locally:

```bash
ollama create pizza-palace -f pizza_gguf/Modelfile
ollama run pizza-palace

# Test
>>> what veg pizzas do you have?
```

### 9.2 Custom Modelfile (More Explicit)

```dockerfile
FROM ./pizza_gguf/unsloth.Q4_K_M.gguf

SYSTEM """You are a warm, upbeat customer support agent for Pizza Palace.
Today's menu: (copy your full menu here)
Use ONLY this menu. Greet warmly, help genuinely, offer a relevant add-on, and stay cheerful with complaints."""

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop <|im_end|>
PARAMETER temperature 0.4
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
```

```bash
ollama create pizza-palace -f Modelfile
ollama run pizza-palace "what veg pizzas do you have?"
```

---

## 10. vLLM Deployment

### 10.1 Serve Merged 16-bit Model

```bash
# On a GPU box (e.g., GCP L4, A100)
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model /path/to/pizza_merged_16bit \
    --served-model-name pizza-palace \
    --max-model-len 1024 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9
```

### 10.2 Test with curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pizza-palace",
    "messages": [
      {"role":"system","content":"You are a Pizza Palace agent. Menu: Margherita ₹210, Cola ₹60. Use only this menu."},
      {"role":"user","content":"veg options?"}
    ],
    "temperature": 0.4
  }'
```

### 10.3 vLLM vs Ollama Decision

| Need | Use |
|------|-----|
| One user, laptop, no GPU | Ollama |
| Small team, Docker | Ollama in container |
| 10+ concurrent users | vLLM |
| Kubernetes, auto-scaling | vLLM Production Stack (Helm) |
| Latency-sensitive production | vLLM with continuous batching |

---

## 11. Spring AI Integration

### 11.1 Ollama Backend

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
spring.ai.ollama.chat.options.model=pizza-palace
spring.ai.ollama.chat.options.temperature=0.4
spring.ai.ollama.chat.options.num-predict=128
```

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

@Service
public class PizzaPalaceService {

    private final ChatClient chatClient;

    public PizzaPalaceService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public String chat(String userMessage) {
        return chatClient.prompt()
            .user(userMessage)
            .call()
            .content();
    }

    public Flux<String> streamChat(String userMessage) {
        return chatClient.prompt()
            .user(userMessage)
            .stream()
            .content();
    }
}
```

```java
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/api/pizza-palace")
public class PizzaPalaceController {

    private final PizzaPalaceService service;

    public PizzaPalaceController(PizzaPalaceService service) {
        this.service = service;
    }

    // POST /api/pizza-palace/chat  body: "what veg pizzas do you have?"
    @PostMapping(consumes = MediaType.TEXT_PLAIN_VALUE)
    public String chat(@RequestBody String message) {
        return service.chat(message);
    }

    // POST /api/pizza-palace/stream  — SSE token streaming
    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE,
                 consumes = MediaType.TEXT_PLAIN_VALUE)
    public Flux<String> stream(@RequestBody String message) {
        return service.streamChat(message);
    }
}
```

### 11.2 vLLM Backend (Production)

```properties
# application.properties
spring.ai.openai.base-url=http://gpu-server:8000/v1
spring.ai.openai.api-key=not-needed
spring.ai.openai.chat.options.model=pizza-palace
spring.ai.openai.chat.options.temperature=0.4
spring.ai.openai.chat.options.max-tokens=128
```

```java
@Service
public class PizzaPalaceVllmService {

    private final ChatClient chatClient;

    public PizzaPalaceVllmService(ChatClient.Builder builder) {
        this.chatClient = builder
            .defaultSystem("""
                You are a warm, upbeat customer support agent for Pizza Palace.
                Use ONLY the menu in the context. Be concise and helpful.
                """)
            .build();
    }

    public String handle(String customerMessage) {
        return chatClient.prompt()
            .user(customerMessage)
            .call()
            .content();
    }
}
```

### 11.3 Full Architecture

```
Customer
   │
   ▼
Spring Boot (ChatClient)
   │
   ├─────── Ollama (local dev) ───────► pizza-palace GGUF
   │
   └─────── vLLM (production) ────────► pizza-palace merged 16-bit

Training happens once in Colab.
Deployment is a simple config change in application.properties.
```

---

## 12. Key Takeaways & Cheat Sheet

### 12.1 Complete Workflow

```
1. Open Colab → File → Open Notebook → GitHub
   URL: Qwen3_PizzaPalace_finetune.ipynb

2. Change runtime to T4 GPU

3. Upload pizza_palace_400_train.jsonl & _val.jsonl

4. Run all cells:
   ├─ Install Unsloth
   ├─ Load Qwen3-4B in 4-bit
   ├─ Attach LoRA r=16, alpha=16
   ├─ Load JSONL datasets
   ├─ Apply Qwen3 chat template
   ├─ Configure SFTTrainer
   ├─ Enable train_on_responses_only
   ├─ Train for 3 epochs
   ├─ Plot train/val loss
   ├─ Quick sanity test
   ├─ Save LoRA adapters
   ├─ Export GGUF for Ollama
   └─ Export merged 16-bit for vLLM

5. Download artifacts

6. Deploy:
   ├─ Ollama (local/edge): ollama create + ollama run
   └─ vLLM (production): vllm serve

7. Consume from Spring AI:
   ├─ Ollama: spring-ai-ollama-spring-boot-starter
   └─ vLLM: spring-ai-openai-spring-boot-starter
```

### 12.2 Artifact Selection Table

| Artifact | Method | Serve With | Spring AI Dependency |
|----------|--------|-----------|---------------------|
| LoRA adapter | `save_pretrained` | HuggingFace PEFT | — |
| GGUF | `save_pretrained_gguf` | Ollama, llama.cpp | `spring-ai-ollama-spring-boot-starter` |
| Merged 16-bit | `save_pretrained_merged` | vLLM, TGI, HF | `spring-ai-openai-spring-boot-starter` (OpenAI-compatible) |

### 12.3 Hyperparameters for This Run

```python
model     = "unsloth/Qwen3-4B"
max_seq_length = 1024
r         = 16
alpha     = 16
lr        = 2e-4
batch     = 2
grad_accum = 4
effective_batch = 8
epochs    = 3
optimizer = "adamw_8bit"
scheduler = "linear"
warmup    = 5 steps
```

### 12.4 Common Colab Errors

| Error | Quick Fix |
|-------|-----------|
| `T4 not available` | Change runtime, try again, or use CPU (slow) |
| `CUDA OOM` | `batch_size=1`, `max_seq_length=512`, or use Qwen3-1.5B |
| `loss NaN` | `learning_rate=1e-4` |
| `model not found` | Check `unsloth/Qwen3-4B` is accessible on HuggingFace |
| `Modelfile not working` | Verify `im_start` / `im_end` tokens and stop parameter |

### 12.5 Glossary

| Term | Definition |
|------|-----------|
| **GGUF** | llama.cpp/Ollama model format with built-in quantization |
| **Merged 16-bit** | LoRA adapters merged into base at 16-bit precision for vLLM/HF |
| **QLoRA** | LoRA where the frozen base model is 4-bit quantized |
| **ChatML** | Qwen's chat format using `<|im_start|>` and `<|im_end|>` tokens |
| **`train_on_responses_only`** | Masking so gradients only update assistant turns |
| **`enable_thinking`** | Qwen3 flag to include/exclude reasoning blocks |
| **T4 GPU** | Free Colab GPU with ~15 GB VRAM |
| **Ollama** | Easy local model server with Modelfiles |
| **vLLM** | Production-grade GPU inference engine |

---

> **Next Session Preview:** Python classroom notes — likely evaluation, testing the bot, or deploying to a real web app.

---

*Notes compiled with deep-dive explanations, examples, and use cases based on:*
*https://directai.blog/2026/08/13/gen-ai-developer-classroom-notes-13-aug-2026/*
*and the Colab notebook: https://github.com/GenAIDevelopment/agenticai/blob/main/aug26/finetune/Qwen3_PizzaPalace_finetune.ipynb*
