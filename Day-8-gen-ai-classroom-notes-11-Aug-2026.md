# DAY-8: Gen-AI Developer Classroom Notes — 11 Aug 2026

> **Source:** https://directai.blog/2026/08/11/gen-ai-developer-classroom-notes-11-aug-2026/
> **Dataset Source:** https://github.com/GenAIDevelopment/agenticai/blob/main/june26/dataset/pizza_palace_400.jsonl
> **Audience:** Gen-AI Developers (Beginner → Intermediate)

---

## Table of Contents

1. [PEFT, LoRA & QLoRA Recap for the Pizza Palace Project](#1-peft-lora--qlora-recap-for-the-pizza-palace-project)
2. [The Use Case: Pizza Palace Customer Service Chatbot](#2-the-use-case-pizza-palace-customer-service-chatbot)
3. [Designing the System Prompt](#3-designing-the-system-prompt)
4. [Building the Training Dataset (300–400 Conversations)](#4-building-the-training-dataset-300400-conversations)
5. [Train / Validation Split (80–20)](#5-train--validation-split-8020)
6. [Fine-Tuning on Google Colab with Unsloth](#6-fine-tuning-on-google-colab-with-unsloth)
7. [Deploying the Fine-Tuned Model](#7-deploying-the-fine-tuned-model)
8. [Spring AI Integration](#8-spring-ai-integration)
9. [Key Takeaways & Cheat Sheet](#9-key-takeaways--cheat-sheet)

---

## 1. PEFT, LoRA & QLoRA Recap for the Pizza Palace Project

### 1.1 The Core Idea Applied to Pizza Palace

For a real chatbot, you do **not** need to retrain the entire 8 billion-parameter base model. You only need to teach it how to speak like Pizza Palace support while keeping the menu accurate.

```
Base Model (Llama 3 8B Instruct)
├── Already knows English, grammar, conversation basics
├── Already knows what a pizza is
└── Does NOT know Pizza Palace's tone, upsell style, or apology format

With LoRA / QLoRA:
  Add small adapter matrices (A and B) to attention/MLP layers
  Only train ~0.3% of parameters
  Result: model responds in warm, upbeat Pizza Palace style
```

### 1.2 PEFT = Parameter Efficient Fine-Tuning

> **PEFT** is the umbrella term for methods that fine-tune only a small fraction of a model's weights. **LoRA** and **QLoRA** are the most popular PEFT techniques.

```
┌────────────────────────────────────────────────────────────┐
│                    PEFT FAMILY TREE                         │
│                                                             │
│  Full Fine-Tuning (FFT) ── updates ALL weights             │
│       │                                                     │
│       └── Expensive, slow, risk of catastrophic forgetting │
│                                                             │
│  Parameter Efficient Fine-Tuning (PEFT)                     │
│       │                                                     │
│       ├── LoRA  ── adds two small matrices A and B         │
│       ├── QLoRA ── LoRA + 4-bit quantization                │
│       ├── Prompt Tuning ── learns soft prompt embeddings     │
│       ├── P-Tuning ── learns continuous prompts              │
│       └── Prefix Tuning ── prepends trainable vectors      │
│                                                             │
│  For the Pizza Palace bot: use QLoRA (free on Colab)       │
└────────────────────────────────────────────────────────────┘
```

### 1.3 LoRA — The "Two Matrices" Trick

Instead of updating the full weight matrix `W` (e.g., `4096 × 4096` = 16.7M parameters), LoRA adds two thin matrices:

```
Original weight update:    ΔW  = new - old     (16.7M parameters)
LoRA approximation:      ΔW  = B × A
                              A = [r × 4096]
                              B = [4096 × r]
                              with r = 16 → only 131,072 new parameters!

For Llama 3 8B:
  Total parameters:   ~8,000,000,000
  LoRA parameters:    ~24,000,000 (only 0.3% trainable)
```

### 1.4 QLoRA — LoRA with 4-Bit Quantization

QLoRA stores the **base model in 4-bit precision** (saving ~4× VRAM) while training the LoRA adapters in **bfloat16**.

```
Without QLoRA (LoRA 16-bit):
  Llama 3 8B ≈ 16 GB VRAM needed

With QLoRA (4-bit base + bf16 adapters):
  Llama 3 8B ≈ 6–8 GB VRAM needed
  → Fits on free Google Colab T4 GPU (15 GB VRAM)
```

### 1.5 Why This Is Perfect for Pizza Palace

- **Small dataset**: 300–400 conversations is enough for a focused behavior task
- **Low cost**: QLoRA runs free on Colab
- **Fast iteration**: Try r=8, then r=16, then r=32 in under an hour each
- **No need for local GPU**: Google Colab provides a T4 for free

---

## 2. The Use Case: Pizza Palace Customer Service Chatbot

### 2.1 Business Requirements

Build a chatbot that:

1. **Knows today's menu** — pizzas, sides, desserts, add-ons, prices
2. **Speaks in a warm, upbeat tone** — not robotic, not pushy
3. **Handles complaints politely** — apologizes, offers replacement/refund
4. **Upsells naturally** — suggests add-ons like Garlic Bread or Cold Coffee when relevant
5. **Stays within the menu** — does not hallucinate items or prices

### 2.2 Behavior vs Facts Again

| Aspect | Is It Behavior or Facts? | Solution |
|--------|-------------------------|----------|
| Warm, upbeat tone | **Behavior** | Fine-tune with consistent assistant responses |
| "Fancy some Garlic Bread on the side?" upsell style | **Behavior** | Fine-tune — this is a recurring pattern |
| Today's specific pizza prices | **Facts** | Put in **system prompt** (or use RAG if prices change daily) |
| Handling refund complaints | **Behavior** | Fine-tune with apologetic response templates |
| Whether calzones are available | **Facts** | System prompt says "Use ONLY this menu" |

> **Design decision:** For this exercise, the menu is treated as a static fact embedded in the system prompt. In production, if prices change daily, you would move the menu to a RAG/vector store and only fine-tune the tone.

### 2.3 Why 300–400 Conversations?

- **< 100 examples**: Model may not learn the style consistently
- **100–300 examples**: Often enough for simple tone/style tasks
- **300–400 examples**: Recommended for a robust customer service persona with multiple intents (ordering, complaints, recommendations, out-of-menu queries)
- **> 1000 examples**: Diminishing returns for style-only fine-tuning; better for complex reasoning

---

## 3. Designing the System Prompt

### 3.1 What Goes Into the System Prompt

The system prompt is the "context card" given to every conversation. In this dataset, it contains:

```
You are a warm, upbeat customer support agent for Pizza Palace. Today's menu:
<list every item with price>
Use ONLY this menu.
Greet warmly, help genuinely, offer a relevant add-on when it fits naturally,
and stay cheerful and apologetic with complaints.
```

### 3.2 Why the Menu Is in the System Prompt (Not Fine-Tuned In)

| Approach | Pros | Cons |
|----------|------|------|
| **Menu in system prompt** | Easy to update daily; no retraining; prices always current | Longer context window usage |
| **Menu in fine-tuning data** | Model "memorizes" prices | Model becomes stale when prices change; retraining needed |

For the learning exercise, embedding the menu in every training example also **teaches the model to only recommend items that appear in the system prompt**, preventing hallucinations.

### 3.3 Sample System Prompt from Dataset

```text
You are a warm, upbeat customer support agent for Pizza Palace. Today's menu:
Mushroom Magic ₹250
Tandoori Paneer ₹360
Corn & Cheese ₹290
Paneer Supreme ₹350
Meat Lovers ₹390
BBQ Chicken ₹350
Keema Kick ₹440
Peri Peri Fries ₹170
Choco Lava Cake ₹120
Chicken Wings ₹220
Cola ₹60
Use ONLY this menu. Greet warmly, help genuinely, offer a relevant add-on when it fits naturally,
and stay cheerful and apologetic with complaints.
```

### 3.4 Assistant Response Patterns Observed in Dataset

From the real `pizza_palace_400.jsonl`, the assistant follows consistent behaviors:

| Customer Intent | Assistant Pattern | Example |
|-----------------|-------------------|---------|
| **Recommendation** | Greets + suggests 2 items + upsells add-on | "We've got some gems — Garden Delight, Paneer Supreme... Add a Cold Coffee?" |
| **Order placement** | Confirms item + price + suggests add-on | "You got it — one Farmhouse (₹310) coming up! Want a Garlic Bread (₹110)?" |
| **Complaint** | Apologizes + offers replacement/refund + asks order number | "I'm truly sorry... I'll send a fresh replacement or a full refund... share your order number" |
| **Out-of-menu item** | Politely declines + suggests closest alternative | "We don't have that on today's menu, but... our Butter Chicken Pizza (₹470) is a wonderful alternative" |
| **Budget request** | Suggests combination within budget | "Within ₹1,000 I'd suggest one Chicken Supreme (₹420), one Farmhouse (₹310) and one Margherita (₹210)" |

---

## 4. Building the Training Dataset (300–400 Conversations)

### 4.1 The JSONL Format

Each line is a single conversation:

```jsonl
{"messages": [
  {"role": "system", "content": "You are a warm, upbeat customer support agent..."},
  {"role": "user", "content": "do u deliver after 11pm"},
  {"role": "assistant", "content": "Hey, thanks for stopping by! Great question! I'll check your branch's details if you tell me where you are. Fancy some Chicken Wings (₹220) on the side?"}
]}
```

### 4.2 Generating 300–400 Examples with Claude / Any LLM

You do not need to write 400 conversations manually. Use a stronger model (Claude, GPT-4o) to generate diverse synthetic training data.

#### Python Script to Generate Pizza Palace Dataset

```python
"""
pizza_dataset_generator.py
Generates synthetic fine-tuning data for Pizza Palace customer support.
Output: pizza_palace_train.jsonl
"""

import json
import random
from openai import OpenAI
from tqdm import tqdm  # progress bar

client = OpenAI()  # uses OPENAI_API_KEY env var

# ── Menu definition ───────────────────────────────────────────────
MENU_ITEMS = [
    ("Margherita", 190, ["veg"]),
    ("Farmhouse", 310, ["veg"]),
    ("Peppy Paneer", 350, ["veg"]),
    ("Paneer Supreme", 350, ["veg"]),
    ("Corn & Cheese", 290, ["veg"]),
    ("Veggie Feast", 350, ["veg"]),
    ("Tandoori Paneer", 360, ["veg"]),
    ("Chicken Tikka", 360, ["non-veg"]),
    ("Chicken Supreme", 420, ["non-veg"]),
    ("BBQ Chicken", 390, ["non-veg"]),
    ("Meat Lovers", 480, ["non-veg"]),
    ("Garlic Bread", 120, ["side"]),
    ("Cheese Garlic Bread", 150, ["side"]),
    ("Peri Peri Fries", 170, ["side"]),
    ("Chicken Wings", 220, ["side"]),
    ("Choco Lava Cake", 120, ["dessert"]),
    ("Cola", 60, ["beverage"]),
    ("Cold Coffee", 140, ["beverage"]),
]

def build_menu_text():
    return "\n".join([f"{name} ₹{price}" for name, price, _ in MENU_ITEMS])

SYSTEM_PROMPT = f"""You are a warm, upbeat customer support agent for Pizza Palace. Today's menu:
{build_menu_text()}
Use ONLY this menu. Greet warmly, help genuinely, offer a relevant add-on when it fits naturally, and stay cheerful and apologetic with complaints."""

# ── Seed customer intents ────────────────────────────────────────
INTENTS = [
    "recommend me something",
    "what veg options do you have?",
    "what non-veg do you have?",
    "one farmhouse please",
    "do you deliver after 11pm",
    "app charged me twice",
    "it's been an hour, still nothing",
    "do you have a calzone",
    "budget around 1000 for 3 pizzas, suggest",
    "no mushrooms my wife hates them",
    "my kids devoured it, thank you",
    "breadsticks were stale",
    "there was a hair in it disgusting",
    "most loaded pizza you have?",
    "do you have a combo deal",
    "can i use my loyalty points",
    "how far do you deliver",
    "whatever's cheapest, i'm broke",
    "double cheese on that",
    "tandoori momos?",
]

def build_messages(user_message: str, assistant_response: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ]
    }

def generate_conversation(intent: str) -> dict:
    """Ask GPT-4o-mini to produce one assistant response for a given intent."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are creating training data for a warm, upbeat Pizza Palace chatbot.
Given a customer intent and the menu, write ONE realistic 'user' message and the matching 'assistant' response.
- User message should be short, casual, sometimes with typos.
- Assistant response must be warm, helpful, sometimes suggest a relevant add-on.
- For complaints: apologize sincerely and offer replacement/refund + ask for order number.
- NEVER recommend items not in the menu.
- Output valid JSON: {"user": "...", "assistant": "..."}"""
            },
            {
                "role": "user",
                "content": f"""Today's menu:\n{build_menu_text()}\n\nCustomer intent: {intent}\n\nGenerate one user message and the assistant response as JSON."""
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.9,
    )
    result = json.loads(response.choices[0].message.content)
    return build_messages(result["user"], result["assistant"])

# ── Generate 400 examples ───────────────────────────────────────
OUTPUT_FILE = "pizza_palace_400.jsonl"
NUM_SAMPLES = 400

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i in tqdm(range(NUM_SAMPLES)):
        intent = random.choice(INTENTS)
        try:
            conv = generate_conversation(intent)
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error at sample {i}: {e}")

print(f"Generated {NUM_SAMPLES} conversations in {OUTPUT_FILE}")
```

### 4.3 Cost Estimate for Synthetic Generation

| Model | Cost per 1K Tokens | ~Tokens per Conversation | Cost per 400 Conversations |
|-------|-------------------|--------------------------|----------------------------|
| GPT-4o-mini | $0.0006 / 1M tokens | ~300 tokens | ~$0.07 |
| GPT-4o | $0.005 / 1M tokens | ~300 tokens | ~$0.60 |
| Claude 3.5 Haiku | $0.00025 / 1K input | ~300 tokens | ~$0.30 |
| Llama 3.1 70B (local) | Free (if you have GPU) | ~300 tokens | Free |

> **Tip:** GPT-4o-mini is the cheapest and fastest way to bootstrap 400 examples. Always review the generated dataset for quality before fine-tuning.

### 4.4 Validating the Dataset

```python
"""validate_pizza_dataset.py"""
import json
from collections import Counter

input_file = "pizza_palace_400.jsonl"

issues = []
intent_counts = Counter()
assistant_lengths = []

with open(input_file, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            issues.append(f"Line {line_no}: invalid JSON - {e}")
            continue

        # Structure validation
        if "messages" not in obj or not isinstance(obj["messages"], list):
            issues.append(f"Line {line_no}: missing 'messages' key")
            continue

        if len(obj["messages"]) != 3:
            issues.append(f"Line {line_no}: expected 3 messages, got {len(obj['messages'])}")

        roles = [m.get("role") for m in obj["messages"]]
        if roles != ["system", "user", "assistant"]:
            issues.append(f"Line {line_no}: wrong roles {roles}")

        contents = [m.get("content", "") for m in obj["messages"]]
        if any(len(c) == 0 for c in contents):
            issues.append(f"Line {line_no}: empty content detected")

        # Track stats
        user_msg = contents[1].lower() if len(contents) > 1 else ""
        assistant_lengths.append(len(contents[2]) if len(contents) > 2 else 0)

        # Simple intent categorization
        if any(w in user_msg for w in ["recommend", "suggest", "best"]):
            intent_counts["recommendation"] += 1
        elif any(w in user_msg for w in ["order", "one ", "give me"]):
            intent_counts["order"] += 1
        elif any(w in user_msg for w in ["charged", "hour", "stale", "hair", "late", "wrong"]):
            intent_counts["complaint"] += 1
        elif any(w in user_msg for w in ["have", "calzone", "momo"]):
            intent_counts["availability"] += 1
        else:
            intent_counts["other"] += 1

print("=== Validation Report ===")
if issues:
    print(f"Found {len(issues)} issues:")
    for issue in issues[:10]:  # show first 10
        print(f"  - {issue}")
else:
    print("No structural issues found.")

print("\n=== Intent Distribution ===")
for intent, count in intent_counts.most_common():
    print(f"  {intent}: {count}")

print(f"\n=== Assistant Response Stats ===")
print(f"  Average length: {sum(assistant_lengths) / len(assistant_lengths):.0f} chars")
print(f"  Min length: {min(assistant_lengths)}")
print(f"  Max length: {max(assistant_lengths)}")
```

---

## 5. Train / Validation Split (80–20)

### 5.1 Why Split?

The **training set** teaches the model the patterns. The **validation set** tells you whether the model is memorizing or generalizing.

```
400 total examples
├── 320 training examples (80%)  → used to update LoRA weights
└── 80 validation examples (20%) → used to compute val_loss, detect overfitting
```

### 5.2 Python Split Script

```python
"""split_pizza_dataset.py"""
import json
import random

INPUT_FILE = "pizza_palace_400.jsonl"
TRAIN_FILE = "pizza_palace_train.jsonl"
VAL_FILE = "pizza_palace_val.jsonl"
VAL_RATIO = 0.20
SEED = 42

# Read all lines
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

# Shuffle deterministically
random.seed(SEED)
random.shuffle(lines)

# Split
split_idx = int(len(lines) * (1 - VAL_RATIO))
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

write_jsonl(train_lines, TRAIN_FILE)
write_jsonl(val_lines, VAL_FILE)

print(f"Train: {len(train_lines)} examples -> {TRAIN_FILE}")
print(f"Val:   {len(val_lines)} examples -> {VAL_FILE}")

# Output:
# Train: 320 examples -> pizza_palace_train.jsonl
# Val:   80 examples -> pizza_palace_val.jsonl
```

### 5.3 Stratified Split (Better — Keep Intent Ratios)

To make sure the validation set has the same mix of intents as the training set, use **stratified sampling**:

```python
from collections import defaultdict

def stratified_split(data, val_ratio=0.2, seed=42):
    """Split data into train/val while preserving intent ratios."""
    random.seed(seed)

    # Group by intent (simplified: use first keyword found)
    buckets = defaultdict(list)
    for item in data:
        user = item["messages"][1]["content"].lower()
        if "recommend" in user or "suggest" in user or "best" in user:
            intent = "recommendation"
        elif any(w in user for w in ["charged", "hour", "stale", "hair", "late"]):
            intent = "complaint"
        elif any(w in user for w in ["order", "give me", "one ", "i want"]):
            intent = "order"
        else:
            intent = "other"
        buckets[intent].append(item)

    train, val = [], []
    for intent, items in buckets.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val

# Use stratified split
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

train_lines, val_lines = stratified_split(lines, val_ratio=0.2, seed=42)
write_jsonl(train_lines, TRAIN_FILE)
write_jsonl(val_lines, VAL_FILE)
```

---

## 6. Fine-Tuning on Google Colab with Unsloth

### 6.1 Colab Notebook Setup

```python
# ═══════════════════════════════════════════════════════════════════
# Cell 1: Install Unsloth
# ═══════════════════════════════════════════════════════════════════
!pip install unsloth
!pip install --no-deps trl peft accelerate bitsandbytes
```

```python
# ═══════════════════════════════════════════════════════════════════
# Cell 2: Upload pizza_palace_train.jsonl and pizza_palace_val.jsonl
# Use the Colab file browser or mount Google Drive
# ═══════════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

# Copy files from Drive to Colab
!cp /content/drive/MyDrive/pizza_palace_400.jsonl /content/
```

```python
# ═══════════════════════════════════════════════════════════════════
# Cell 3: Load Dataset
# ═══════════════════════════════════════════════════════════════════
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "/content/pizza_palace_train.jsonl",
    "eval": "/content/pizza_palace_val.jsonl"
})

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['messages'],
#         num_rows: 320
#     })
#     eval: Dataset({
#         features: ['messages'],
#         num_rows: 80
#     })
# })
```

### 6.2 Load Model and Attach LoRA

```python
# ═══════════════════════════════════════════════════════════════════
# Cell 4: Load Base Model (QLoRA)
# ═══════════════════════════════════════════════════════════════════
import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # auto: bf16 on Ampere+, fp16 otherwise
    load_in_4bit=True,
)

# ═══════════════════════════════════════════════════════════════════
# Cell 5: Attach LoRA Adapters
# ═══════════════════════════════════════════════════════════════════
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                       # Rank: multiple of 8
    lora_alpha=32,              # 2 × r — aggressive learning for clear style
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",       # Attention
        "gate_proj", "up_proj", "down_proj"           # MLP/FFN
    ],
    lora_dropout=0,             # 0 is optimized in Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% extra VRAM savings
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

### 6.3 Configure Training

```python
# ═══════════════════════════════════════════════════════════════════
# Cell 6: Training Arguments
# ═══════════════════════════════════════════════════════════════════
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="/content/pizza_palace_lora",
    num_train_epochs=2,                 # 1-3 is safe; 2 for this style task
    per_device_train_batch_size=2,      # Colab T4 safe
    gradient_accumulation_steps=8,      # Effective batch = 2 × 8 = 16
    learning_rate=2e-4,                 # Standard LoRA starting point
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=40,
    save_strategy="steps",
    save_steps=40,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    seed=3407,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    dataset_text_field="text",  # or use chat_template if already applied
    max_seq_length=2048,
    args=training_args,
    packing=False,  # Don't pack short sequences — each conv is independent
)

# Train only on assistant responses (recommended for chat data)
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

### 6.4 Train and Save

```python
# ═══════════════════════════════════════════════════════════════════
# Cell 7: Train!
# ═══════════════════════════════════════════════════════════════════
trainer_stats = trainer.train()

print(f"Training time: {trainer_stats.metrics['train_runtime']:.1f}s")

# Save LoRA adapter
model.save_pretrained("/content/pizza_palace_adapter")
tokenizer.save_pretrained("/content/pizza_palace_adapter")

# Push to HuggingFace Hub (optional)
# model.push_to_hub("your-username/pizza-palace-lora-llama-3.1-8b")
# tokenizer.push_to_hub("your-username/pizza-palace-lora-llama-3.1-8b")
```

### 6.5 Expected Colab Output

```
TrainOutput(global_step=320, training_loss=0.8234, ...)
Training time: ~900s (15 minutes on Colab T4)

Memory used: ~6.5 GB / 15 GB
Trainable params: ~24,000,000 (0.3% of 8B)
```

---

## 7. Deploying the Fine-Tuned Model

### 7.1 Export to GGUF for Ollama

```python
# ═══════════════════════════════════════════════════════════════════
# Cell 8: Export Fine-Tuned Model to GGUF
# ═══════════════════════════════════════════════════════════════════
# Merge LoRA weights back into base model for single-file serving
model.save_pretrained_gguf(
    "/content/pizza_palace_gguf",
    tokenizer,
    quantization_method="q4_k_m",  # 4-bit, good quality/speed tradeoff
)

# Download from Colab to local machine
from google.colab import files
files.download("/content/pizza_palace_gguf/unsloth.Q4_K_M.gguf")
```

### 7.2 Create Ollama Modelfile

```dockerfile
# Modelfile for Pizza Palace bot
FROM /path/to/unsloth.Q4_K_M.gguf

SYSTEM """You are a warm, upbeat customer support agent for Pizza Palace. Today's menu:
Margherita ₹190
Farmhouse ₹310
Peppy Paneer ₹350
Paneer Supreme ₹350
Corn & Cheese ₹290
Veggie Feast ₹350
Tandoori Paneer ₹360
Chicken Tikka ₹360
Chicken Supreme ₹420
BBQ Chicken ₹390
Meat Lovers ₹480
Garlic Bread ₹120
Cheese Garlic Bread ₹150
Peri Peri Fries ₹170
Chicken Wings ₹220
Choco Lava Cake ₹120
Cola ₹60
Cold Coffee ₹140
Use ONLY this menu. Greet warmly, help genuinely, offer a relevant add-on when it fits naturally, and stay cheerful and apologetic with complaints."""

PARAMETER temperature 0.4
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
```

```bash
# On your local machine
ollama create pizza-palace-bot -f Modelfile
ollama run pizza-palace-bot

# Test
>>> what veg pizzas do you have?
```

### 7.3 Alternative: vLLM Production Deployment

```bash
# pip install vllm
# Serve the merged 16-bit model or original + LoRA adapter

python -m vllm.entrypoints.openai.api_server \
  --model your-username/pizza-palace-merged-llama-3.1-8b \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --port 8000

# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pizza-palace",
    "messages": [
      {"role": "user", "content": "what veg pizzas do you have?"}
    ],
    "temperature": 0.4
  }'
```

---

## 8. Spring AI Integration

### 8.1 Ollama Backend (Local Development)

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
spring.ai.ollama.chat.options.model=pizza-palace-bot
spring.ai.ollama.chat.options.temperature=0.4
spring.ai.ollama.chat.options.top-p=0.9
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

    public String chat(String message) {
        return chatClient.prompt()
            .user(message)
            .call()
            .content();
    }

    public Flux<String> streamChat(String message) {
        return chatClient.prompt()
            .user(message)
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
@RequestMapping("/api/pizza")
public class PizzaPalaceController {

    private final PizzaPalaceService service;

    public PizzaPalaceController(PizzaPalaceService service) {
        this.service = service;
    }

    @PostMapping(consumes = MediaType.TEXT_PLAIN_VALUE)
    public String chat(@RequestBody String message) {
        return service.chat(message);
    }

    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE,
                 consumes = MediaType.TEXT_PLAIN_VALUE)
    public Flux<String> stream(@RequestBody String message) {
        return service.streamChat(message);
    }
}
```

### 8.2 vLLM / OpenAI-Compatible Backend (Production)

```properties
# application.properties — point to vLLM or any OpenAI-compatible server
spring.ai.openai.base-url=http://localhost:8000/v1
spring.ai.openai.api-key=not-needed
spring.ai.openai.chat.options.model=pizza-palace
spring.ai.openai.chat.options.temperature=0.4
```

```java
@Service
public class PizzaPalaceVllmService {

    private final ChatClient chatClient;

    public PizzaPalaceVllmService(ChatClient.Builder builder) {
        this.chatClient = builder
            .defaultSystem("""
                You are a warm, upbeat customer support agent for Pizza Palace.
                Use ONLY the menu provided. Greet warmly, help genuinely, and stay cheerful.
                """)
            .build();
    }

    public String handleCustomer(String message) {
        return chatClient.prompt()
            .user(message)
            .call()
            .content();
    }
}
```

### 8.3 Architecture for Production

```
┌─────────────────────────────────────────────────────────────────┐
│                PIZZA PALACE CHATBOT PRODUCTION                  │
│                                                                 │
│   Customer Message                                              │
│        │                                                        │
│        ▼                                                        │
│   Spring Boot App (ChatClient + VectorStore optional)          │
│        │                                                        │
│        ▼                                                        │
│   Ollama (local dev)  or  vLLM (prod)                          │
│        │                                                        │
│        ▼                                                        │
│   Fine-Tuned Llama 3 8B GGUF / 16-bit model                    │
│   └── LoRA adapters encode warm, upbeat Pizza Palace tone       │
│                                                                 │
│   System prompt carries today's menu + policies                │
│   (update daily without retraining)                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Key Takeaways & Cheat Sheet

### 9.1 Project Workflow Recap

```
1. Define use case         → Pizza Palace customer support
2. Decide behavior vs facts → Tone = fine-tune; Menu = system prompt
3. Write system prompt     → Warm persona + today's menu + rules
4. Generate 400 examples  → Use GPT-4o-mini / Claude with seed intents
5. Validate dataset        → Check JSONL structure, intent distribution
6. Split 80/20             → 320 train, 80 validation
7. Fine-tune on Colab      → Unsloth QLoRA, r=16, alpha=32, lr=2e-4
8. Export to GGUF          → Save for Ollama / vLLM
9. Deploy                  → Ollama (dev) or vLLM (prod)
10. Integrate with Spring AI → ChatClient pointing to local/prod endpoint
```

### 9.2 Pizza Palace Fine-Tune Hyperparameters (Copy-Paste Defaults)

```python
r                  = 16
lora_alpha         = 32          # = 2 × r
lora_dropout       = 0
target_modules     = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
learning_rate      = 2e-4
batch_size         = 2
gradient_accum     = 8           # effective = 16
num_epochs         = 2
weight_decay       = 0.01
scheduler          = "cosine"
warmup_ratio       = 0.05
max_seq_length     = 2048
seed               = 3407
```

### 9.3 JSONL Quality Checklist

```
□ 300–400 lines, each valid JSON
□ Each line has exactly 3 messages: system, user, assistant
□ System prompt is consistent across all examples
□ User messages are varied (recommend, order, complaint, availability, budget)
□ Assistant responses match the target tone
□ No item is recommended that is not in the system prompt menu
□ No empty assistant responses
□ Train/validation split created (80/20)
□ Validation set covers all major intents
```

### 9.4 When to Choose Ollama vs vLLM

| Scenario | Backend | Spring AI Dependency |
|----------|---------|---------------------|
| Local development, single user | Ollama | `spring-ai-ollama-spring-boot-starter` |
| Small team, Docker deploy | Ollama in Docker | `spring-ai-ollama-spring-boot-starter` |
| Production, high traffic | vLLM | `spring-ai-openai-spring-boot-starter` (OpenAI-compatible) |
| Cloud fine-tuned model (OpenAI) | OpenAI | `spring-ai-openai-spring-boot-starter` |

### 9.5 Glossary

| Term | Definition |
|------|-----------|
| **PEFT** | Parameter Efficient Fine-Tuning — update only a small fraction of weights |
| **LoRA** | Low-Rank Adaptation — adds two small matrices A and B |
| **QLoRA** | LoRA with 4-bit quantized base model |
| **JSONL** | JSON Lines — one JSON object per line |
| **System prompt** | Instructions and context prepended to every conversation |
| **Synthetic dataset** | Training data generated by another LLM |
| **Stratified split** | Train/val split that preserves class/intent ratios |
| **GGUF** | llama.cpp/Ollama model file format with quantization |
| **Colab T4** | Free NVIDIA T4 GPU provided by Google Colab (~15 GB VRAM) |
| **Ollama** | Easy local LLM server with Modelfiles |
| **vLLM** | High-throughput GPU inference server for production |

---

> **Next Session Preview:** Continued Python classroom notes — likely covering dataset processing, prompt engineering, or evaluation metrics.

---

*Notes compiled with deep-dive explanations, examples, and use cases based on:*
*https://directai.blog/2026/08/11/gen-ai-developer-classroom-notes-11-aug-2026/*
