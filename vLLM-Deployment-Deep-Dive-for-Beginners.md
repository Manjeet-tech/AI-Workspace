# Deep Dive: vLLM Deployment for the Qwen3 Pizza Palace Chatbot

> **Linked Section:** `d0854eca` — "Serve with vLLM" from the [Qwen3_PizzaPalace_finetune.ipynb](https://github.com/GenAIDevelopment/agenticai/blob/main/aug26/finetune/Qwen3_PizzaPalace_finetune.ipynb) notebook
> **Audience:** Beginners to Gen-AI deployment
> **Prerequisite:** You have already completed the QLoRA fine-tune in Colab (Day-9 notes)

---

## 1. The Big Picture: Why vLLM Exists

After you finish training your chatbot in Colab, you have a model that understands the warm, upbeat Pizza Palace tone. But a Jupyter notebook is not an API. To let real customers talk to the bot, you need a **server** that:

- Accepts HTTP requests (like any web API)
- Generates text quickly
- Handles multiple customers at the same time
- Uses the GPU efficiently

```
Before vLLM:  One customer at a time, slow, wastes GPU
After vLLM:   Many customers at once, fast, GPU fully utilized
```

**vLLM** is a production-ready inference engine built for exactly this.

---

## 2. The Two Export Paths from the Notebook

The fine-tune notebook gives you three artifacts. Two of them are for serving:

| Artifact | Created By | File Type | Best For |
|----------|-----------|-----------|----------|
| **GGUF (q4_k_m)** | `save_pretrained_gguf` | Single `.gguf` file | Ollama, laptops, edge devices |
| **Merged 16-bit** | `save_pretrained_merged` | Folder of `.safetensors` files | vLLM, production GPU servers |

### Why Two Formats?

Think of it like exporting a video file:

- **GGUF** is like a compressed `.mp4` — small, plays on any device, but slightly lower quality.
- **Merged 16-bit** is like a raw ProRes file — bigger, needs a powerful player, but the highest quality and fastest for professional editing.

For a restaurant chatbot serving hundreds of customers per hour, you want the **high-quality, high-speed format**: merged 16-bit served by vLLM.

---

## 3. What Does "Merged 16-bit" Mean?

### 3.1 Recall LoRA

During training, you did **not** change the base Qwen3-4B model. Instead, you trained two small matrices (A and B) called **LoRA adapters**.

```
Base model weights (frozen):     W        (4 billion numbers)
LoRA adapters:                   A × B    (12 million numbers)

Final behavior:  W_new = W + (A × B)
```

### 3.2 Merging

For vLLM, you need a single model file where the adapter update is **added back** into the original weights:

```
Step 1: Start with frozen base W
Step 2: Load trained LoRA adapters A and B
Step 3: Compute W_merged = W + (A × B)
Step 4: Save W_merged as full 16-bit weights
```

```python
model.save_pretrained_merged(
    "pizza_merged_16bit",
    tokenizer,
    save_method = "merged_16bit",
)
```

### 3.3 Why 16-bit?

| Format | Bits per Number | Size (Qwen3-4B) | Speed | Quality |
|--------|----------------|-----------------|-------|---------|
| 4-bit (QLoRA training) | 4 | ~2.5 GB | Training only | Slightly lower |
| 4-bit GGUF (q4_k_m) | 4 | ~2.5 GB | Good | Good |
| 8-bit GGUF (q8_0) | 8 | ~4.5 GB | Better | Very good |
| 16-bit merged | 16 | ~8 GB | Fastest | Best |

16-bit gives the **best quality and fastest GPU throughput** because the GPU doesn't have to spend time unpacking compressed numbers.

---

## 4. vLLM Under the Hood (Simplified)

### 4.1 The Problem vLLM Solves

A GPU is very fast, but generating text is memory-bound. Every word the model produces requires storing "key" and "value" tensors for every layer. This memory is called the **KV cache**.

If you serve one customer at a time, most of the GPU sits idle while waiting for the next token. vLLM fixes this with three techniques:

```
1. PagedAttention
   Splits the KV cache into fixed-size "pages" (like RAM pages in a computer).
   Avoids wasted memory and enables dynamic allocation.

2. Continuous Batching
   Doesn't wait for one request to finish before starting the next.
   New requests jump in while others are mid-generation.

3. Parallel Decoding
   Processes many tokens from many users simultaneously on the GPU.
```

### 4.2 Analogy: A Pizza Kitchen

| Without vLLM | With vLLM |
|--------------|-----------|
| One pizza at a time, oven mostly empty | Multiple pizzas in the oven, timing optimized |
| Wait until pizza 1 is fully baked before adding pizza 2 | Add new pizzas continuously; remove finished ones as they are done |
| Chef stands idle between orders | Chef always has work |

vLLM is the **master chef** that keeps the GPU (oven) full.

---

## 5. Step-by-Step: Deploy vLLM

### 5.1 Step 1: Get the Merged 16-bit Folder

From Colab:

```python
# After training completes
model.save_pretrained_merged(
    "pizza_merged_16bit",
    tokenizer,
    save_method="merged_16bit",
)

# Download (or upload to Google Drive / HuggingFace Hub)
from google.colab import files
!zip -r pizza_merged_16bit.zip pizza_merged_16bit
files.download("pizza_merged_16bit.zip")
```

You now have a folder containing:

```
pizza_merged_16bit/
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

### 5.2 Step 2: Move to a GPU Server

vLLM needs an NVIDIA GPU. Options:

| Platform | GPU | Cost | Good For |
|----------|-----|------|----------|
| Local RTX 3090/4090 | 24 GB | Free after hardware | Development |
| Google Cloud L4 | 24 GB | ~$0.80/hour | Small production |
| Google Cloud A100 | 40–80 GB | ~$3/hour | Large production |
| AWS g5.xlarge | 24 GB (A10G) | ~$1/hour | Production |
| RunPod / Vast.ai | Various | Variable | Experiments |

### 5.3 Step 3: Install vLLM

On your GPU server:

```bash
pip install vllm
```

If you get CUDA errors, install the version matching your CUDA:

```bash
pip install vllm==0.5.5  # or latest stable
```

### 5.4 Step 4: Start the Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/pizza_merged_16bit \
    --served-model-name pizza-palace \
    --max-model-len 1024 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9
```

#### What Each Flag Means

| Flag | Value | Meaning |
|------|-------|---------|
| `--model` | `/path/to/pizza_merged_16bit` | The folder with your merged model |
| `--served-model-name` | `pizza-palace` | The name clients use in API calls |
| `--max-model-len` | `1024` | Max tokens per request (system + user + response) |
| `--dtype` | `bfloat16` | Use 16-bit precision (fast, good quality) |
| `--gpu-memory-utilization` | `0.9` | Use 90% of GPU VRAM for KV cache |

### 5.5 Step 5: Test with curl

Open another terminal and send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pizza-palace",
    "messages": [
      {"role":"system","content":"You are a Pizza Palace agent. Menu: Margherita ₹210, Paneer Supreme ₹330, Cola ₹60. Use only this menu."},
      {"role":"user","content":"what veg pizzas do you have?"}
    ],
    "temperature": 0.4,
    "max_tokens": 128
  }'
```

Expected response:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Hey there! We have some great veg options — Margherita (₹210) and Paneer Supreme (₹330) are customer favourites. Would you like to add a Cola (₹60) to your order?"
    }
  }]
}
```

---

## 6. vLLM vs Ollama: When to Use Which

| Scenario | Use Ollama | Use vLLM |
|----------|-----------|----------|
| Local laptop, no internet budget | ✅ | ❌ |
| Single developer testing | ✅ | ⚠️ overkill |
| Docker container for small team | ✅ | ⚠️ overkill |
| 10+ concurrent customers | ❌ | ✅ |
| Cloud production API | ❌ | ✅ |
| Need OpenAI-compatible API | ✅ | ✅ |
| Need highest throughput | ⚠️ okay | ✅ |
| Want multi-GPU scaling | ❌ | ✅ |
| Need request batching / continuous serving | ❌ | ✅ |

### Simple Rule

- **Ollama** = your personal pizza oven at home.
- **vLLM** = the commercial pizza kitchen for a busy restaurant chain.

---

## 7. Connecting Your Frontend / Mobile App to vLLM

### 7.1 Python Client Example

```python
from openai import OpenAI

# vLLM speaks the same API as OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM doesn't check this by default
)

response = client.chat.completions.create(
    model="pizza-palace",
    messages=[
        {"role": "system", "content": "You are a warm Pizza Palace agent..."},
        {"role": "user", "content": "what's the cheapest pizza?"},
    ],
    temperature=0.4,
    max_tokens=128,
)

print(response.choices[0].message.content)
```

### 7.2 Streaming Responses

```python
stream = client.chat.completions.create(
    model="pizza-palace",
    messages=[
        {"role": "system", "content": "You are a warm Pizza Palace agent..."},
        {"role": "user", "content": "recommend me something"},
    ],
    temperature=0.4,
    max_tokens=128,
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

---

## 8. Spring AI Integration

### 8.1 Add Dependency

```xml
<!-- pom.xml -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>
```

### 8.2 Configure Application

```properties
# application.properties
spring.ai.openai.base-url=http://localhost:8000/v1
spring.ai.openai.api-key=not-needed
spring.ai.openai.chat.options.model=pizza-palace
spring.ai.openai.chat.options.temperature=0.4
spring.ai.openai.chat.options.max-tokens=128
```

> **Why OpenAI starter for vLLM?** vLLM exposes the exact same HTTP API format as OpenAI (`/v1/chat/completions`). Spring AI's OpenAI client works out of the box.

### 8.3 Service Class

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

@Service
public class PizzaPalaceVllmService {

    private final ChatClient chatClient;

    public PizzaPalaceVllmService(ChatClient.Builder builder) {
        this.chatClient = builder
            .defaultSystem("""
                You are a warm, upbeat customer support agent for Pizza Palace.
                Use ONLY the menu items in the conversation. Greet warmly,
                recommend with confidence, and suggest a relevant add-on.
                """)
            .build();
    }

    // Blocking response
    public String chat(String userMessage) {
        return chatClient.prompt()
            .user(userMessage)
            .call()
            .content();
    }

    // Streaming response (Server-Sent Events)
    public Flux<String> streamChat(String userMessage) {
        return chatClient.prompt()
            .user(userMessage)
            .stream()
            .content();
    }
}
```

### 8.4 REST Controller

```java
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/api/pizza")
public class PizzaPalaceController {

    private final PizzaPalaceVllmService service;

    public PizzaPalaceController(PizzaPalaceVllmService service) {
        this.service = service;
    }

    // POST /api/pizza/chat  body: "what veg pizzas do you have?"
    @PostMapping(value = "/chat", consumes = MediaType.TEXT_PLAIN_VALUE)
    public String chat(@RequestBody String message) {
        return service.chat(message);
    }

    // POST /api/pizza/stream  — returns tokens as they arrive
    @PostMapping(value = "/stream",
                 produces = MediaType.TEXT_EVENT_STREAM_VALUE,
                 consumes = MediaType.TEXT_PLAIN_VALUE)
    public Flux<String> stream(@RequestBody String message) {
        return service.streamChat(message);
    }
}
```

---

## 9. Production Checklist

```
Before launching to real customers:

□ GPU server is running and reachable
□ vLLM is started and healthy (test with curl)
□ System prompt includes correct, current menu
□ Spring Boot app points to correct base URL
□ API is behind authentication (add API key in production)
□ Logging and monitoring enabled
□ Rate limiting configured (prevent abuse)
□ Fallback to human support for sensitive complaints
□ Load balancer if running multiple vLLM replicas
```

---

## 10. Common Beginner Mistakes

| Mistake | Why It Fails | Fix |
|---------|-------------|-----|
| Passing the `.gguf` file to vLLM | vLLM doesn't read GGUF | Use merged 16-bit safetensors |
| Using `--model unsloth/Qwen3-4B` without merging | vLLM won't apply your LoRA adapters | Merge first with `save_pretrained_merged` |
| `--max-model-len` too small | Long system prompts get cut off | Set 1024 or higher |
| Forgetting the `api_key` in Spring AI | May cause connection issues | Set `api-key=not-needed` or a dummy value |
| Running vLLM without a GPU | It will fail or be extremely slow | Use a server with an NVIDIA GPU |
| Not passing the system prompt | Model defaults to generic tone | Always include menu in `messages[0]` |

---

## 11. One-Sentence Summary

> **vLLM takes your merged 16-bit fine-tuned model and turns it into a fast, OpenAI-compatible API that can serve hundreds of customers at once — while Spring AI lets your Java backend talk to it with just a few lines of code.**

---

*Deep dive compiled from the vLLM deployment section of the Qwen3 Pizza Palace fine-tuning notebook.*
