# Gen-AI Developer Classroom Notes — 28 Jul 2026

> **Source:** https://directai.blog/2026/07/28/gen-ai-developer-classroom-notes-28-jul-2026/
> **Audience:** Gen-AI Developers (Beginner → Intermediate)

---

## Table of Contents

1. [Models](#1-models)
2. [Accessing Models](#2-accessing-models)
3. [Models by Modality](#3-models-by-modality)
4. [Programmatic Access — Core Concepts](#4-programmatic-access--core-concepts)
5. [Exercise 1 — Accessing Models via Code](#5-exercise-1--accessing-models-via-code)
6. [Exercise 2 — Vision Model (Image Understanding)](#6-exercise-2--vision-model-image-understanding)
7. [Exercise 3 — Multi-Modal Pipeline (Music Composition)](#7-exercise-3--multi-modal-pipeline-music-composition)
8. [Understanding How LLMs Work](#8-understanding-how-llms-work)
9. [Key Takeaways & Cheat Sheet](#9-key-takeaways--cheat-sheet)

---

## 1. Models

There are **two fundamental categories** of AI models you will work with as a Gen-AI developer.

---

### 1.1 Open Source Models

These models have their weights publicly available. You can download, fine-tune, self-host, and inspect them freely.

| Model Family | By | Notable Models | Why Use It |
|--------------|----|----------------|------------|
| **OpenAI (OSS)** | OpenAI | GPT-2, Whisper, CLIP | Research-grade, widely documented |
| **Qwen** | Alibaba | Qwen2, Qwen2.5, QwQ | Strong multilingual + math reasoning |
| **GLM** | Tsinghua / Zhipu AI | GLM-4, ChatGLM | Chinese-English bilingual strength |
| **Google Gemma** | Google DeepMind | Gemma 2B, 7B, 27B | Lightweight, optimized for edge/local use |

> **When to choose open source:**
> - You need **data privacy** (no data leaves your infrastructure)
> - You want to **fine-tune** on your own domain data
> - You are **cost-sensitive** at scale (no per-token API cost)
> - You need **offline / air-gapped** deployment

---

### 1.2 Vendor (Proprietary) Models

These are hosted and managed by the company. You access them via API — you never see the weights.

| Vendor | Popular Models | Strengths |
|--------|---------------|-----------|
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus | Safety-focused, long context, nuanced reasoning |
| **OpenAI** | GPT-4o, o1, o3 | Broad capability, strong ecosystem, function calling |
| **Google** | Gemini 1.5 Pro, Gemini 2.0 Flash | Multimodal (text+image+audio+video), huge context window |
| **Amazon** | Nova, Titan | Deeply integrated with AWS services (Bedrock) |
| **Groq** | LLaMA 3 on GroqChip | Extremely **fast inference** (low latency) via custom hardware |

> **When to choose vendor models:**
> - You want **state-of-the-art performance** without infrastructure overhead
> - You need **rapid prototyping** — just call an API
> - Your use case is not privacy-sensitive
> - You need **multimodal** capabilities out of the box

---

### 1.3 Open Source vs Vendor — Decision Guide

```
Are you okay sending data to a third party?
    ├── No  → Open Source (self-host)
    └── Yes → Do you need cutting-edge performance?
                  ├── Yes → Vendor API (OpenAI, Claude, Gemini)
                  └── No  → Open Source hosted on Cloud (cheaper)
```

---

## 2. Accessing Models

### 2.1 Open Source — Hosting Options

#### Option A: Run on Your Own Hardware (Local)
- Download model weights (e.g., from Hugging Face)
- Run with tools like **Ollama**, **LM Studio**, or directly via `transformers`
- **Use case:** Developer laptop testing, offline environments, enterprise on-prem

```bash
# Example: Run Gemma locally with Ollama
ollama pull gemma2:2b
ollama run gemma2:2b
```

#### Option B: Cloud Hosted
Deploy the open source model on a cloud VM with a GPU:

| Cloud | Service | Example |
|-------|---------|---------|
| **GCP** | Vertex AI Model Garden | Deploy Gemma on Vertex AI endpoint |
| **AWS** | Amazon Bedrock / SageMaker | Host Llama 3 on SageMaker |
| **Azure** | Azure AI Studio / AML | Deploy Mistral via Azure ML |

> **Trade-off:** Cloud hosted = pay for GPU hours; Self-hosted = one-time hardware cost

---

### 2.2 Vendor Models — Access Options

#### Option A: Direct Vendor APIs
Call the model directly from the vendor's platform:

| Vendor | API Endpoint Base URL |
|--------|-----------------------|
| OpenAI | `https://api.openai.com/v1` |
| Anthropic | `https://api.anthropic.com/v1` |
| Google | `https://generativelanguage.googleapis.com` |
| Groq | `https://api.groq.com/openai/v1` |

#### Option B: Cloud-Hosted Vendor Models
Access vendor models through your existing cloud provider:

| Cloud | Service | What You Get |
|-------|---------|--------------|
| **AWS** | Amazon Bedrock | Claude, Llama, Titan, Mistral — unified API |
| **Azure** | Azure OpenAI Service | GPT-4o, o1 — enterprise SLA + compliance |
| **GCP** | Vertex AI | Gemini models + partner models |

> **Why use cloud-hosted vendor models?**  
> Enterprise compliance (SOC2, HIPAA), unified billing, VPC/private networking, no direct vendor account needed.

---

## 3. Models by Modality

Modality = **the type of data** the model can understand or generate.

| Modality | Input | Output | Example Models | Real-World Use Cases |
|----------|-------|--------|----------------|----------------------|
| **Text** | Text | Text | GPT-4o, Claude, Gemini | Chatbots, summarization, Q&A, translation |
| **Code** | Text/Code | Code | GitHub Copilot, CodeGemma, DeepSeek Coder | Auto-complete, bug fixing, code review, SQL generation |
| **Image** | Image/Text | Image/Text | Gemini Vision, GPT-4V, DALL-E 3, Stable Diffusion | OCR, image captioning, photo editing, product image generation |
| **Audio** | Audio/Text | Audio/Text | Whisper, ElevenLabs, Gemini Audio | Transcription, voice assistants, dubbing, podcast generation |
| **Video** | Video/Text | Video/Text | Gemini 1.5 Pro, Sora, Runway | Video summarization, subtitle generation, video creation |

> **Multimodal models** (like Gemini 1.5 Pro) handle **multiple modalities in a single model** — e.g., you can send an image + audio + text in one API call and get a text response.

---

## 4. Programmatic Access — Core Concepts

To call any AI model from code, you always need **three things**:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   PROMPT    │ +   │  API KEY     │ +   │    SDK      │
│ (your input)│     │(credentials) │     │ (library)   │
└─────────────┘     └──────────────┘     └─────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                    ┌───────▼────────┐
                    │   AI Model     │
                    │  (Cloud/API)   │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │   RESPONSE     │
                    │ (tokens/output)│
                    └────────────────┘
```

| Component | What It Is | Example |
|-----------|-----------|---------|
| **Prompt** | The text (or media) input you send to the model | `"What is the capital of France?"` |
| **API Key** | A secret token that authenticates your request | `AIza...XYZ` (never hardcode in code!) |
| **SDK** | A language library that wraps the HTTP API calls | `google-generativeai`, `openai`, `anthropic` |

> **Security Rule:** Always store API keys in environment variables or a `.env` file. **Never commit them to Git.**

#### 🐍 Python
```python
# Bad ❌
api_key = "AIzaSyXXXXXXXXXX"

# Good ✅
import os
api_key = os.environ.get("GEMINI_API_KEY")
```

#### ☕ Java (Spring AI)
```java
// Bad ❌
String apiKey = "AIzaSyXXXXXXXXXX";

// Good ✅ — declare in application.properties (never hardcode)
// spring.ai.vertex.ai.gemini.api-key=${GEMINI_API_KEY}
// Spring AI auto-reads the env var GEMINI_API_KEY at startup
```
```properties
# application.properties
spring.ai.vertex.ai.gemini.project-id=${GCP_PROJECT_ID}
spring.ai.vertex.ai.gemini.location=us-central1
spring.ai.vertex.ai.gemini.chat.options.model=gemini-1.5-flash
```

---

## 5. Exercise 1 — Accessing Models via Code

### Step 1: Get Your API Key (Google Gemini)

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **"Get API Key"** → **"Create API key"**
4. Copy and store securely (e.g., in `.env` file)

---

### Step 2: Add the SDK / Dependency

#### 🐍 Python
```bash
pip install google-generativeai
```

#### ☕ Java (Spring AI — Maven)
```xml
<!-- pom.xml -->
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework.ai</groupId>
            <artifactId>spring-ai-bom</artifactId>
            <version>1.0.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>

<dependencies>
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-vertex-ai-gemini-spring-boot-starter</artifactId>
    </dependency>
</dependencies>
```

#### ☕ Java (Spring AI — Gradle)
```groovy
// build.gradle
implementation platform('org.springframework.ai:spring-ai-bom:1.0.0')
implementation 'org.springframework.ai:spring-ai-vertex-ai-gemini-spring-boot-starter'
```

---

### Step 3: Access via Code

#### 🐍 Python
```python
import google.generativeai as genai
import os

# Load API key from environment variable
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Select a model
model = genai.GenerativeModel("gemini-1.5-flash")

# Send a prompt
response = model.generate_content("What is the capital of France?")

# Print the response
print(response.text)
```

#### ☕ Java (Spring AI)
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

@Service
public class GeminiService {

    private final ChatClient chatClient;

    // Spring AI auto-wires the configured Gemini model via application.properties
    public GeminiService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public String ask(String prompt) {
        // Send a prompt and get the response text
        return chatClient.prompt()
                .user(prompt)
                .call()
                .content();
    }
}
```
```java
// Calling the service (e.g., from a Spring Boot @RestController or main method)
@RestController
public class GeminiController {

    private final GeminiService geminiService;

    public GeminiController(GeminiService geminiService) {
        this.geminiService = geminiService;
    }

    @GetMapping("/ask")
    public String ask(@RequestParam String prompt) {
        return geminiService.ask(prompt);  // "What is the capital of France?"
    }
}
```

**Expected Output:**
```
The capital of France is Paris.
```

> **Spring AI Key Concept:** `ChatClient` is the unified abstraction. Swapping from Gemini to OpenAI or Claude only requires changing the starter dependency and `application.properties` — **no Java code changes needed**.

---

### Step 4: Access via curl (HTTP/REST)

```bash
curl -X POST \
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{
      "parts": [{"text": "What is the capital of France?"}]
    }]
  }'
```

> **Why curl?** Useful for testing APIs without writing code, debugging, or when building non-Python services (e.g., shell scripts, DevOps pipelines).

---

### How It All Works (Under the Hood)

#### 🐍 Python Flow
```
Your Python Code
    │
    ▼
google-generativeai SDK
    │  (wraps HTTP calls)
    ▼
HTTPS POST → Google API Gateway
    │
    ▼
Gemini Model (on Google's TPUs/GPUs)
    │  (processes your prompt, generates tokens)
    ▼
JSON Response → SDK parses → response.text
```

#### ☕ Java (Spring AI) Flow
```
Your Spring Boot Application
    │
    ▼
ChatClient (Spring AI abstraction layer)
    │
    ▼
VertexAiGeminiChatModel (Spring AI auto-configured bean)
    │  (wraps HTTP/gRPC calls)
    ▼
HTTPS POST → Google Vertex AI / Gemini API Gateway
    │
    ▼
Gemini Model (on Google's TPUs/GPUs)
    │  (processes your prompt, generates tokens)
    ▼
ChatResponse → .content() → String
```

> **Why Spring AI?** The `ChatClient` interface is the same whether you target Gemini, GPT-4o, or Claude. This makes it easy to **switch models** or **A/B test** providers without rewriting business logic.

---

## 6. Exercise 2 — Vision Model (Image Understanding)

### Use Case
> *"I have a selfie — convert it into a passport-size photograph for a job application."*

### Why a Regular Text Model Won't Work
A text-only LLM cannot **see** or process images. You need a **Vision Model** — a multimodal model that accepts image input.

### What is a Vision Model?
A Vision Model combines:
- **Vision Encoder** (e.g., ViT — Vision Transformer) → converts image pixels into embeddings
- **Language Model** → interprets those embeddings and reasons about them

```
[Your Selfie Image]
        │
        ▼
  Vision Encoder
  (image → embeddings)
        │
        ▼
  Language Model
  (reason + instruct)
        │
        ▼
  Image Generation Model
  (Imagen, DALL-E, Stable Diffusion)
        │
        ▼
  [Passport-size Output Photo]
```

### Implementation Approach

#### 🐍 Python
```python
import google.generativeai as genai
import PIL.Image
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Load your selfie
image = PIL.Image.open("selfie.jpg")

# Describe what you want
prompt = """
Analyze this selfie photo and describe what adjustments are needed
to make it passport-size compliant:
- Plain white background
- Face centered, taking 70-80% of frame
- Neutral expression
- Standard 35x45mm dimensions
"""

response = model.generate_content([prompt, image])
print(response.text)
```

#### ☕ Java (Spring AI)
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.content.Media;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;
import org.springframework.util.MimeTypeUtils;

@Service
public class VisionService {

    private final ChatClient chatClient;

    public VisionService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public String analyzePassportPhoto(String imageFileName) {
        // Load image from classpath (e.g., src/main/resources/selfie.jpg)
        Resource imageResource = new ClassPathResource(imageFileName);

        return chatClient.prompt()
                .user(u -> u
                    .text("""
                        Analyze this selfie photo and describe what adjustments are needed
                        to make it passport-size compliant:
                        - Plain white background
                        - Face centered, taking 70-80% of frame
                        - Neutral expression
                        - Standard 35x45mm dimensions
                        """)
                    .media(MimeTypeUtils.IMAGE_JPEG, imageResource)  // attach image
                )
                .call()
                .content();
    }
}
```
```java
// Calling the service
@RestController
public class VisionController {

    private final VisionService visionService;

    public VisionController(VisionService visionService) {
        this.visionService = visionService;
    }

    @PostMapping("/analyze-photo")
    public String analyzePhoto(@RequestParam String imagePath) {
        return visionService.analyzePassportPhoto(imagePath);
    }
}
```

> **Spring AI Note:** The `.media()` method on the user prompt builder handles multimodal (image + text) input. Spring AI automatically encodes the image as Base64 and sends it to the Gemini Vision API.

### Suitable Models for This Task

| Model | Capability | Notes |
|-------|-----------|-------|
| **Gemini 1.5 Pro** | Image understanding + instruction | Best for analysis + guidance |
| **GPT-4o** | Vision + generation guidance | Excellent for detailed image reasoning |
| **DALL-E 3** | Image generation from description | Generate the final passport photo |
| **Stable Diffusion (img2img)** | Image transformation | Transform selfie style/background |

> **Real-world pipeline:** Gemini/GPT-4V analyzes → gives instructions → Stable Diffusion / Imagen transforms the photo.

---

## 7. Exercise 3 — Multi-Modal Pipeline (Music Composition)

### Use Case
> *"I want to compose a special song for Spain for winning FIFA."*

### Why This is Complex
This is NOT a single-model task. It requires **chaining multiple AI models** — each handling a different modality.

### Full Pipeline Architecture

```
[User's Idea / Voice Input]
        │
        ▼ (1) Speech to Text
   Transcribed Text
   "A victorious anthem for Spain FIFA 2026 winners..."
        │
        ▼ (2) LLM (Text Generation)
   Song Lyrics + Music Description
   "Verse 1: Under the golden sun of Iberia..."
        │
        ├──────────────────────────────────┐
        ▼ (3a) Music Generation Model       ▼ (3b) Vocal Library / TTS
   Instrumental Track                   AI Vocals / Sung Lyrics
   (Metagen / Lyria 3)                  (ElevenLabs / LOVO)
        │                                   │
        └──────────────┬─────────────────────┘
                       ▼ (4) Audio Mixing
               Final Song (MP3/WAV)
```

### Tech Stack Breakdown

| Step | Component | Tool / Model | What It Does |
|------|-----------|-------------|--------------|
| 1 | **Speech to Text** | Whisper (OpenAI), Gemini Audio | Converts your spoken idea into text |
| 2 | **Lyric Generation** | GPT-4o, Claude, Gemini | Writes song lyrics from the concept |
| 3a | **Music Generation** | **Metagen**, **Lyria 3** (Google DeepMind) | Generates instrumental music |
| 3b | **Vocal Library / TTS** | ElevenLabs, LOVO, Suno | Synthesizes human-like vocals |
| 4 | **Audio Mixing** | Audacity API, custom FFmpeg pipeline | Merges vocals + instruments |

### About the Music Models

**Metagen**
- Generates music from text descriptions
- Supports style, tempo, instrumentation control
- Use case: Background scores, game music, custom jingles

**Lyria 3 (Google DeepMind)**
- State-of-the-art music generation model
- Supports lyrics-to-song generation
- High audio quality, multiple genre support
- Integrated into Google's AI ecosystem

### Sample Prompt for Music Generation

```
Generate an uplifting FIFA victory anthem for Spain.
Style: Orchestral + Flamenco fusion
Tempo: 120 BPM
Mood: Triumphant, emotional, celebratory
Duration: 3 minutes
Key: D Major
Include: Spanish guitar, choir, brass section
```

> **Key Insight:** Each model in the pipeline is specialized. The skill of a Gen-AI developer is knowing **which model to use** and **how to chain them** to solve the full problem.

---

## 8. Understanding How LLMs Work

### Token Generation — The Core Loop

An LLM does NOT "think" and then respond. It generates output **one token at a time**, in a loop:

```
Input Prompt (tokens)
        │
        ▼
┌───────────────────────────┐
│     Transformer Model      │
│  (attention + feedforward) │
└───────────┬───────────────┘
            │
            ▼
     Probability Distribution
     over entire vocabulary
            │
            ▼
     Pick Next Token
     (greedy / sampling)
            │
            ▼
   Append to sequence → loop back
            │
            ▼ (stop condition)
   ┌────────────────────────┐
   │ Token limit reached?   │──Yes──► STOP
   │ <EOS> token generated? │──Yes──► STOP
   └────────────────────────┘
            │ No
            └──► Continue generating
```

### What is a Token?

- A token ≈ **~4 characters** or **~0.75 words** in English
- `"Hello world"` = 2 tokens
- `"What is the capital of France?"` ≈ 7–8 tokens
- Models have a **context window** (max tokens: input + output combined)
  - GPT-4o: 128K tokens
  - Gemini 1.5 Pro: 1M tokens
  - Claude 3.5: 200K tokens

### Stop Conditions

| Condition | Description |
|-----------|-------------|
| **Token limit reached** | Model hits the `max_tokens` parameter you set |
| **`<EOS>` token generated** | Model itself decides the response is complete |
| **Stop sequences** | Custom strings you define (e.g., `"\n\n"`) that halt generation |

### The Big Question: If It's Just Predicting the Next Token — How Is It Intelligent?

This is one of the most important conceptual questions in Gen-AI.

**Answer — Emergent Intelligence:**

LLMs are trained on **trillions of tokens** from the internet — books, code, papers, conversations. To predict the next token accurately in complex text, the model must internally learn:

- **Facts** (capital of France = Paris)
- **Reasoning** (if A → B, and B → C, then A → C)
- **Code logic** (understanding syntax, semantics, algorithms)
- **Common sense** (objects fall when dropped)
- **Language structure** (grammar, idioms, tone)

These capabilities **emerge** from the sheer scale of next-token prediction training. The model isn't explicitly programmed to reason — it learns to do so because that's what produces correct next tokens.

```
Simple Rule:  "Predict the next token"
        +
Massive Scale: Trillions of tokens, billions of parameters
        =
Emergent Behaviors: Reasoning, coding, translation, creativity
```

> **Analogy:** A chess engine that has played millions of games doesn't "think" like a human, but produces moves that look strategic and intelligent. LLMs similarly produce outputs that appear intelligent because they've internalized patterns from vast human knowledge.

### Temperature — Controlling Creativity

| Temperature | Behavior | Use Case |
|-------------|----------|---------|
| `0.0` | Deterministic, always most likely token | Factual Q&A, structured output |
| `0.3–0.7` | Balanced — some creativity, mostly accurate | General chatbot, summarization |
| `1.0+` | High randomness, creative but less accurate | Creative writing, music lyrics, brainstorming |

#### 🐍 Python
```python
# Example: controlling temperature in Gemini
response = model.generate_content(
    "Write a song verse about Spain winning FIFA",
    generation_config=genai.types.GenerationConfig(
        temperature=0.9,      # creative
        max_output_tokens=200
    )
)
```

#### ☕ Java (Spring AI)
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.vertexai.gemini.VertexAiGeminiChatOptions;
import org.springframework.stereotype.Service;

@Service
public class CreativeService {

    private final ChatClient chatClient;

    public CreativeService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public String composeSongVerse() {
        return chatClient.prompt()
                .user("Write a song verse about Spain winning FIFA")
                .options(VertexAiGeminiChatOptions.builder()
                        .temperature(0.9)       // high creativity
                        .maxOutputTokens(200)   // limit response length
                        .build())
                .call()
                .content();
    }
}
```

> **Spring AI Note:** `ChatOptions` are **per-request overrides** — they take precedence over the defaults set in `application.properties`. This lets you use low temperature for factual endpoints and high temperature for creative ones in the same app.

---

## 9. Key Takeaways & Cheat Sheet

### Model Selection Quick Guide

```
What do you need?
├── Text understanding/generation  → GPT-4o / Claude / Gemini
├── Fast cheap inference           → Groq (Llama 3)
├── Privacy / no cloud             → Gemma / Qwen (self-hosted)
├── Image understanding            → Gemini Vision / GPT-4V
├── Image generation               → DALL-E 3 / Stable Diffusion / Imagen
├── Audio transcription            → Whisper
├── Audio generation               → ElevenLabs / LOVO
└── Music generation               → Metagen / Lyria 3
```

### Three Things Always Needed for API Access

| # | Component | Python Example | Java (Spring AI) Example |
|---|-----------|---------------|--------------------------|
| 1 | **Prompt** | `"Summarize this document"` | `"Summarize this document"` |
| 2 | **API Key** | `.env` / `os.environ.get(...)` | `application.properties` / env var |
| 3 | **SDK** | `pip install google-generativeai` | `spring-ai-vertex-ai-gemini-spring-boot-starter` |

### LLM Mental Model

```
LLM = Very sophisticated autocomplete
    + trained on all of human knowledge
    + at scale → emergent intelligence
```

### Modality Cheat Sheet

| If your input is... | And output should be... | Use |
|---------------------|------------------------|-----|
| Text | Text | Any LLM |
| Image + Text | Text | Vision Model (Gemini, GPT-4V) |
| Text | Image | DALL-E 3, Imagen, Stable Diffusion |
| Audio | Text | Whisper, Gemini Audio |
| Text | Audio | ElevenLabs, TTS APIs |
| Text | Music | Metagen, Lyria 3, Suno |

---

## References & Further Reading

- [Google AI Studio](https://aistudio.google.com/) — Get Gemini API key, test prompts
- [Hugging Face](https://huggingface.co/models) — Browse & download open source models
- [Ollama](https://ollama.ai/) — Run LLMs locally
- [OpenAI Docs](https://platform.openai.com/docs) — GPT API reference
- [Anthropic Docs](https://docs.anthropic.com) — Claude API reference
- [Google Gemini Docs](https://ai.google.dev/docs) — Gemini API reference
- [Amazon Bedrock](https://aws.amazon.com/bedrock/) — Multi-vendor models on AWS

---

*Notes compiled and expanded from Direct AI Blog — Classroom session 28 July 2026*
*Enhanced with deep insights, code examples, use cases, and architecture diagrams*
