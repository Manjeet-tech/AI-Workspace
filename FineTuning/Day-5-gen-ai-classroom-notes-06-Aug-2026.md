# DAY-5: Gen-AI Developer Classroom Notes — 06 Aug 2026

> **Source:** https://directai.blog/2026/08/06/gen-ai-developer-classroom-notes-06-aug-2026/
> **Audience:** Gen-AI Developers (Beginner → Intermediate)

---

## Table of Contents

1. [Encoders and Decoders in Transformers](#1-encoders-and-decoders-in-transformers)
2. [How Image Generation Works](#2-how-image-generation-works)
3. [Making a Model Behave Differently — Fine-Tuning](#3-making-a-model-behave-differently--fine-tuning)
4. [Key Takeaways & Cheat Sheet](#4-key-takeaways--cheat-sheet)

---

## 1. Encoders and Decoders in Transformers

### 1.1 The Big Picture

The original 2017 "Attention is All You Need" Transformer paper introduced a two-part architecture designed for **sequence-to-sequence** tasks (e.g., machine translation). Since then, the field has evolved into three distinct model families, each serving a different purpose.

```
┌─────────────────────────────────────────────────────────────────┐
│                   TRANSFORMER FAMILY TREE                       │
│                                                                 │
│  ┌─────────────┐   ┌─────────────────────┐   ┌─────────────┐  │
│  │   ENCODER   │   │  ENCODER + DECODER  │   │   DECODER   │  │
│  │  (Understand)│   │    (Transform)      │   │  (Generate) │  │
│  │             │   │                     │   │             │  │
│  │  BERT       │   │  T5, BART, mBART    │   │  GPT-4      │  │
│  │  RoBERTa    │   │  MarianMT           │   │  Claude     │  │
│  │  DistilBERT │   │  Helsinki-NLP       │   │  Llama 3    │  │
│  │  ELECTRA    │   │  Pegasus            │   │  Mistral    │  │
│  │             │   │                     │   │  Gemini     │  │
│  └─────────────┘   └─────────────────────┘   └─────────────┘  │
│       READ                TRANSFORM                 WRITE       │
└─────────────────────────────────────────────────────────────────┘
```

---

### 1.2 The Encoder — "Understanding the Input"

The **Encoder** reads the entire input sequence simultaneously (bidirectionally) and converts it into a rich numerical representation called **context embeddings** or a **hidden state**.

#### How it works internally

```
Input Text:  "The bank can guarantee deposits."
                │
                ▼
        [Tokenization]
                │
                ▼
   Token IDs: [101, 1996, 2924, 2064, 6302, 12816, 1012, 102]
                │
                ▼
   [Embedding Layer]  →  each token becomes a 768-dim vector
                │
                ▼
   [Positional Encoding]  →  adds position info to each vector
                │
                ▼
   [Multi-Head Self-Attention]  ← attends to ALL tokens simultaneously
                │                  "bank" sees "deposits" → finance context
                ▼
   [Feed-Forward Network]
                │
                ▼
   Context Embeddings: rich vector per token capturing meaning in context
```

#### Why bidirectional attention matters

The word **"bank"** can mean:
- A financial institution → "bank account"
- The side of a river → "river bank"

An encoder reads **both left and right context at the same time**, so it correctly understands "bank" in context. This is why BERT performs so well at classification and extraction tasks.

#### Use Cases for Encoder-Only Models

| Task | Example | Model Used |
|------|---------|------------|
| Sentiment Analysis | "This movie is great!" → Positive | BERT, RoBERTa |
| Named Entity Recognition (NER) | "Apple Inc. in Cupertino" → [ORG], [LOC] | BERT |
| Question Answering (Extractive) | Find the answer span in a document | BERT, DistilBERT |
| Semantic Search / Embeddings | Find similar documents | Sentence-BERT |
| Text Classification | Spam detection, toxic content | DistilBERT |
| Document Similarity | Deduplication of news articles | RoBERTa |

#### Code Example — Encoder for Sentiment Analysis

```python
from transformers import pipeline

# Load a pre-trained encoder-based sentiment classifier
classifier = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english")

texts = [
    "The product is absolutely amazing, I love it!",
    "Terrible experience. The app crashes constantly.",
    "It's okay, nothing special."
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"Text   : {text}")
    print(f"Label  : {result['label']}, Score: {result['score']:.4f}\n")

# Output:
# Text   : The product is absolutely amazing...
# Label  : POSITIVE, Score: 0.9998
#
# Text   : Terrible experience. The app crashes...
# Label  : NEGATIVE, Score: 0.9997
#
# Text   : It's okay, nothing special.
# Label  : NEGATIVE, Score: 0.6103
```

#### Spring AI Equivalent (Java) — Encoder-style Sentiment via Structured Output

> **Spring AI note:** Spring AI does not load BERT classifiers directly. Instead it uses an LLM (GPT, Claude, Ollama) as the inference engine and returns a structured result — achieving the same classification outcome with one API call.

**Maven / Gradle dependency:**
```xml
<!-- pom.xml -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>
```

**`application.properties`:**
```properties
spring.ai.openai.api-key=${OPENAI_API_KEY}
spring.ai.openai.chat.options.model=gpt-4o-mini
```

**Service class:**
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

@Service
public class SentimentService {

    // SentimentResult is a Java record — Spring AI maps the LLM JSON response here
    public record SentimentResult(String label, double score) {}

    private final ChatClient chatClient;

    public SentimentService(ChatClient.Builder builder) {
        // ChatClient.Builder is auto-configured by Spring Boot
        this.chatClient = builder
            .defaultSystem("""
                You are a sentiment classifier.
                Classify the user's text as POSITIVE or NEGATIVE.
                Always respond with valid JSON matching: {"label":"POSITIVE|NEGATIVE","score":0.0-1.0}
                """)
            .build();
    }

    public SentimentResult analyze(String text) {
        // .entity() calls the LLM and deserialises JSON → SentimentResult
        return chatClient.prompt()
            .user(text)
            .call()
            .entity(SentimentResult.class);   // structured output — no manual JSON parsing needed
    }

    // ── Batch version ────────────────────────────────────────────────
    public List<SentimentResult> analyzeBatch(List<String> texts) {
        return texts.stream()
            .map(this::analyze)
            .toList();
    }
}
```

**Controller:**
```java
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/sentiment")
public class SentimentController {

    private final SentimentService service;

    public SentimentController(SentimentService service) {
        this.service = service;
    }

    // GET /api/sentiment?text=I+love+this+product
    @GetMapping
    public SentimentService.SentimentResult analyze(@RequestParam String text) {
        return service.analyze(text);
    }

    // POST /api/sentiment/batch  body: ["text1","text2"]
    @PostMapping("/batch")
    public List<SentimentService.SentimentResult> analyzeBatch(@RequestBody List<String> texts) {
        return service.analyzeBatch(texts);
    }
}
```

**Using `EmbeddingModel` for Semantic Similarity (Encoder concept in Spring AI):**
```java
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class SemanticSimilarityService {

    private final EmbeddingModel embeddingModel;

    public SemanticSimilarityService(EmbeddingModel embeddingModel) {
        // Uses text-embedding-ada-002 or text-embedding-3-small by default
        this.embeddingModel = embeddingModel;
    }

    // Produces a 1536-dim float vector — equivalent to BERT's context embedding
    public float[] embed(String text) {
        return embeddingModel.embed(text);
    }

    // Cosine similarity between two texts (like Sentence-BERT)
    public double cosineSimilarity(String textA, String textB) {
        float[] a = embed(textA);
        float[] b = embed(textB);
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

> **Python vs Spring AI mapping:**
> | Python (HuggingFace) | Spring AI (Java) |
> |----------------------|-----------------|
> | `pipeline("sentiment-analysis")` | `ChatClient` + `.entity(SentimentResult.class)` |
> | `model.encode(text)` (Sentence-BERT) | `EmbeddingModel.embed(text)` |
> | `pipeline("feature-extraction")` | `EmbeddingModel.embedForResponse(List.of(text))` |

---

### 1.3 The Decoder — "Generating the Output"

The **Decoder** generates text **one token at a time**, autoregressively. At each step, it predicts the next token based on:
1. All previously generated tokens
2. Its internal knowledge from training (baked into weights)

#### The Autoregressive Loop

```
Prompt:  "The capital of France is"
              │
              ▼
         [Decoder Step 1]
         Sees: "The capital of France is"
         Predicts: "Paris"  (highest probability next token)
              │
              ▼
         [Decoder Step 2]
         Sees: "The capital of France is Paris"
         Predicts: "."
              │
              ▼
         [Decoder Step 3]
         Sees: "The capital of France is Paris."
         Predicts: <EOS>  → stop generation
              │
              ▼
Output:  "The capital of France is Paris."
```

#### Causal (Masked) Self-Attention

Unlike encoders, decoders use **causal masking** — when generating token N, the model can only attend to tokens 1 through N-1, never to future tokens. This prevents "cheating" during both training and inference.

```
Attention mask for "The cat sat on mat":

           The  cat  sat  on  mat
The      [  1    0    0    0    0 ]
cat      [  1    1    0    0    0 ]
sat      [  1    1    1    0    0 ]
on       [  1    1    1    1    0 ]
mat      [  1    1    1    1    1 ]

1 = "can attend to"    0 = "masked / blocked"
```

#### Why Most Modern LLMs Are Decoder-Only

The key insight: **for general-purpose language tasks, you only need generation.** Even "understanding" tasks can be rephrased as generation tasks:

| Task | Rephrased as Generation |
|------|------------------------|
| Sentiment Analysis | "Classify the sentiment: [text]. Answer:" → "Positive" |
| Summarization | "Summarize: [long text]. Summary:" → [generates summary] |
| Translation | "Translate to French: Hello. Answer:" → "Bonjour" |
| Code generation | "Write Python to sort a list:" → [generates code] |
| QA | "Answer: What is the capital of France? Answer:" → "Paris" |

Decoder-only models (GPT-4, Claude, Llama, Mistral, Gemini) have proven that a single architecture trained at scale can handle virtually any task through **prompting** and **instruction tuning** — no need for the extra complexity of a separate encoder.

#### Code Example — Decoder-Only Text Generation

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

prompt = "Explain the water cycle in 3 bullet points:"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

#### Spring AI Equivalent (Java) — Decoder-Only Text Generation

> **Spring AI note:** `ChatClient` wraps any decoder-only LLM (OpenAI GPT, Anthropic Claude, Ollama local models, Azure OpenAI, Mistral). The autoregressive generation loop happens inside the model — Spring AI gives you clean call/stream abstractions.

**`application.properties` (choose one provider):**
```properties
# ── Option A: OpenAI (GPT-4o, GPT-4o-mini) ────────────────────────
spring.ai.openai.api-key=${OPENAI_API_KEY}
spring.ai.openai.chat.options.model=gpt-4o-mini
spring.ai.openai.chat.options.temperature=0.7

# ── Option B: Ollama local model (Qwen3, Llama 3, Mistral) ─────────
# spring.ai.ollama.chat.options.model=qwen3:0.6b
# spring.ai.ollama.base-url=http://localhost:11434
```

**Service class — standard call + streaming:**
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

@Service
public class TextGenerationService {

    private final ChatClient chatClient;

    public TextGenerationService(ChatClient.Builder builder) {
        this.chatClient = builder
            .defaultOptions(ChatOptions.builder()
                .temperature(0.7)         // balanced creativity (same as Python top_p=0.9)
                .maxTokens(200)           // same as max_new_tokens=200
                .build())
            .build();
    }

    // ── Blocking call — returns full response at once ────────────────
    public String generate(String userPrompt) {
        return chatClient.prompt()
            .user(userPrompt)
            .call()
            .content();                    // decoded string — equivalent to tokenizer.decode(...)
    }

    // ── System + User roles — mirrors the chat template in Python ────
    public String generateWithSystem(String systemInstruction, String userPrompt) {
        return chatClient.prompt()
            .system(systemInstruction)     // e.g. "You are a helpful science teacher"
            .user(userPrompt)
            .call()
            .content();
    }

    // ── Streaming (token-by-token) — mirrors autoregressive decoding ─
    public Flux<String> generateStream(String userPrompt) {
        return chatClient.prompt()
            .user(userPrompt)
            .stream()                      // returns Flux<String> — each element is a new token/chunk
            .content();
    }
}
```

**Controller with SSE streaming endpoint:**
```java
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/api/generate")
public class TextGenerationController {

    private final TextGenerationService service;

    public TextGenerationController(TextGenerationService service) {
        this.service = service;
    }

    // POST /api/generate  body: "Explain the water cycle in 3 bullet points:"
    @PostMapping(consumes = MediaType.TEXT_PLAIN_VALUE)
    public String generate(@RequestBody String prompt) {
        return service.generate(prompt);
    }

    // POST /api/generate/stream  — Server-Sent Events (token streaming)
    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE,
                 consumes = MediaType.TEXT_PLAIN_VALUE)
    public Flux<String> stream(@RequestBody String prompt) {
        return service.generateStream(prompt);    // browser receives tokens as they arrive
    }
}
```

> **Python vs Spring AI mapping:**
> | Python (HuggingFace) | Spring AI (Java) |
> |----------------------|-----------------|
> | `model.generate(**inputs, max_new_tokens=200)` | `chatClient.prompt().user(...).call().content()` |
> | `temperature=0.7, top_p=0.9` | `ChatOptions.builder().temperature(0.7).build()` |
> | `tokenizer.apply_chat_template(messages)` | `.system("...")` + `.user("...")` on `ChatClient` |
> | `tokenizer.decode(outputs[...])` | `.call().content()` — returns decoded string directly |
> | Streaming with `TextIteratorStreamer` | `.stream().content()` → returns `Flux<String>` |

---

### 1.4 The Encoder-Decoder — "Transform Input to Output"

The full original Transformer uses both halves. The encoder reads the input, and the decoder generates the output while cross-attending to the encoder's representations.

```
Input Sentence (French):  "Bonjour le monde"
         │
         ▼
   ┌──────────────┐
   │   ENCODER    │  → produces context vector H (rich representation)
   └──────────────┘
         │
         │ (Cross-Attention — decoder "looks at" encoder output)
         ▼
   ┌──────────────┐
   │   DECODER    │  → generates "Hello" → "world" → <EOS>
   └──────────────┘
         │
         ▼
Output (English):  "Hello world"
```

#### Use Cases for Encoder-Decoder Models

| Task | Model | Example |
|------|-------|---------|
| Machine Translation | Helsinki-NLP/opus-mt-fr-en | French → English |
| Text Summarization | BART, Pegasus | Long article → short summary |
| Question Generation | T5 | "Generate a question from this context" |
| Grammar Correction | T5 | Fix grammatical errors |
| Data-to-Text | T5 | Structured table → natural language |
| Dialogue Systems | BART | Multi-turn conversations |

#### Code Example — Encoder-Decoder for Translation

```python
from transformers import MarianMTModel, MarianTokenizer

# Helsinki-NLP provides 1,000+ translation models
model_name = "Helsinki-NLP/opus-mt-fr-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

french_texts = [
    "Je m'appelle Marie et j'habite à Paris.",
    "Le machine learning est fascinant."
]

# Tokenize
inputs = tokenizer(french_texts, return_tensors="pt", padding=True)

# Translate (encode + decode internally)
translated = model.generate(**inputs)
english = tokenizer.batch_decode(translated, skip_special_tokens=True)

for fr, en in zip(french_texts, english):
    print(f"FR: {fr}")
    print(f"EN: {en}\n")

# Output:
# FR: Je m'appelle Marie et j'habite à Paris.
# EN: My name is Marie and I live in Paris.
#
# FR: Le machine learning est fascinant.
# EN: Machine learning is fascinating.
```

#### Spring AI Equivalent (Java) — Encoder-Decoder Translation

> **Spring AI note:** Spring AI routes translation through a hosted LLM (GPT-4o, Claude) which internally uses an encoder-decoder or decoder-only architecture to perform the same seq2seq transformation. You get identical results without managing model weights locally.

**Service class — single text and batch translation:**
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class TranslationService {

    private final ChatClient chatClient;

    public TranslationService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    /**
     * Translate a single text between any language pair.
     *
     * @param text     the source text
     * @param fromLang e.g. "French", "German", "Spanish"
     * @param toLang   e.g. "English", "Japanese"
     */
    public String translate(String text, String fromLang, String toLang) {
        return chatClient.prompt()
            .system("""
                You are a professional translator.
                Translate the user's %s text to %s.
                Return ONLY the translated text — no explanations, no quotes.
                """.formatted(fromLang, toLang))
            .user(text)
            .call()
            .content();
    }

    // ── Structured output — returns source + translation together ────
    public record TranslationPair(String source, String translation) {}

    public TranslationPair translateWithSource(String text, String fromLang, String toLang) {
        String translated = translate(text, fromLang, toLang);
        return new TranslationPair(text, translated);
    }

    // ── Batch translation (list of texts, same language pair) ────────
    public List<TranslationPair> translateBatch(List<String> texts, String fromLang, String toLang) {
        return texts.stream()
            .map(t -> translateWithSource(t, fromLang, toLang))
            .toList();
    }

    // ── Auto-detect source language ──────────────────────────────────
    public String translateAutoDetect(String text, String toLang) {
        return chatClient.prompt()
            .system("""
                You are a professional translator.
                Automatically detect the source language.
                Translate the text to %s.
                Return ONLY the translated text.
                """.formatted(toLang))
            .user(text)
            .call()
            .content();
    }
}
```

**Controller:**
```java
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/translate")
public class TranslationController {

    private final TranslationService service;

    public TranslationController(TranslationService service) {
        this.service = service;
    }

    // GET /api/translate?text=Bonjour&from=French&to=English
    @GetMapping
    public TranslationService.TranslationPair translate(
            @RequestParam String text,
            @RequestParam String from,
            @RequestParam String to) {
        return service.translateWithSource(text, from, to);
    }

    // POST /api/translate/batch?from=French&to=English
    // body: ["Je m'appelle Marie.", "Le machine learning est fascinant."]
    @PostMapping("/batch")
    public List<TranslationService.TranslationPair> translateBatch(
            @RequestBody List<String> texts,
            @RequestParam String from,
            @RequestParam String to) {
        return service.translateBatch(texts, from, to);
    }

    // GET /api/translate/auto?text=Bonjour&to=English
    @GetMapping("/auto")
    public String autoTranslate(@RequestParam String text, @RequestParam String to) {
        return service.translateAutoDetect(text, to);
    }
}
```

> **Python vs Spring AI mapping:**
> | Python (HuggingFace) | Spring AI (Java) |
> |----------------------|-----------------|
> | `MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")` | No local model; uses hosted LLM API |
> | `model.generate(**inputs)` (encoder-decoder inference) | `chatClient.prompt().system(...).user(...).call()` |
> | `tokenizer.batch_decode(translated)` | `.content()` — already decoded string |
> | Hardcoded language pair (one model per pair) | Single `ChatClient` handles ANY language pair |
> | Runs locally (CPU/GPU) | API call to cloud provider |

---

### 1.5 Summary Comparison Table

| Feature | Encoder Only | Encoder-Decoder | Decoder Only |
|---------|-------------|-----------------|--------------|
| Architecture | Bidirectional attention | Both | Causal/unidirectional |
| Primary Task | Understanding | Transformation | Generation |
| Generates Text? | No (classification output) | Yes | Yes |
| Examples | BERT, RoBERTa | T5, BART, MarianMT | GPT-4, Claude, Llama |
| Context Awareness | Full bidirectional | Full (encoder side) | Left-to-right only |
| Best For | NLP tasks needing understanding | Seq2Seq (translate, summarize) | General-purpose language tasks |
| Training Signal | Masked Language Modeling (MLM) | Seq2Seq loss | Causal Language Modeling (CLM) |

---

## 2. How Image Generation Works

### 2.1 From Pixels to Patches

Modern image AI models don't process images pixel-by-pixel — that would be computationally prohibitive for a 512×512 image (262,144 pixels). Instead, they use **patches**.

#### What is a Patch?

A **patch** is a fixed-size square tile of an image. Just as an NLP model breaks text into tokens, a Vision Transformer (ViT) breaks an image into patches.

```
Original Image (224 × 224 pixels)
┌──────────────────────────────────────┐
│ ██████ ██████ ██████ ██████ ██████  │
│ ██████ ██████ ██████ ██████ ██████  │  ← Each ██████ block = one 16×16 patch
│ ██████ ██████ ██████ ██████ ██████  │
│ ██████ ██████ ██████ ██████ ██████  │
│ ██████ ██████ ██████ ██████ ██████  │
│ ██████ ██████ ██████ ██████ ██████  │
│ ██████ ██████ ██████ ██████ ██████  │
│ ██████ ██████ ██████ ██████ ██████  │
└──────────────────────────────────────┘

With patch size = 16:
  Number of patches = (224/16) × (224/16) = 14 × 14 = 196 patches
  Each patch = 16 × 16 × 3 channels = 768 values
  → Flattened to a 768-dim vector → treated like a "token"
```

This is the core idea of **Vision Transformer (ViT)**: treat image patches as a sequence of tokens and apply a standard Transformer encoder to them.

---

### 2.2 The Diffusion Model — How Image Generation Actually Works

Modern image generators (Stable Diffusion, DALL-E 3, Midjourney, Imagen) use **Diffusion Models**. Here's the intuition:

#### Forward Process (Adding Noise — Training)

```
Real Image                           Pure Noise
    │                                    │
    │ ──── add Gaussian noise ────►      │
    │      step by step (T steps)        │
    │                                    │
[Cat photo]  →  [Slightly noisy]  →  [Very noisy]  →  [Pure noise]
    t=0               t=250              t=750              t=1000
```

#### Reverse Process (Removing Noise — Inference / Generation)

```
Pure Noise                         Generated Image
    │                                    │
    │ ◄─── predict & remove noise ────  │
    │       step by step                │
    │                                   │
[Pure noise]  →  [Mostly noise]  →  [Rough shapes]  →  [Clear image]
    t=1000             t=750              t=250              t=0
```

The **neural network** (a U-Net or Transformer) learns to **predict the noise** at each step. By repeating this in reverse, it reconstructs a clean image from noise.

---

### 2.3 Text-to-Image: Conditioning on a Prompt

The magic of systems like DALL-E 3 and Stable Diffusion is **conditioning** the diffusion process on a text prompt.

```
Text Prompt: "A red panda sitting on a mushroom, digital art"
         │
         ▼
   [Text Encoder (CLIP)]
   Converts text → embedding vector (captures semantic meaning)
         │
         ▼
         │◄──── text embedding injected at every denoising step
         │      (via Cross-Attention inside U-Net)
         ▼
   [Denoising U-Net / DiT]
   Iteratively removes noise, guided by text embedding
         │
         ▼
   Generated Image: red panda on mushroom, art style applied
```

#### Latent Diffusion (How Stable Diffusion Saves Computation)

Processing a 512×512 image with a Transformer at full resolution is extremely expensive. Stable Diffusion uses a **Variational Autoencoder (VAE)** to work in a compressed **latent space**:

```
Full Resolution Image (512×512×3 = 786,432 values)
         │
    [VAE Encoder]  ← compresses by 8×
         │
Latent Space (64×64×4 = 16,384 values)  ← 48× smaller!
         │
    [Diffusion happens HERE — much cheaper]
         │
    [VAE Decoder]  ← upsamples back
         │
Generated Image (512×512×3)
```

This is the "Computation" insight from the blog: diffusion models work in compressed latent space (patches/latent codes), not raw pixels — making generation feasible on consumer GPUs.

---

### 2.4 Image Generation Pipeline Summary

```
┌─────────────────────────────────────────────────────────┐
│              TEXT-TO-IMAGE PIPELINE                     │
│                                                         │
│  "A sunset over mountains"                              │
│           │                                             │
│    [CLIP Text Encoder]                                  │
│           │ text embedding                              │
│           ▼                                             │
│    [Gaussian Noise]                                     │
│           │                                             │
│    ┌──────────────┐                                     │
│    │   Denoising  │ ← cross-attention with text embed   │
│    │  U-Net / DiT │ (repeated ~50 steps / DDPM)         │
│    └──────────────┘                                     │
│           │ latent (64×64×4)                            │
│    [VAE Decoder]                                        │
│           │                                             │
│    Final Image (512×512×3)                              │
└─────────────────────────────────────────────────────────┘
```

---

### 2.5 Code Example — Image Generation with Diffusers

```python
# pip install diffusers transformers accelerate torch
from diffusers import StableDiffusionPipeline
import torch

# Load the pipeline (downloads model weights ~4GB)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16  # use fp16 for speed on GPU
)
pipe = pipe.to("cuda")  # move to GPU

prompt = "A red panda sitting on a giant mushroom in a forest, digital art, highly detailed"
negative_prompt = "blurry, low quality, distorted, ugly"

# Generate image
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,   # number of denoising steps
    guidance_scale=7.5,       # how closely to follow the prompt (CFG scale)
    height=512,
    width=512
).images[0]

image.save("red_panda.png")
print("Image saved!")
```

#### Spring AI Equivalent (Java) — Image Generation with `ImageModel`

> **Spring AI note:** Spring AI's `ImageModel` abstraction supports **DALL-E 3** (OpenAI), **Stability AI** (Stable Diffusion API), and **Azure OpenAI** image endpoints — giving you the same diffusion pipeline via a clean Java API.

**Maven dependency:**
```xml
<!-- pom.xml — OpenAI DALL-E 3 -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>

<!-- OR Stability AI (Stable Diffusion) -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-stability-ai-spring-boot-starter</artifactId>
</dependency>
```

**`application.properties`:**
```properties
# ── DALL-E 3 (OpenAI) ─────────────────────────────────────────────
spring.ai.openai.api-key=${OPENAI_API_KEY}
spring.ai.openai.image.options.model=dall-e-3
spring.ai.openai.image.options.quality=standard
spring.ai.openai.image.options.response-format=url

# ── Stability AI (Stable Diffusion) ───────────────────────────────
# spring.ai.stability.ai.api-key=${STABILITY_API_KEY}
# spring.ai.stability.ai.image.options.model=stable-diffusion-xl-1024-v1-0
```

**Service class:**
```java
import org.springframework.ai.image.*;
import org.springframework.ai.openai.OpenAiImageOptions;
import org.springframework.stereotype.Service;

@Service
public class ImageGenerationService {

    private final ImageModel imageModel;    // auto-configured by Spring Boot

    public ImageGenerationService(ImageModel imageModel) {
        this.imageModel = imageModel;
    }

    // ── Simple generation — returns URL of the generated image ───────
    public String generateImage(String prompt) {
        ImagePrompt imagePrompt = new ImagePrompt(
            prompt,
            OpenAiImageOptions.builder()
                .model("dall-e-3")
                .quality("standard")      // "standard" or "hd"
                .n(1)                     // number of images (DALL-E 3 supports only 1)
                .height(1024)             // equivalent to height=512 in Python (DALL-E 3 min is 1024)
                .width(1024)
                .responseFormat("url")    // "url" or "b64_json"
                .build()
        );

        ImageResponse response = imageModel.call(imagePrompt);
        return response.getResult().getOutput().getUrl();
    }

    // ── Full response with metadata ──────────────────────────────────
    public record ImageResult(String url, String revisedPrompt) {}

    public ImageResult generateWithMetadata(String prompt) {
        ImagePrompt imagePrompt = new ImagePrompt(
            prompt,
            OpenAiImageOptions.builder()
                .model("dall-e-3")
                .quality("hd")            // high definition — more detail
                .style("vivid")           // "vivid" (dramatic) or "natural"
                .n(1)
                .height(1024).width(1024)
                .responseFormat("url")
                .build()
        );

        ImageResponse response = imageModel.call(imagePrompt);
        Image output = response.getResult().getOutput();

        // DALL-E 3 rewrites your prompt for best results — capture the revised version
        return new ImageResult(output.getUrl(), output.getRevisedPrompt());
    }

    // ── Base64 response (when you want to save locally) ──────────────
    public byte[] generateAsBytes(String prompt) throws Exception {
        ImagePrompt imagePrompt = new ImagePrompt(
            prompt,
            OpenAiImageOptions.builder()
                .model("dall-e-3").n(1)
                .height(1024).width(1024)
                .responseFormat("b64_json")    // receive raw bytes instead of URL
                .build()
        );
        String b64 = imageModel.call(imagePrompt)
                               .getResult().getOutput().getB64Json();
        return java.util.Base64.getDecoder().decode(b64);
    }
}
```

**Controller:**
```java
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/image")
public class ImageController {

    private final ImageGenerationService service;

    public ImageController(ImageGenerationService service) {
        this.service = service;
    }

    // GET /api/image/generate?prompt=A+red+panda+on+a+mushroom
    @GetMapping("/generate")
    public ImageGenerationService.ImageResult generate(@RequestParam String prompt) {
        return service.generateWithMetadata(prompt);
        // Response: {"url":"https://...","revisedPrompt":"A vibrant red panda..."}
    }

    // GET /api/image/download?prompt=... — returns raw PNG bytes
    @GetMapping(value = "/download", produces = MediaType.IMAGE_PNG_VALUE)
    public ResponseEntity<byte[]> download(@RequestParam String prompt) throws Exception {
        byte[] imageBytes = service.generateAsBytes(prompt);
        return ResponseEntity.ok()
            .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=generated.png")
            .body(imageBytes);
    }
}
```

> **Python vs Spring AI mapping:**
> | Python (Diffusers) | Spring AI (Java) |
> |--------------------|-----------------|
> | `StableDiffusionPipeline.from_pretrained(...)` | `ImageModel` (auto-configured) |
> | `pipe(prompt=..., num_inference_steps=50)` | `imageModel.call(new ImagePrompt(prompt, options))` |
> | `guidance_scale=7.5` | Style/quality options in `OpenAiImageOptions` |
> | `negative_prompt=...` | Supported in Stability AI options; DALL-E 3 uses prompt rewriting |
> | `image.save("output.png")` | `Base64.getDecoder().decode(b64Json)` → write bytes to file |
> | Runs diffusion locally on GPU | Diffusion runs on OpenAI/Stability AI cloud GPU |

#### Key Parameters Explained

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| `num_inference_steps` | 20–50 | More steps = better quality, slower |
| `guidance_scale` (CFG) | 7–12 | Higher = follows prompt more strictly, less creative |
| `negative_prompt` | "blurry, ugly..." | What to avoid in the output |
| `height / width` | 512, 768, 1024 | Output resolution |
| `seed` | any integer | Reproducibility — same seed = same image |

---

### 2.6 Common Image Generation Architectures (Quick Reference)

| Architecture | Key Idea | Examples |
|-------------|---------|---------|
| **Diffusion (U-Net)** | U-Net denoises in latent space | Stable Diffusion 1.x/2.x |
| **DiT** (Diffusion Transformer) | Transformer instead of U-Net | Stable Diffusion 3, FLUX |
| **DALL-E 3** | CLIP + autoregressive / diffusion | OpenAI (closed source) |
| **Imagen** | T5 encoder + cascaded diffusion | Google (research) |
| **ViT-VQGAN** | Discrete tokens + transformer | DALL-E 1 style |

---

## 3. Making a Model Behave Differently — Fine-Tuning

### 3.1 The Core Concept

A pre-trained LLM (e.g., Llama 3, Mistral) has billions of weights learned from trillions of tokens of internet text. These weights encode general-purpose knowledge. But what if you need the model to:

- Only answer about **your company's products**?
- Respond in a **specific tone or format**?
- Understand **domain-specific jargon** (medical, legal, financial)?
- Perform a **specialized task** (code review, SQL generation)?

You need to **adjust the weights**. There are two main strategies:

```
┌──────────────────────────────────────────────────────────┐
│               FINE-TUNING SPECTRUM                       │
│                                                          │
│  ◄─────────────────────────────────────────────────►    │
│  Cheapest                                   Most Powerful│
│                                                          │
│  [Prompting] → [RAG] → [LoRA/QLoRA] → [Full Fine-Tune]  │
│     No weight    No       Few adapter     All weights    │
│     changes     weights   matrices        updated        │
│                 updated   updated                        │
└──────────────────────────────────────────────────────────┘
```

---

### 3.2 Option 1: Full Fine-Tuning — "Adjust ALL the Weights"

#### What It Is

Full fine-tuning updates every parameter of the pre-trained model during training on your custom dataset.

```
Pre-trained Model (e.g., Llama 3 8B)
 ├── Layer 1:  W1  (size: 4096 × 4096) ← UPDATED
 ├── Layer 2:  W2  (size: 4096 × 4096) ← UPDATED
 ├── ...
 └── Layer 32: W32 (size: 4096 × 4096) ← UPDATED

Total parameters updated: ~8,000,000,000 (8 Billion)
```

#### Advantages

- **Maximum performance** — model fully adapts to new domain/task
- **Can change fundamental behavior** — personality, output format, domain knowledge
- **Best for** — significant distribution shift (e.g., general model → medical specialist)

#### Disadvantages

- **Extremely expensive** — requires multiple high-end GPUs (A100/H100)
- **Catastrophic forgetting** — model may lose general capabilities
- **Requires large dataset** — typically 10,000+ high-quality examples
- **Slow** — training Llama 3 8B can take days/weeks

#### When to Use Full Fine-Tuning

- Your task is radically different from the pre-training data
- You have a massive, high-quality custom dataset (>100K examples)
- You have the compute budget (cloud TPUs/GPUs)
- Example: Training a specialized medical diagnosis model from a general LLM

#### Cost Estimate (Rough Guide)

| Model Size | Parameters | GPU Memory Needed | Rough Training Cost |
|-----------|-----------|------------------|-------------------|
| 1B | 1 Billion | ~8 GB | Hours on 1×A100 |
| 7B | 7 Billion | ~80 GB | Days on 4×A100 |
| 13B | 13 Billion | ~160 GB | Days on 8×A100 |
| 70B | 70 Billion | ~640 GB+ | Weeks on 32×A100 |

---

### 3.3 Option 2: LoRA — "Two Matrices" (Low-Rank Adaptation)

#### The Intuition — Why Two Matrices?

This is the key insight from the blog note **"Come up with two matrices."** — this refers to **LoRA (Low-Rank Adaptation)**, published by Hu et al. (2021) from Microsoft.

The big idea: **weight changes during fine-tuning tend to have low "rank"** — meaning they can be well-approximated by the product of two smaller matrices.

#### The Math (Simplified)

Normally, a weight matrix `W` in the model has shape `[d_out × d_in]`. For Llama 3 8B, this might be `[4096 × 4096]` = 16.7 million parameters per layer.

**LoRA says:** Instead of updating `W` directly, express the update `ΔW` as the product of two small matrices:

```
Full weight update (expensive):
  W_new = W + ΔW
  ΔW shape: [4096 × 4096]  = 16.7M params to learn

LoRA approximation (cheap):
  W_new = W + B × A
  A shape: [r × 4096]       ← small
  B shape: [4096 × r]       ← small
  where r = "rank" (e.g., 8, 16, 32 — much smaller than 4096!)

  Total new params = r × 4096 + 4096 × r = 2 × r × 4096
  With r=16: 2 × 16 × 4096 = 131,072 params ← 127× fewer!
```

#### Visual Diagram

```
During Inference:
                 Input x
                    │
      ┌─────────────┴─────────────┐
      │                           │
  [Frozen W]              [A matrix (r×d_in)]
  (original weights)             │
      │                   [B matrix (d_out×r)]
      │                           │
      │  W·x                 B·A·x  (scaled by α/r)
      │                           │
      └─────────────┬─────────────┘
                    │
                 Output = W·x + (α/r)·B·A·x
```

- The **original weights `W` are FROZEN** — never updated
- Only `A` and `B` are trained — a tiny fraction of parameters
- At inference, you just add `B×A` to `W` — zero added latency

#### LoRA Key Parameters

| Parameter | Typical Value | Meaning |
|-----------|--------------|---------|
| `r` (rank) | 4, 8, 16, 32 | Bottleneck dimension; higher = more capacity |
| `alpha` (α) | r or 2×r | Scaling factor for LoRA output |
| `target_modules` | q_proj, v_proj | Which weight matrices to apply LoRA to |
| Effective scale | α / r | Usually set to 1 (when α = r) |

---

### 3.4 QLoRA — LoRA on a Quantized Base Model

QLoRA combines two techniques:
1. **Quantize** the base model to 4-bit (NF4) — reduces memory by 4×
2. **Train LoRA adapters** on top of the quantized model — in full precision (bfloat16)

This enables fine-tuning a **65B parameter model on a single 48GB GPU** — something previously impossible.

```
┌──────────────────────────────────────────┐
│             QLoRA Stack                  │
│                                          │
│  [4-bit Quantized Base Model]  ← Frozen  │
│         ↕ (dequantize on the fly)        │
│  [bf16 LoRA adapters A and B]  ← Trained │
│                                          │
│  Memory: ~4× less than full fine-tuning  │
└──────────────────────────────────────────┘
```

---

### 3.5 Code Example — Fine-Tuning with LoRA using PEFT + TRL

```python
# pip install transformers peft trl accelerate bitsandbytes datasets

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ── Step 1: Load base model in 4-bit (QLoRA setup) ─────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# ── Step 2: Configure LoRA adapters ────────────────────────────────
lora_config = LoraConfig(
    r=16,                          # rank — bottleneck size
    lora_alpha=32,                 # scaling = alpha / r = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Wrap model with LoRA adapters (freezes base model weights)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 3,407,872 || all params: 1,238,684,672 || trainable%: 0.2752

# ── Step 3: Prepare a toy dataset ──────────────────────────────────
# Format: instruction-following pairs (replace with your real dataset)
SAMPLE_QA = [
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("Explain photosynthesis briefly.", "Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce glucose and oxygen."),
    ("What is 15 * 7?", "15 multiplied by 7 equals 105."),
]
data = {
    "text": [
        f"### Question: {q}\n### Answer: {a}" for q, a in SAMPLE_QA
    ]
}
dataset = Dataset.from_dict(data)

# ── Step 4: Configure training ──────────────────────────────────────
training_args = SFTConfig(
    output_dir="./lora-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# ── Step 5: Train ──────────────────────────────────────────────────
trainer.train()

# ── Step 6: Save only the LoRA adapter weights (tiny!) ─────────────
model.save_pretrained("./lora-adapter")
tokenizer.save_pretrained("./lora-adapter")
# Saved files: adapter_config.json + adapter_model.safetensors (~50MB)
# Compare: Full model weights ~2.5GB

# ── Step 7: Load and run inference ─────────────────────────────────
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base_model, "./lora-adapter")
model = model.merge_and_unload()  # merge LoRA weights into base — zero added latency
```

#### Spring AI Equivalent (Java) — Using a Fine-Tuned Model

> **Spring AI note:** Spring AI does **not** perform model training (LoRA/QLoRA). Training always happens offline in Python (PEFT/TRL). Once training is complete, you **deploy** the fine-tuned model and consume it through Spring AI as if it were any other model. Spring AI gives you three paths depending on where you host the fine-tuned model.

---

**Path A: OpenAI Fine-Tuned Model (hosted fine-tune)**

After uploading your dataset and running `openai api fine_tuning.jobs.create`, OpenAI gives you a model ID like `ft:gpt-4o-mini:your-org:task-name:abc123`.

```properties
# application.properties — just swap the model name
spring.ai.openai.api-key=${OPENAI_API_KEY}
spring.ai.openai.chat.options.model=ft:gpt-4o-mini:acme-corp:support-bot:abc123
```

```java
// Service code is IDENTICAL to standard ChatClient — zero changes needed
@Service
public class FineTunedSupportService {

    private final ChatClient chatClient;

    public FineTunedSupportService(ChatClient.Builder builder) {
        this.chatClient = builder
            .defaultSystem("You are Acme Corp's customer support assistant.")
            .build();
    }

    public String answer(String customerQuestion) {
        return chatClient.prompt()
            .user(customerQuestion)
            .call()
            .content();
        // Spring AI sends request to ft:gpt-4o-mini:acme-corp:support-bot:abc123
        // — the fine-tuned weights are already baked in on OpenAI's servers
    }
}
```

---

**Path B: Ollama Local Fine-Tuned Model (LoRA merged model served locally)**

After `model.merge_and_unload()` in Python, convert and serve via Ollama:

```bash
# 1. Convert merged model to GGUF format (llama.cpp)
python -m llama_cpp.convert ./merged-model --outfile my-model.gguf

# 2. Create a Modelfile for Ollama
echo 'FROM ./my-model.gguf
SYSTEM "You are a specialized assistant fine-tuned on custom data."' > Modelfile

# 3. Import and run
ollama create my-finetuned-model -f Modelfile
ollama run my-finetuned-model
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
spring.ai.ollama.chat.options.model=my-finetuned-model
```

```java
@Service
public class LocalFineTunedService {

    private final ChatClient chatClient;

    public LocalFineTunedService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
        // Ollama serves the LoRA-merged model locally — no GPU cloud cost
    }

    public String query(String prompt) {
        return chatClient.prompt()
            .user(prompt)
            .call()
            .content();
    }
}
```

---

**Path C: Custom REST Endpoint (any fine-tuned model hosted behind an API)**

If you self-host via vLLM, TGI (Text Generation Inference), or any OpenAI-compatible server:

```properties
# application.properties — point to your self-hosted vLLM server
spring.ai.openai.base-url=http://my-gpu-server:8000/v1
spring.ai.openai.api-key=not-needed         # or your internal token
spring.ai.openai.chat.options.model=my-lora-finetuned-llama3
```

```java
// Same ChatClient code — Spring AI talks OpenAI-compatible protocol
@Service
public class VllmFineTunedService {

    private final ChatClient chatClient;

    public VllmFineTunedService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public String generate(String prompt) {
        return chatClient.prompt()
            .user(prompt)
            .call()
            .content();
    }
}
```

---

**Dynamic model switching (A/B testing base vs fine-tuned):**
```java
@Service
public class ModelSwitchingService {

    private final ChatClient.Builder builder;

    public ModelSwitchingService(ChatClient.Builder builder) {
        this.builder = builder;
    }

    public String queryModel(String prompt, String modelId) {
        // Override the default model per-request
        return builder.build()
            .prompt()
            .options(ChatOptions.builder().model(modelId).build())
            .user(prompt)
            .call()
            .content();
    }

    // Compare base vs fine-tuned side by side
    public record ModelComparison(String baseResponse, String fineTunedResponse) {}

    public ModelComparison compare(String prompt) {
        String base       = queryModel(prompt, "gpt-4o-mini");
        String fineTuned  = queryModel(prompt, "ft:gpt-4o-mini:acme:task:id");
        return new ModelComparison(base, fineTuned);
    }
}
```

> **Python vs Spring AI mapping:**
> | Python (PEFT + TRL) | Spring AI (Java) |
> |---------------------|-----------------|
> | `LoraConfig(r=16, ...)` | N/A — training is Python-side only |
> | `SFTTrainer.train()` | N/A — training is Python-side only |
> | `model.save_pretrained("./lora-adapter")` | Training output → deploy to Ollama/vLLM/OpenAI |
> | `PeftModel.from_pretrained(base, adapter)` | `spring.ai.*.chat.options.model=<fine-tuned-id>` |
> | `model.merge_and_unload()` | Ollama imports merged GGUF file |
> | Local GPU inference | Ollama (local) or cloud API endpoint |

---

### 3.6 Comparing Fine-Tuning Strategies

| Strategy | Params Updated | GPU Memory | Training Speed | Use Case |
|----------|---------------|------------|----------------|----------|
| Full Fine-Tune | 100% (all) | Very High (8+ A100s) | Slowest | Domain shift, large budget |
| LoRA (r=16) | ~0.1–0.5% | Moderate (1 A100) | Fast | Task-specific adaptation |
| QLoRA (r=16, 4-bit) | ~0.1–0.5% | Low (1 consumer GPU) | Moderate | Limited compute, large models |
| Prompt Tuning | <0.01% | Minimal | Fastest | Slight style/format changes |
| RAG (no fine-tune) | 0% | None | N/A | Adding external knowledge |

---

### 3.7 Practical Decision Guide — Which Strategy to Choose?

```
START HERE
    │
    ▼
Do you have a custom knowledge base that changes frequently?
    │
    ├── YES → Use RAG (Retrieval-Augmented Generation)
    │          No fine-tuning needed. Add vector DB + retriever.
    │
    └── NO
         │
         ▼
    Is it a style/format/tone change only?
         │
         ├── YES → Use System Prompt or Prompt Engineering
         │
         └── NO
              │
              ▼
         Do you have <10K examples and limited GPU?
              │
              ├── YES → Use QLoRA (4-bit + LoRA adapters)
              │          Works on a single consumer GPU (RTX 3090/4090)
              │
              └── NO (you have large dataset + compute budget)
                   │
                   ▼
              Is the task radically different from pre-training?
                   │
                   ├── YES → Full Fine-Tuning (adjust ALL weights)
                   │
                   └── NO → LoRA Fine-Tuning (two matrices, efficient)
```

---

### 3.8 Real-World Use Cases

| Company / Scenario | Strategy | Details |
|-------------------|----------|---------|
| Customer Support Bot | QLoRA | Fine-tune on 5K support tickets; adapts tone + product knowledge |
| Legal Document Analyzer | Full Fine-Tune | Train on 500K legal documents; model learns legalese |
| Code Review Assistant | LoRA | Fine-tune on internal codebase conventions + style guides |
| Medical Q&A | QLoRA + RAG | LoRA for medical reasoning; RAG for latest drug information |
| Financial Report Generator | LoRA | Train on 10K earnings reports; learns structured output format |
| Multi-language Chatbot | Full Fine-Tune | Extend model to low-resource languages not in pre-training |

---

## 4. Key Takeaways & Cheat Sheet

### 4.1 Architecture Selection Quick Reference

```
┌─────────────────────────────────────────────────────────┐
│  What kind of task?         →  Which architecture?      │
│                                                         │
│  Classify / extract text    →  ENCODER (BERT, RoBERTa)  │
│  Translate / summarize      →  ENCODER-DECODER (T5)     │
│  Generate / chat / code     →  DECODER (GPT, Claude,    │
│                                         Llama, Mistral) │
│  Understand images          →  ViT (patch-based encoder)│
│  Generate images            →  Diffusion (SDXL, FLUX)   │
└─────────────────────────────────────────────────────────┘
```

### 4.2 The LoRA "Two Matrices" Explained in One Line

> Instead of learning a huge `ΔW` (same shape as the original weight), decompose it into `B × A` where `B` and `A` are much smaller — this captures the essence of the weight update with a fraction of the parameters.

### 4.3 Mental Models to Remember

| Concept | Mental Model |
|---------|-------------|
| Encoder | A reader who reads the whole book before answering |
| Decoder | A writer who writes one word at a time, never looking ahead |
| Encoder-Decoder | A translator who reads the source fully, then writes the translation |
| Diffusion | A photo developing from foggy noise into a clear image, guided by the caption |
| LoRA | Learning the "difference" cheaply by keeping two small sticky notes instead of rewriting the whole textbook |
| Patch (image) | Cutting an image into 196 puzzle pieces, treating each piece as a "word" |

### 4.4 Glossary

| Term | Definition |
|------|-----------|
| **Encoder** | Transformer block that reads input bidirectionally; produces context embeddings |
| **Decoder** | Transformer block that generates output autoregressively (token-by-token) |
| **Causal Masking** | Technique preventing decoder from attending to future tokens |
| **LoRA** | Low-Rank Adaptation — fine-tuning via two small matrices A and B |
| **QLoRA** | Quantized LoRA — LoRA on a 4-bit quantized base model |
| **Patch** | Fixed-size image tile used as a token in Vision Transformers |
| **Diffusion** | Generative process: learn to reverse noise → image transformation |
| **VAE** | Variational Autoencoder — compresses images into latent space |
| **CFG Scale** | Classifier-Free Guidance — controls how closely image follows prompt |
| **Full Fine-Tuning** | Updating all model weights on a custom dataset |
| **Rank (r)** | Bottleneck dimension in LoRA; controls adapter capacity |
| **Catastrophic Forgetting** | Loss of general capability after full fine-tuning on narrow data |

---

### 4.5 Three-Line Summary of Today's Lesson

1. **Transformers have three flavors** — Encoder (understand), Decoder (generate), Encoder-Decoder (transform). Most modern LLMs are decoder-only because generation subsumes everything.

2. **Image generation** works by learning to reverse a noise process, guided by a text embedding, in a compressed latent space made of image patches.

3. **To change model behavior**, either retrain all weights (expensive, powerful) or use LoRA's two-matrix trick to adapt efficiently with a fraction of the parameters.

---

> **Next Session Preview:** Likely covering RLHF (Reinforcement Learning from Human Feedback), DPO (Direct Preference Optimization), or advanced RAG patterns.

---

*Notes compiled with deep-dive explanations, examples, and use cases based on:*
*https://directai.blog/2026/08/06/gen-ai-developer-classroom-notes-06-aug-2026/*
