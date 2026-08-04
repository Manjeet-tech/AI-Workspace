# DAY-2: Gen-AI Developer Classroom Notes — 29 Jul 2026

> **Source:** https://directai.blog/2026/07/29/gen-ai-developer-classroom-notes-29-jul-2026/
> **Audience:** Gen-AI Developers (Beginner → Intermediate)

---

## Table of Contents

1. [Evolution of AI — The Big Picture](#1-evolution-of-ai--the-big-picture)
2. [Machine Learning](#2-machine-learning)
3. [Neural Networks + Deep Learning](#3-neural-networks--deep-learning)
4. [Natural Language Processing (NLP)](#4-natural-language-processing-nlp)
5. [Word2Vec → Numerical Vectors → Semantics](#5-word2vec--numerical-vectors--semantics)
6. [Embeddings — Evolution of Word2Vec](#6-embeddings--evolution-of-word2vec)
7. [Transformers and Attention](#7-transformers-and-attention)
8. [Hardware: GPU & TPU](#8-hardware-gpu--tpu)
9. [Key Takeaways & Cheat Sheet](#9-key-takeaways--cheat-sheet)

---

## 1. Evolution of AI — The Big Picture

Every technology that powers Gen-AI today was built on top of the previous one. Understanding this evolution helps you make smarter architectural decisions.

```
1950s–1980s  →  Rule-Based AI / Expert Systems
                 (IF-THEN rules, hand-coded knowledge)
                         │
                         ▼
1990s–2000s  →  Machine Learning
                 (learn patterns from data, not rules)
                         │
                         ▼
2000s–2010s  →  Neural Networks + Deep Learning
                 (multi-layer networks, GPUs unlock scale)
                         │
                         ▼
2010s        →  NLP — Natural Language Processing
                 (machines start understanding human language)
                         │
                         ▼
2013         →  Word2Vec (words as numeric vectors)
                         │
                         ▼
2017         →  Transformers + Attention ("Attention Is All You Need")
                         │
                         ▼
2018–2022    →  BERT, GPT-2, GPT-3, PaLM
                 (pre-trained large language models)
                         │
                         ▼
2022–present →  Gen-AI (GPT-4, Gemini, Claude, Llama)
                 (generate text, code, images, audio, video)
```

> **Key Insight:** Gen-AI is not a single invention — it is the **convergence** of all these milestones, made possible by massive scale, transformer architecture, and specialized hardware (GPU/TPU).

---

## 2. Machine Learning

### What Is It?

> Traditional Programming: `Data + Rules → Output`
> Machine Learning: `Data + Output (labels) → Rules (Model)`

In ML, instead of writing rules manually, you **show the model many examples** and let it discover the rules (patterns) on its own.

### Types of Machine Learning

| Type | How It Learns | Example Use Cases |
|------|--------------|-------------------|
| **Supervised** | Labeled data (input + correct output) | Spam detection, image classification, price prediction |
| **Unsupervised** | Unlabeled data, finds structure | Customer segmentation, topic modeling, anomaly detection |
| **Reinforcement** | Rewards/penalties from environment | Game AI, robotics, recommendation systems |
| **Self-supervised** | Labels derived from the data itself | GPT pre-training (predict next word), BERT (mask and predict) |

> **Gen-AI Connection:** LLMs are trained using **self-supervised learning** — the training data is raw internet text, and the "label" is simply the next word in the sentence. No human labeling needed at scale.

### Classic ML Workflow

```
Raw Data
    │
    ▼
Feature Engineering   ← (humans manually extract meaningful features)
    │
    ▼
ML Algorithm          ← (Linear Regression, Decision Tree, SVM, etc.)
    │
    ▼
Trained Model
    │
    ▼
Predictions
```

#### 🐍 Python — Simple ML Example (Scikit-learn)
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Training data: hours studied → exam score
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([40, 50, 60, 70, 80])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict: student studied 6 hours
prediction = model.predict([[6]])
print(f"Predicted score: {prediction[0]:.1f}")  # ~90.0
```

#### ☕ Java (Spring AI / Tribuo — ML)
```java
// Using Tribuo (Oracle's ML library for Java)
// Maven: com.oracle.labs.mlrg:tribuo-regression-sgd:4.3.1

import org.tribuo.*;
import org.tribuo.regression.*;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.data.csv.CSVLoader;

// Load dataset
var regressionFactory = new RegressionFactory();
var csvLoader = new CSVLoader<>(regressionFactory);
var trainingData = csvLoader.loadDataSource("study_scores.csv", "score");

// Train a linear regression model
var trainer = new LinearSGDTrainer();
var model = trainer.train(trainingData.getData());

// Predict
var features = new ArrayExample<>(new RegressionFactory().getUnknownOutput());
features.add("hours_studied", 6.0);
var prediction = model.predict(features);
System.out.println("Predicted score: " + prediction.getOutput().getValues()[0]);
```

---

## 3. Neural Networks + Deep Learning

### What Is a Neural Network?

Inspired by the human brain — a neural network is made of **layers of connected nodes (neurons)**. Each neuron takes inputs, applies a weight and activation function, and passes output to the next layer.

```
Input Layer       Hidden Layers        Output Layer
┌────────┐        ┌────────────┐       ┌────────┐
│ pixel1 │──┐     │  neuron1   │──┐    │        │
│ pixel2 │──┼────▶│  neuron2   │──┼───▶│  cat?  │
│ pixel3 │──┤     │  neuron3   │──┤    │  dog?  │
│  ...   │──┘     │    ...     │──┘    │        │
└────────┘        └────────────┘       └────────┘
   (3072)           (512 nodes)          (10 classes)
```

### Why "Deep" Learning?

"Deep" = **many hidden layers**. The depth enables the model to learn progressively abstract features:
- Layer 1: detects edges
- Layer 2: detects shapes
- Layer 3: detects faces/objects
- Layer N: detects semantic concepts

### Deep Learning vs Classical ML

| | Classical ML | Deep Learning |
|--|-------------|---------------|
| **Feature Engineering** | Manual (domain expertise needed) | Automatic (learned from data) |
| **Data needed** | Small-medium | Large (millions of examples) |
| **Hardware** | CPU sufficient | GPU/TPU required |
| **Interpretability** | High (decision trees, linear) | Low ("black box") |
| **Performance on unstructured data** | Poor | Excellent |

### Key Deep Learning Architectures

| Architecture | Best For | Example |
|---|---|---|
| **CNN** (Convolutional Neural Network) | Images | Image classification, face recognition |
| **RNN** (Recurrent Neural Network) | Sequences | Time series, early NLP |
| **LSTM** (Long Short-Term Memory) | Long sequences | Speech recognition, translation |
| **Transformer** | Anything sequential | GPT, BERT, Gemini, Claude |

> **Gen-AI Connection:** Every modern LLM is a **Transformer-based deep learning model** with hundreds of layers (blocks) and billions of parameters.

#### 🐍 Python — Simple Neural Network (PyTorch)
```python
import torch
import torch.nn as nn

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),   # input: 28x28 image flattened
            nn.ReLU(),             # activation function
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)    # output: 10 digit classes (0–9)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleNN()
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
# Total parameters: 234,506
```

#### ☕ Java (Deep Learning4J)
```java
// Maven: org.deeplearning4j:deeplearning4j-core:1.0.0-M2.1

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
    .list()
    .layer(new DenseLayer.Builder()
        .nIn(784).nOut(256)
        .activation(Activation.RELU).build())
    .layer(new DenseLayer.Builder()
        .nIn(256).nOut(128)
        .activation(Activation.RELU).build())
    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nIn(128).nOut(10)
        .activation(Activation.SOFTMAX).build())
    .build();

MultiLayerNetwork model = new MultiLayerNetwork(config);
model.init();
System.out.println("Total parameters: " + model.numParams());
```

---

## 4. Natural Language Processing (NLP)

### What Is NLP?

NLP = teaching machines to **read, understand, and generate** human language.

### The Core NLP Challenge

Human language is:
- **Ambiguous:** "I saw the man with the telescope" (who has the telescope?)
- **Context-dependent:** "bank" = river bank or financial bank?
- **Evolving:** slang, new words, cultural references
- **Implicit:** sarcasm, idioms, tone

### NLP Evolution

```
1990s: Rule-based NLP
       └── Manually written grammars, regex, dictionaries

2000s: Statistical NLP
       └── n-grams, TF-IDF, Naive Bayes
       └── "Predict using frequency of words"

2013: Word2Vec
       └── Words as vectors, semantic relationships

2017: Transformers
       └── Context-aware representations

2018+: Pre-trained LLMs (BERT, GPT)
       └── One model for ALL NLP tasks
```

### Core NLP Tasks

| Task | Description | Example | Modern Model |
|------|-------------|---------|-------------|
| **Tokenization** | Split text into units | `"Hello world"` → `["Hello", "world"]` | SentencePiece, BPE |
| **Named Entity Recognition** | Find people, places, orgs | "Apple is in Cupertino" → [Apple=ORG, Cupertino=LOC] | BERT-NER |
| **Sentiment Analysis** | Positive / Negative / Neutral | "Great product!" → Positive | DistilBERT |
| **Machine Translation** | Translate languages | "Bonjour" → "Hello" | NLLB, GPT-4 |
| **Summarization** | Shorten long text | 10-page doc → 3 sentences | BART, GPT |
| **Question Answering** | Answer from context | "What is the capital?" → "Paris" | BERT, Gemini |

#### 🐍 Python — NLP with HuggingFace Transformers
```python
from transformers import pipeline

# Sentiment Analysis
sentiment = pipeline("sentiment-analysis")
result = sentiment("Spring AI makes Java developers love Gen-AI!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Named Entity Recognition
ner = pipeline("ner", grouped_entities=True)
result = ner("Sundar Pichai is the CEO of Google in Mountain View.")
for entity in result:
    print(f"{entity['word']} → {entity['entity_group']}")
# Sundar Pichai → PER
# Google        → ORG
# Mountain View → LOC
```

#### ☕ Java (Spring AI — NLP via LLM)
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

@Service
public class NlpService {

    private final ChatClient chatClient;

    public NlpService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public String analyzeSentiment(String text) {
        return chatClient.prompt()
                .user("Analyze the sentiment of this text and respond with POSITIVE, NEGATIVE, or NEUTRAL only: " + text)
                .call()
                .content();
    }

    public String extractEntities(String text) {
        return chatClient.prompt()
                .user("""
                    Extract named entities (Person, Organization, Location) from this text.
                    Return JSON format: {"persons": [], "organizations": [], "locations": []}
                    Text: """ + text)
                .call()
                .content();
    }
}
```

---

## 5. Word2Vec → Numerical Vectors → Semantics

### The Core Problem: Computers Can't Read Words

A model can only work with **numbers**. How do you convert words into numbers while preserving meaning?

### Naive Approach: One-Hot Encoding (Bad)

```
Vocabulary: [king, queen, man, woman, apple]

king  = [1, 0, 0, 0, 0]
queen = [0, 1, 0, 0, 0]
man   = [0, 0, 1, 0, 0]
woman = [0, 0, 0, 1, 0]
apple = [0, 0, 0, 0, 1]
```

**Problem:** Every word is equally distant from every other. `king` and `queen` are as "different" as `king` and `apple`. **No semantics.**

---

### Word2Vec — The Breakthrough (2013, Google)

Word2Vec learns vectors by training on the principle:
> **"You shall know a word by the company it keeps."** — J.R. Firth

Words that appear in similar contexts get similar vectors.

```
Training sentence: "The king ruled the kingdom wisely"
Training sentence: "The queen ruled the realm wisely"

Result:
king  ≈ [0.21, 0.85, -0.32, 0.67, ...]  ← 300-dim vector
queen ≈ [0.19, 0.87, -0.30, 0.72, ...]  ← very similar!
apple ≈ [-0.45, 0.12, 0.88, -0.21, ...]  ← very different
```

### The Famous Word Vector Arithmetic

```
king - man + woman = queen

vector("king")   = [0.21, 0.85, -0.32, 0.67]
vector("man")    = [0.11, 0.60,  0.10, 0.45]
vector("woman")  = [0.09, 0.62,  0.12, 0.50]

king - man + woman ≈ [0.19, 0.87, -0.30, 0.72] ≈ vector("queen") ✅
```

> **Why This Matters for Gen-AI:** This proved that **meaning can be captured mathematically**. It was the foundation for everything that followed — embeddings, attention, and LLMs.

### Limitations of Word2Vec

| Limitation | Example | Impact |
|------------|---------|--------|
| **No context** | "bank" always has one vector | Cannot distinguish river bank vs financial bank |
| **Fixed vocabulary** | Out-of-vocabulary words unknown | New words / slang break the model |
| **No morphology** | "run", "running", "ran" are separate | Treats related words as unrelated |

These limitations were solved by **contextual embeddings** (ELMo, BERT, GPT).

#### 🐍 Python — Word2Vec with Gensim
```python
from gensim.models import Word2Vec

# Training corpus (sentences as list of words)
sentences = [
    ["king", "rules", "the", "kingdom"],
    ["queen", "rules", "the", "realm"],
    ["man", "works", "in", "office"],
    ["woman", "works", "in", "office"],
    ["paris", "is", "capital", "of", "france"],
    ["berlin", "is", "capital", "of", "germany"],
]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, epochs=100)

# Find similar words
print(model.wv.most_similar("king", topn=3))
# [('queen', 0.87), ('realm', 0.72), ('rules', 0.65)]

# Vector arithmetic: king - man + woman = ?
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)
print(result)  # [('queen', 0.83)]

# Find cosine similarity
similarity = model.wv.similarity("paris", "berlin")
print(f"Paris ↔ Berlin similarity: {similarity:.3f}")  # ~0.92 (both are capitals)
```

#### ☕ Java (Spring AI — Embeddings)
```java
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class EmbeddingService {

    private final EmbeddingClient embeddingClient;

    public EmbeddingService(EmbeddingClient embeddingClient) {
        this.embeddingClient = embeddingClient;
    }

    public float[] getEmbedding(String text) {
        // Convert text to a numerical vector (embedding)
        EmbeddingResponse response = embeddingClient.embedForResponse(List.of(text));
        return response.getResults().get(0).getOutput();
    }

    public double cosineSimilarity(float[] a, float[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot   += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    public void compareSimilarity() {
        float[] king  = getEmbedding("king");
        float[] queen = getEmbedding("queen");
        float[] apple = getEmbedding("apple");

        System.out.println("king ↔ queen : " + cosineSimilarity(king, queen));   // ~0.92
        System.out.println("king ↔ apple : " + cosineSimilarity(king, apple));   // ~0.35
    }
}
```
```properties
# application.properties — use Gemini embedding model
spring.ai.vertex.ai.gemini.embedding.options.model=text-embedding-004
```

---

## 6. Embeddings — Evolution of Word2Vec

### What Are Embeddings?

An **embedding** is a dense numerical vector that represents data (text, image, audio) in a high-dimensional space, where **similar things are geometrically close**.

```
High-dimensional vector space (e.g., 1536 dimensions):

       [queen]
      /
[king]          [france]
      \        /
       [man] [paris]
              \
               [berlin] ← [germany]

Semantically related = spatially close
```

### Word2Vec vs Modern Embeddings

| Feature | Word2Vec | Modern Embeddings (BERT/OpenAI) |
|---------|----------|--------------------------------|
| **Context** | None — one vector per word | Context-aware — same word, different contexts = different vectors |
| **Subwords** | No | Yes — handles new/rare words |
| **Dimensionality** | 100–300 | 768–3072 |
| **Training** | Shallow network | Deep transformer |
| **Multilingual** | Separate per language | Single multilingual model |

### Why Embeddings Are Critical for Gen-AI Applications

#### Use Case 1: Semantic Search
```
User query: "How do I fix a memory leak?"

Traditional search (keyword): looks for exact words "memory" + "leak"
Embedding search: finds "garbage collection issue", "heap overflow" — same meaning, different words
```

#### Use Case 2: RAG (Retrieval-Augmented Generation)
```
1. Split your documents into chunks
2. Convert each chunk to an embedding vector
3. Store in a Vector Database (Pinecone, Weaviate, pgvector)
4. At query time: embed the user's question
5. Find top-K most similar document chunks
6. Send those chunks + question to LLM → grounded answer
```

#### Use Case 3: Recommendation Systems
```
"Users who liked [product A] also like [product B]"
→ Embed product descriptions
→ Find products with similar vectors
→ Recommend
```

#### 🐍 Python — Semantic Search with Embeddings
```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Knowledge base (your documents)
documents = [
    "Python is a high-level programming language.",
    "Java is an object-oriented language used for enterprise apps.",
    "Spring AI simplifies integrating AI into Java applications.",
    "Embeddings convert text into numerical vectors.",
    "Transformers use attention mechanisms for NLP tasks.",
]

# Encode all documents into vectors
doc_embeddings = model.encode(documents)

# User query
query = "How to use AI in Java?"
query_embedding = model.encode([query])

# Compute cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity(query_embedding, doc_embeddings)[0]

# Rank and retrieve top results
ranked = sorted(zip(scores, documents), reverse=True)
print("Top matches:")
for score, doc in ranked[:2]:
    print(f"  [{score:.3f}] {doc}")
# [0.812] Spring AI simplifies integrating AI into Java applications.
# [0.743] Java is an object-oriented language used for enterprise apps.
```

#### ☕ Java (Spring AI — Vector Store + Semantic Search)
```java
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Service
public class SemanticSearchService {

    private final VectorStore vectorStore;

    public SemanticSearchService(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    public void indexDocuments() {
        // Add documents to the vector store (auto-embeds them)
        vectorStore.add(List.of(
            new Document("Python is a high-level programming language."),
            new Document("Java is an object-oriented language for enterprise apps."),
            new Document("Spring AI simplifies integrating AI into Java applications."),
            new Document("Embeddings convert text into numerical vectors."),
            new Document("Transformers use attention mechanisms for NLP tasks.")
        ));
    }

    public List<Document> search(String query) {
        // Spring AI embeds the query and finds similar documents automatically
        return vectorStore.similaritySearch(
            SearchRequest.query(query).withTopK(2)
        );
    }
}
```
```properties
# application.properties — use an in-memory vector store for dev
spring.ai.vectorstore.simple.defaults.top-k=5
```

---

## 7. Transformers and Attention

### Why Transformers Were Revolutionary

Before Transformers (2017), RNNs/LSTMs processed text **sequentially** — word by word. This caused:
- **Vanishing gradient** — model "forgets" early parts of long sentences
- **Slow training** — can't parallelize across words

The Transformer paper **"Attention Is All You Need"** (Vaswani et al., 2017, Google) solved this.

### What Is Attention?

Attention = **how much focus should each word give to every other word** when computing its meaning?

```
Sentence: "The animal didn't cross the street because it was too tired."

What does "it" refer to?  →  "animal" (not "street")

Without attention:  model processes linearly, may lose context
With attention:     "it" strongly attends to "animal" → correct resolution
```

### Self-Attention (Visualized)

```
Input:  [The] [cat] [sat] [on] [the] [mat]

For the word "cat":
  Attention to "The"  → 0.10 (low — article)
  Attention to "cat"  → 0.50 (self — high)
  Attention to "sat"  → 0.30 (high — action of the cat)
  Attention to "on"   → 0.05
  Attention to "the"  → 0.03
  Attention to "mat"  → 0.02

Weighted sum of all word vectors → new context-aware "cat" vector
```

### Transformer Architecture

```
┌──────────────────────────────────────────────────────┐
│                   TRANSFORMER BLOCK                  │
│                                                      │
│  Input Tokens → [Embedding + Positional Encoding]    │
│                           │                          │
│              ┌────────────▼────────────┐             │
│              │    Multi-Head Attention  │             │
│              │  (parallel attention     │             │
│              │   across different       │             │
│              │   "representation        │             │
│              │    subspaces")           │             │
│              └────────────┬────────────┘             │
│                           │                          │
│              ┌────────────▼────────────┐             │
│              │   Feed-Forward Network  │             │
│              │   (position-wise MLP)   │             │
│              └────────────┬────────────┘             │
│                           │                          │
│                    Output Tokens                     │
└──────────────────────────────────────────────────────┘
     (This block is stacked N times — e.g., 96 times in GPT-4)
```

### Multi-Head Attention

Instead of one attention mechanism, Transformers run **multiple attention heads in parallel**, each learning different types of relationships:
- Head 1: syntax relationships (subject → verb)
- Head 2: coreference (pronoun → noun)
- Head 3: semantic similarity
- Head N: positional patterns

### Encoder vs Decoder Transformers

| Type | Architecture | Training Task | Examples | Best For |
|------|-------------|--------------|---------|---------|
| **Encoder-only** | Transformer encoder | Mask & predict (MLM) | BERT, RoBERTa | Classification, NER, embeddings |
| **Decoder-only** | Transformer decoder | Predict next token | GPT-4, Llama, Gemini | Text generation, chatbots |
| **Encoder-Decoder** | Both | Seq2Seq | T5, BART, mT5 | Translation, summarization |

> **Gen-AI Connection:** ChatGPT, Claude, Gemini — all are **decoder-only** transformers trained to predict the next token. This is why they are natural at text generation.

#### 🐍 Python — Transformer Embeddings (BERT)
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Contextual embeddings — same word, different context
sentences = [
    "I went to the river bank to fish.",       # bank = river bank
    "I went to the bank to deposit money.",    # bank = financial bank
]

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] token embedding — sentence-level representation
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    print(f"'{sentence[:30]}...' → vector shape: {cls_embedding.shape}")
    # Both are shape [1, 768] but have different values for "bank"
```

#### ☕ Java (Spring AI — Chat with Transformer-backed LLM)
```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class TransformerChatService {

    private final ChatClient chatClient;

    public TransformerChatService(ChatClient.Builder builder) {
        // The underlying model IS a Transformer (Gemini / GPT / Claude)
        this.chatClient = builder.build();
    }

    public String chat(String systemContext, String userMessage) {
        // System message sets the "persona" / context
        // User message is the actual query
        // Together they form the prompt — processed by the Transformer
        return chatClient.prompt()
                .system(systemContext)
                .user(userMessage)
                .call()
                .content();
    }

    // Example: context-aware disambiguation (like attention resolves "it")
    public String disambiguate() {
        return chat(
            "You are an NLP assistant. Resolve pronoun coreference.",
            "The trophy didn't fit in the suitcase because it was too big. What was too big?"
        );
        // Response: "The trophy was too big." (attention correctly resolves "it")
    }
}
```

---

## 8. Hardware: GPU & TPU

### Why Specialized Hardware?

Training and running neural networks requires **massive parallelism** — billions of multiply-add operations simultaneously. A standard CPU is not designed for this.

### CPU vs GPU vs TPU

| Feature | CPU | GPU | TPU |
|---------|-----|-----|-----|
| **Design goal** | General purpose, serial | Parallel graphics / compute | AI matrix operations |
| **Cores** | 8–128 powerful cores | 1,000–18,000 smaller cores | Tensor processing units |
| **Best at** | Control flow, branching | Parallel floating-point math | Matrix multiplications (neural nets) |
| **Memory bandwidth** | ~100 GB/s | ~2,000 GB/s | ~900 GB/s (HBM) |
| **Who makes it** | Intel, AMD | NVIDIA, AMD | Google (TPU), AWS (Trainium/Inferentia) |
| **Use in AI** | Inference (small models) | Training + inference | Large-scale training + serving |

### Why GPUs for Deep Learning?

Neural network training = **trillions of matrix multiplications**

```
Matrix multiply (weight update):
[n×m] × [m×k] → [n×k]

CPU: processes row by row (serial) → SLOW
GPU: processes ALL cells in parallel → 100x–1000x FASTER
```

### GPU Cloud Options

| Provider | Service | GPU Available |
|----------|---------|---------------|
| **Google Cloud** | Vertex AI / GKE | A100, H100, TPU v4/v5 |
| **AWS** | SageMaker / EC2 P-instances | A100, H100, AWS Trainium |
| **Azure** | Azure ML / NDv5 | H100, A100 |
| **Groq** | GroqCloud | GroqChip (LPU — Language Processing Unit) |
| **RunPod / Lambda Labs** | GPU cloud | A100, H100 (cost-effective) |

### As a Gen-AI Developer — When Do You Need a GPU?

```
Are you training a model from scratch?
    └── Yes → Need GPU (A100/H100) — Cloud: SageMaker, Vertex AI

Are you fine-tuning an open-source model?
    └── Yes → Need GPU (at least T4/A10) — Colab Pro, RunPod

Are you running inference on a small model locally?
    └── Yes → CPU may be enough — Ollama, LM Studio

Are you calling a vendor API (Gemini, GPT, Claude)?
    └── No GPU needed — it's all handled by the vendor
```

> **Gen-AI Developer Insight:** As an application developer calling LLM APIs (Gemini, OpenAI, Claude), you almost never need a GPU yourself. GPUs matter when you are training, fine-tuning, or self-hosting large models.

#### 🐍 Python — Check GPU Availability
```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} | Memory: {memory:.1f} GB")
else:
    device = torch.device("cpu")
    print("No GPU found — using CPU")

# Move model to GPU for faster inference
model = model.to(device)
```

#### ☕ Java (Spring AI — GPU-backed cloud model, transparent to developer)
```java
// As a Spring AI developer, GPU/TPU is ABSTRACTED AWAY.
// When you call ChatClient, the request goes to Gemini (running on TPUs)
// or OpenAI (running on NVIDIA H100s) — you never manage hardware directly.

@Service
public class HardwareTransparentService {

    private final ChatClient chatClient;

    public HardwareTransparentService(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    public String ask(String question) {
        // This request runs on Google TPU v5 behind the scenes
        // Zero GPU configuration needed by the Spring developer
        return chatClient.prompt()
                .user(question)
                .call()
                .content();
    }
}
```

---

## 9. Key Takeaways & Cheat Sheet

### AI Evolution at a Glance

```
Machine Learning → Deep Learning → NLP → Word2Vec → Embeddings → Transformers → LLMs → Gen-AI
```

Each step **built on and improved** the previous one.

---

### Concept Quick Reference

| Concept | One-Line Definition | Gen-AI Relevance |
|---------|-------------------|------------------|
| **Machine Learning** | Learn patterns from data instead of hand-coded rules | LLMs are trained with self-supervised ML |
| **Neural Network** | Layers of weighted connections that learn representations | LLMs are deep neural networks (100+ layers) |
| **NLP** | Making machines understand human language | Gen-AI is the pinnacle of NLP |
| **Word2Vec** | Words as dense vectors preserving semantic meaning | Foundation of modern embeddings |
| **Embeddings** | Context-aware numerical representations of data | Powers semantic search, RAG, recommendations |
| **Transformer** | Architecture using self-attention for parallel sequence processing | ALL modern LLMs are Transformers |
| **GPU/TPU** | Parallel compute hardware enabling large-scale training | Required for training; abstracted for API users |

---

### Python vs Java (Spring AI) — Quick Comparison

| Task | Python Library | Java (Spring AI) |
|------|---------------|-----------------|
| Text generation | `google-generativeai`, `openai` | `ChatClient` |
| Embeddings | `sentence-transformers`, `openai` | `EmbeddingClient` |
| Vector search | `faiss`, `chromadb` | `VectorStore` |
| NLP pipelines | `transformers` (HuggingFace) | `ChatClient` + structured output |
| Local models | `transformers`, `ollama` | `spring-ai-ollama-spring-boot-starter` |

---

### The Path to Understanding LLMs

```
You now understand:

✅ ML         → LLMs are trained on data, not programmed
✅ Deep Learning  → LLMs are very deep neural networks
✅ NLP         → LLMs solve all NLP tasks in one model
✅ Word2Vec    → Words can be numbers with semantic meaning
✅ Embeddings  → Context-aware, powers semantic search & RAG
✅ Transformers → The architecture that makes LLMs possible
✅ GPU/TPU     → The hardware that makes scale possible
```

---

## References & Further Reading

- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — The original Transformer paper
- [Word2Vec Paper (2013)](https://arxiv.org/abs/1301.3781) — Mikolov et al., Google
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Best visual explanation
- [HuggingFace](https://huggingface.co/) — Open source NLP models and datasets
- [Spring AI Docs](https://docs.spring.io/spring-ai/reference/) — Spring AI reference
- [Deep Learning Book](https://www.deeplearningbook.org/) — Goodfellow et al. (free online)
- [Fast.ai](https://www.fast.ai/) — Practical deep learning courses

---

*Notes compiled and expanded from Direct AI Blog — Classroom session 29 July 2026*
*Enhanced with deep insights, code examples (Python + Java Spring AI), architecture diagrams, and use cases*
