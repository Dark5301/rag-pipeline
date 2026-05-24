# RAG Pipeline — The Gift of the Magi

A fully functional **Retrieval-Augmented Generation (RAG)** pipeline built from scratch using PydanticAI, Qdrant, and OpenAI-compatible embeddings. This project demonstrates how to load, clean, chunk, embed, store, retrieve, and generate grounded answers from a text document using modern AI tooling.

The document used in this demo is *The Gift of the Magi* by O. Henry, sourced from Project Gutenberg.

---

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. Instead of relying solely on an LLM's training data, RAG pulls relevant context from your own documents at query time and passes it to the LLM — grounding the answer in your actual data rather than the model's memory.

```
Your Document → Clean → Chunk → Embed → Store in Vector DB
                                               ↓
User Query → Embed Query → Search Vector DB → Retrieve Top Chunks
                                               ↓
                             LLM gets: Query + Retrieved Chunks
                                               ↓
                                       Grounded Answer
```

---

## Project Structure

```
project/
├── RAG.py        # Ingestion pipeline — loads, cleans, chunks, embeds, stores
├── agent.py      # Query pipeline — embeds query, retrieves, generates answer
├── rag_text.txt  # Source document (The Gift of the Magi)
└── README.md
```

---

## Pipeline Overview

### RAG.py — Ingestion Pipeline (runs once at setup)

The `RAGPipeline` class handles the entire ingestion side:

**1. `load_documents()`**
Reads the raw text file into memory using Python's built-in file handling. Returns the entire file as a single string.

**2. `chunk_document(text)`**
Cleans and splits the document into chunks:
- Extracts only the story content using Project Gutenberg's start/end markers
- Strips all header metadata and legal boilerplate
- Splits the story into paragraphs on `\n\n` (double newlines)
- Groups paragraphs into chunks of ~500 characters
- Applies 100-character overlap between chunks so meaning doesn't get cut off at boundaries
- Handles edge cases: oversized paragraphs, empty chunks, duplicates, and orphan fragments
- Returns a clean `list[str]` of chunks

**3. `embedding(texts)`**
Sends all chunks to the embedding API in one batch call. Returns a `list[dict]` where each item contains:
```python
{
    "id"        : 0,
    "text"      : "original chunk text",
    "embedding" : [0.23, -0.81, 0.44, ...]  # 1536 floats
}
```

**4. `store_embeddings(embedded_chunks)`**
Stores all embedded chunks into Qdrant as `PointStruct` objects — each point contains the embedding vector and the original text as payload.

**5. `main()`**
Orchestrates the full ingestion flow: load → chunk → embed → store.

---

### agent.py — Query Pipeline (runs on every user query)

**`query_embedding(user_query)`**
Converts the user's question into a list of 1536 floats using the same embedding model used during ingestion. This is what gets compared against stored vectors in Qdrant.

**`chunk_retrieval(query_embedding)`**
Searches Qdrant for the top 3 most semantically similar chunks to the query embedding. Returns a `list[str]` of the most relevant chunk texts.

**`retrieve_chunks` (PydanticAI tool)**
Registered as a tool on the PydanticAI agent. When the agent decides it needs context, it automatically calls this tool — which embeds the query and retrieves relevant chunks from Qdrant — and uses the returned text to generate a grounded answer.

**`main(query)`**
Runs the agent with the user's question and prints the final answer.

---

## Tech Stack

| Component | Tool |
|---|---|
| Agent framework | PydanticAI |
| LLM | gpt-4o-mini via aicredits.in |
| Embedding model | text-embedding-3-small via aicredits.in |
| Vector database | Qdrant (in-memory) |
| HTTP client | AsyncOpenAI |
| Environment variables | python-dotenv |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Dark5301/rag-pipeline.git
cd rag-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install pydantic-ai openai qdrant-client python-dotenv
```

### 4. Set up environment variables

Create a `.env` file in the root of the project:

```
AICREDITS_API_KEY=your_api_key_here
```

### 5. Run the pipeline

```bash
python3 agent.py
```

---

## How It Works — Step by Step

```
Step 1  ✅  Load raw text file into memory
Step 2  ✅  Extract only the story using Gutenberg markers
Step 3  ✅  Split into paragraphs and group into 500-char chunks with overlap
Step 4  ✅  Embed all chunks via text-embedding-3-small (batch API call)
Step 5  ✅  Store embeddings + original text in Qdrant (in-memory)
Step 6  ✅  Embed user query using same embedding model
Step 7  ✅  Search Qdrant for top 3 most semantically similar chunks
Step 8  ✅  PydanticAI agent reads retrieved chunks and generates grounded answer
```

---

## Example Queries and Answers

```
Query: What did Della sell to buy Jim a gift?
Answer: Della sold her long, beautiful hair to Madame Sofronie for twenty dollars.

Query: What did Jim sell to buy Della a gift?
Answer: Jim sold his gold watch to buy Della a set of combs.

Query: Why are Jim and Della called the magi at the end?
Answer: Jim and Della are called the magi because they embody the true spirit
        of wise and selfless giving — just as the biblical Magi brought gifts
        to the manger, Jim and Della sacrificed their greatest treasures for
        each other, making them the wisest gift-givers of all.
```

---

## Key Design Decisions

**Why paragraph-based chunking?**
Splitting on natural paragraph boundaries (`\n\n`) preserves the semantic integrity of each chunk. Fixed-size character splitting risks cutting sentences mid-thought.

**Why 100-character overlap?**
Overlap ensures that meaning spanning two consecutive chunks is not lost. Without overlap, a sentence split across a boundary would be irretrievable.

**Why Qdrant in-memory?**
For a demo project, in-memory Qdrant requires zero infrastructure setup. Switching to a persistent Qdrant instance is a one-line change when moving to production.

**Why a PydanticAI tool instead of manual retrieval?**
Registering retrieval as a `@agent.tool_plain` lets the LLM decide when to retrieve — rather than always retrieving regardless of query type. This is closer to how production agentic RAG systems work.

**Why separate ingestion and query pipelines?**
`RAG.py` runs once at startup. `agent.py` runs on every query. Keeping them separate avoids re-embedding the entire document on every user request.

---

## Limitations

- Qdrant is in-memory — all data is lost when the process ends. For persistence, switch to a local or cloud Qdrant instance.
- The pipeline is hardcoded to a single document. Extending to multiple documents would require chunk metadata tagging and collection management.
- No conversation memory — each query is independent with no multi-turn context.

---

## Author

Built by Prince Singh as a learning project to understand RAG pipelines from the ground up — including file I/O, text chunking, vector embeddings, semantic search, and agentic generation with PydanticAI.
