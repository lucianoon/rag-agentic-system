# Agentic RAG System 🤖

*[Versão em português](README.md)*

[![CI](https://github.com/lucianoon/rag-agentic-system/actions/workflows/ci.yml/badge.svg)](https://github.com/lucianoon/rag-agentic-system/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **Retrieval-Augmented Generation (RAG)** system where retrieval is a *tool*,
not a fixed pipeline stage. A Claude tool-use loop decides when to search,
refines the query and searches again when the first results fall short, and
iterates under a bounded step cap. The final answer is then scored for
groundedness against the evidence the tools actually returned.

> **Not the same project as [Enterprise RAG System](https://github.com/lucianoon/enterprise-rag-system).**
> That repo retrieves **once** and optimizes *ranking quality* (hybrid BM25 +
> vector fusion, Recall@K / MRR). This one optimizes *multi-step reasoning* —
> questions a single query cannot answer.

## ✨ Features

- **Agentic loop with tool use**: the model chooses which tools to call
  (`search_documents`, `get_task_history`), reads the results and iterates until
  it can answer — bounded by `agent.max_iterations`
- **Two interchangeable brains**: Claude (native tool use, via
  `ANTHROPIC_API_KEY`) or a deterministic scripted model that keeps the same
  loop running offline (CI, demos without a key)
- **Groundedness verification**: the final answer is checked against the
  evidence the tools returned (`verification.min_confidence`); unconfirmed
  answers are flagged
- **Multi-source retrieval**: document ingestion from the filesystem
- **Flexible embeddings**: Sentence-Transformers with automatic TF-IDF fallback
- **Vector search**: in-memory cosine similarity
- **Task memory**: SQLite storage for task history
- **Interactive CLI**: a friendly command-line interface
- **Configurable pipeline**: YAML-based configuration

## 🚀 Installation

### Prerequisites
- Python 3.8 or newer
- Git

### Quick start

```bash
# Clone the repository
git clone https://github.com/lucianoon/rag-agentic-system.git
cd rag-agentic-system

# Install the dependencies
pip install -r requirements.txt

# Run the system
python main.py
```

## 📖 How to use

### Interactive mode (default)

```bash
python main.py
```

This starts the interactive RAG agent:

```
🤖 RAG Agentic System - Interactive Mode
Type 'help' for commands, 'quit' to exit

RAG> What is machine learning?
🔍 Processing: What is machine learning?

📄 Response:
Based on the retrieved documents...
```

### Single-task mode

```bash
python main.py --task "Explain quantum computing"
```

### Adding documents

1. Put your `.txt` or `.md` files in the `data/processed/` directory
2. Run the system — it indexes them automatically at startup

### Configuration

Edit `config/default.yaml` to customize:
- Embedding models
- Vector store settings
- Retrieval parameters
- Memory settings

### Available commands

In interactive mode:
- `<question>` — ask a question
- `stats` — show system statistics
- `history` — show recent task history
- `clear` — clear the vector store
- `quit`/`exit`/`q` — exit

## Swapping model or provider

The agent loop depends only on the `AgentModel` protocol, so three backends run
the very same loop machinery:

| `llm.provider` | Backend |
|---|---|
| `null` / `auto` (default) | Claude if a key exists; otherwise OpenAI-compatible if an endpoint exists; otherwise scripted |
| `anthropic` | Claude with native tool use |
| `openai` | Any OpenAI-compatible endpoint — OpenAI, OpenRouter, Groq, Together, vLLM, Ollama, LM Studio |
| `scripted` | Deterministic offline model (this is what CI runs) |

The OpenAI-compatible backend translates both ways: tool schemas and message
history on the way out, tool calls on the way back — the loop keeps speaking
Anthropic formats.

```bash
# OpenRouter, Groq, Together…
export RAG_LLM_BASE_URL=https://openrouter.ai/api/v1
export RAG_LLM_API_KEY=sk-or-v1-...
export RAG_LLM_MODEL=meta-llama/llama-3.3-70b-instruct

# Local Ollama — no credential at all
export RAG_LLM_BASE_URL=http://localhost:11434/v1
export RAG_LLM_MODEL=llama3.1
```

This backend requires `pip install openai`.

## Code architecture

A module-by-module walkthrough of the components (`types`, `config`,
`embeddings`, `vector_store`, `retriever`, `tools`, `agent`):
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## ⚙️ Detailed configuration

### Embeddings
```yaml
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: null  # null = auto, "cpu" or "cuda"
  use_tfidf_fallback: true  # Use TF-IDF if transformers is unavailable
  vector_dimension: 384  # Vector dimension
```

### Vector store
```yaml
vector_store:
  backend: "simple"  # Currently only "simple" is supported
  embedding_dimension: 384  # Must match the embeddings
  similarity_metric: "cosine"  # Similarity metric
  top_k: 5  # Number of documents to retrieve
```

### Retrieval
```yaml
retrieval:
  sources:
    - "data/processed"  # Directories to scan
  file_extensions:
    - ".txt"  # Allowed file extensions
    - ".md"
  chunk_size: 512  # Maximum words per chunk
  chunk_overlap: 64  # Overlapping words between chunks
```

### Memory
```yaml
memory:
  enabled: true  # Enable/disable memory
  database_path: "data/memory.db"  # SQLite database path
  cleanup_days: 30  # Delete logs older than X days
  importance_threshold: 0.3  # Threshold for storing tasks
```

## 🔄 Data flow

```
1. User asks a question
   ↓
2. The question and the tool definitions go to the model
   ↓
3. The model decides to call search_documents
   ↓
4. The query is embedded and the vector store returns the nearest chunks
   ↓
5. The model reads the results and either refines the query and searches
   again (back to step 3) or writes its answer
   ↓
6. The answer is scored for groundedness against everything the tools returned
   ↓
7. Answer + metadata are saved to memory
   ↓
8. The answer is shown to the user, flagged if groundedness is below threshold
```

Steps 3–5 are the loop: how many times they run is the model's decision,
bounded by `agent.max_iterations`.

## 🧪 Development

### Testing the system

The project has a test suite in `tests/` covering document chunking and
ingestion, cosine-similarity vector search, the TF-IDF embedding fallback,
SQLite task memory, YAML configuration loading and the full agent flow.

The tests need **no** LLM, API key or network access: the TF-IDF fallback path
is forced, so no sentence-transformers model is downloaded.

```bash
# Install the minimum dependencies for the tests
pip install numpy scikit-learn pyyaml pytest

# Run the full suite
pytest

# Run a specific module
pytest tests/test_vector_store.py -v

# Format the code
ruff format .

# Lint and type-check
ruff check .
mypy
```

The same three gates — `ruff check`, `mypy` and `pytest` — run automatically in
CI (GitHub Actions) on every push and pull request to `main`; see
`.github/workflows/ci.yml`.

### Adding new retrievers

```python
from src.rag_agent.retrieval import DocumentIngestor

class MyRetriever:
    def load_documents(self):
        # Your logic here
        pass
```

## 📊 Usage examples

### Example 1: simple question answering

```python
from src.rag_agent import AgenticRAG, load_config, create_context

# Set up
config = load_config()
context = create_context(config)
agent = AgenticRAG(context)
agent.initialize()

# Ask
response = agent.query("What is Python?")
print(response.answer)
```

### Example 2: adding documents manually

```python
# Add documents
agent.add_documents(["doc1.txt", "doc2.txt"])

# Search
response = agent.query("Search the added documents")
```

### Example 3: viewing statistics

```python
stats = agent.get_stats()
print(f"Documents: {stats['total_documents']}")
print(f"Embeddings: {stats['embeddings_stored']}")
```

## 📚 Next steps

Planned improvements:
- FAISS/Qdrant backend for a persistent vector store
- New tools for the agent (full document read, note taking)
- LLM-judge verification on top of the lexical heuristic
- Packaging with `pyproject.toml` (Dockerfile and compose already exist)

## 📄 License

This project is licensed under the MIT License.
