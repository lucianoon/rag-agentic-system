# Agentic RAG System 🤖

*[Versão em português](README.md)*

A **Retrieval-Augmented Generation (RAG)** system where retrieval is a *tool*,
not a fixed pipeline stage. A Claude tool-use loop decides when to search,
refines the query and searches again when the first results fall short, and
iterates under a bounded step cap. The final answer is then scored for
groundedness against the evidence the tools actually returned.

> **Different from [Enterprise RAG System](https://github.com/lucianoon/enterprise-rag-system):**
> that repo retrieves once and optimizes *ranking quality* (hybrid BM25 + vector
> fusion, Recall@K / MRR). This one optimizes *multi-step reasoning* — questions
> a single query cannot answer.

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

## 📁 Project structure

```
rag-agentic-system/
├── main.py                     # CLI entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── config/
│   └── default.yaml            # Configuration file
├── src/rag_agent/              # Main application code
│   ├── __init__.py             # Package initialization
│   ├── agent.py                # Main RAG agent
│   ├── loop.py                 # Agentic loop: turns, tool use, verification
│   ├── tools.py                # Tool definitions and execution
│   ├── llm.py                  # Model backends (Claude / scripted)
│   ├── context.py              # Shared execution context
│   ├── config.py               # Configuration management
│   ├── embeddings.py           # Embedding backends
│   ├── memory.py               # Memory storage
│   ├── pipeline.py             # Pipeline orchestration
│   ├── retrieval.py            # Document retrieval
│   ├── types.py                # Data models
│   └── vector_store.py         # Vector storage
└── data/
    └── processed/              # Put documents here
```

## 🔧 Code walkthrough

### System architecture

The system is organized into independent modules that work together:

#### 1. **types.py** — data models
Defines the fundamental data structures:
- `Document`: a document with content and metadata
- `RetrievalResult`: the result of a vector search (document + score)
- `TaskLog`: the execution history of a task
- `AgentResponse`: the final response returned to the user

```python
# Example: creating a document
doc = Document(
    id="doc1",
    content="Document content",
    metadata={"source": "file.txt"}
)
```

#### 2. **config.py** — configuration management
Loads and manages system configuration from YAML:
- `EmbeddingConfig`: embedding settings
- `VectorStoreConfig`: vector storage settings
- `RetrievalConfig`: document retrieval parameters
- `MemoryConfig`: memory settings
- `AgentConfig`: agent parameters

```python
# Load configuration
config = load_config()  # Loads config/default.yaml
```

#### 3. **embeddings.py** — embedding backend
Turns text into numeric vectors:
- Supports Sentence-Transformers (better quality)
- Automatic fallback to TF-IDF (no GPU needed)
- Automatic vector normalization

```python
# Create the embedding backend
embeddings = EmbeddingBackend(config=config.embeddings)

# Turn text into a vector
vector = embeddings.embed_single("example text")
```

**How it works:**
1. Tries Sentence-Transformers (neural models)
2. Falls back to TF-IDF (statistics-based) if unavailable
3. Returns normalized vectors for similarity search

#### 4. **vector_store.py** — vector storage
Stores and searches documents by similarity:
- In-memory storage (a Python dictionary)
- Cosine similarity search
- Operations: add, search, delete, clear

```python
# Create the vector store
vector_store = VectorStore(config=config.vector_store)

# Add documents
vector_store.add([(document, vector)])

# Search for similar documents
results = vector_store.search(query_vector, top_k=5)
```

**How the search works:**
1. Receives a query vector
2. Computes cosine similarity against every stored vector
3. Returns the top_k most similar documents

#### 5. **retrieval.py** — document retrieval
Loads documents from disk and prepares them for indexing:

**DocumentIngestor**: loads files from disk
- Scans directories recursively
- Filters by extension (.txt, .md)
- Splits long texts into chunks

```python
# Create the ingestor
ingestor = DocumentIngestor(config=config.retrieval)

# Load documents as chunks
chunks = ingestor.load_chunks()
```

**FileSystemRetriever**: combines ingestion + embeddings + search
```python
# Create the retriever
retriever = FileSystemRetriever(
    config=config.retrieval,
    embeddings=embeddings,
    vector_store=vector_store
)

# Ingest documents
retriever.ingest()

# Search by query
results = retriever.search("my question")
```

**Text chunking:**
- Splits documents into smaller pieces (chunks)
- Uses `chunk_size` words per chunk
- Keeps `chunk_overlap` words between chunks to preserve context

#### 6. **memory.py** — memory storage
Saves task history in SQLite:
- Stores queries and answers
- Records reasoning steps
- Allows querying the history
- Automatic cleanup of old data

```python
# Create memory
memory = MemoryStore(config=config.memory)

# Save a task
log = TaskLog(task_id="task1", query="question")
memory.store(log)

# Query recent history
recent_tasks = memory.recent(limit=10)
```

#### 7. **pipeline.py** — pipeline orchestration
Coordinates the execution flow:

**ExecutionContext**: bundles every dependency
```python
context = ExecutionContext(
    config=config,
    embeddings=embeddings,
    retriever=retriever,
    vector_store=vector_store,
    memory=memory
)
```

**Pipeline**: retrieval and memory-logging utilities
1. Retrieves relevant documents
2. Records task logs in memory

```python
pipeline = Pipeline(context)
pipeline.initialize()  # Prepares resources

# Process a query
documents = pipeline.retrieve_documents("question")
response = pipeline.process("question", "answer", documents)
```

#### 8. **agent.py + loop.py + tools.py + llm.py** — the agent

The heart of the system is a tool-use loop: the model receives the task and the
available tools, decides what to call, reads the results and iterates until it
produces a final answer (or hits `agent.max_iterations`). The answer is then
verified against the evidence the tools collected
(`verification.min_confidence`) — answers that drift from the retrieved
documents get an explicit warning.

Two models implement the same interface (`config.llm.provider`):

- `anthropic` — Claude with native tool use (requires `ANTHROPIC_API_KEY`)
- `scripted` — a deterministic offline model: searches once and answers
  extractively, while keeping the loop, tools, memory and verification
  genuinely exercised without network access (this is what CI runs)
- `null`/`auto` (default) — Claude when the key exists, otherwise scripted

```python
# Create the agent
agent = AgenticRAG(context)
agent.initialize()

# Ask a question
response = agent.query("What is AI?")

# Access the response
print(response.answer)
print(response.references)      # Documents used
print(response.steps)           # Tool calls + answer + verification
print(response.metadata)        # agent_mode, iterations, tool_calls, confidence
```

**Execution flow:**
1. Receives the user's query
2. Sends it to the model along with the tool definitions
3. The model calls `search_documents` (possibly several times, refining the
   query between calls) and reads the results
4. Steps 2–3 repeat until the model answers or `agent.max_iterations` is reached
5. The answer is scored for groundedness against the accumulated evidence and
   flagged if it drifts
6. Answer and metadata are saved to memory
7. A structured response is returned

#### 9. **main.py** — CLI interface
The system's entry point:

**Interactive mode:**
- A question-and-answer loop
- Commands: stats, history, clear, quit
- Displays formatted answers

**Single-task mode:**
- Runs one question and exits
- Useful for scripting

```bash
# Interactive
python main.py

# Single task
python main.py --task "your question here"

# With a custom configuration
python main.py --config my_config.yaml
```

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
black src/ tests/

# Check code quality
flake8 src/ tests/
```

The same suite runs automatically in CI (GitHub Actions) on every push and pull
request to `main` — see `.github/workflows/ci.yml`.

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

## 🤝 Contributing

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 📚 Next steps

Planned improvements:
- FAISS/Qdrant backend for a persistent vector store
- New tools for the agent (full document read, note taking)
- LLM-judge verification on top of the lexical heuristic
- Packaging with pyproject.toml and a Dockerfile

## 🆘 Support

For problems or questions:
- Email: lucianomevam@outlook.com
- GitHub Issues: https://github.com/lucianoon/rag-agentic-system/issues
