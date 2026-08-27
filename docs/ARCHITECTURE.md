# Code architecture

A module-by-module walkthrough of the system. For the overview and usage, see
the [README](../README.en.md).

The system is organized into independent modules that work together:

### 1. **types.py** — data models
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

### 2. **config.py** — configuration management
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

### 3. **embeddings.py** — embedding backend
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

### 4. **vector_store.py** — vector storage
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

### 5. **retrieval.py** — document retrieval
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

### 6. **memory.py** — memory storage
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

### 7. **pipeline.py** — pipeline orchestration
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

### 8. **agent.py + loop.py + tools.py + llm.py** — the agent

The heart of the system is a tool-use loop: the model receives the task and the
available tools, decides what to call, reads the results and iterates until it
produces a final answer (or hits `agent.max_iterations`). The answer is then
verified against the evidence the tools collected
(`verification.min_confidence`) — answers that drift from the retrieved
documents get an explicit warning.

Three backends implement the same interface (`config.llm.provider`) —
configuration details live in
[Swapping model or provider](../README.en.md#swapping-model-or-provider):

- `anthropic` — Claude with native tool use (requires `ANTHROPIC_API_KEY`)
- `openai` — any OpenAI-compatible endpoint (OpenRouter, Groq, vLLM, Ollama…)
- `scripted` — a deterministic offline model: searches once and answers
  extractively, while keeping the loop, tools, memory and verification
  genuinely exercised without network access (this is what CI runs)
- `null`/`auto` (default) — Claude, then OpenAI-compatible, then scripted

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

### 9. **main.py** — CLI interface
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

