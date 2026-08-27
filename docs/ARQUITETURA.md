# Arquitetura do código

Passeio módulo a módulo pelo sistema. Para o panorama e o uso, veja o
[README](../README.md).

O sistema é organizado em módulos independentes que trabalham juntos:

### 1. **types.py** - Modelos de Dados
Define as estruturas de dados fundamentais:
- `Document`: Representa um documento com conteúdo e metadados
- `RetrievalResult`: Resultado de uma busca vetorial (documento + score)
- `TaskLog`: Registra o histórico de execução de uma tarefa
- `AgentResponse`: Resposta final retornada ao usuário

```python
# Exemplo: Criando um documento
doc = Document(
    id="doc1",
    content="Conteúdo do documento",
    metadata={"source": "arquivo.txt"}
)
```

### 2. **config.py** - Gerenciamento de Configuração
Carrega e gerencia configurações do sistema via YAML:
- `EmbeddingConfig`: Configurações de embeddings
- `VectorStoreConfig`: Configurações do armazenamento vetorial
- `RetrievalConfig`: Parâmetros de recuperação de documentos
- `MemoryConfig`: Configurações de memória
- `AgentConfig`: Parâmetros do agente

```python
# Carregar configuração
config = load_config()  # Carrega config/default.yaml
```

### 3. **embeddings.py** - Backend de Embeddings
Converte texto em vetores numéricos:
- Suporta Sentence-Transformers (melhor qualidade)
- Fallback automático para TF-IDF (não precisa GPU)
- Normalização automática de vetores

```python
# Criar backend de embeddings
embeddings = EmbeddingBackend(config=config.embeddings)

# Converter texto em vetor
vector = embeddings.embed_single("texto de exemplo")
```

**Como funciona:**
1. Tenta usar Sentence-Transformers (modelos neurais)
2. Se não disponível, usa TF-IDF (baseado em estatísticas)
3. Retorna vetores normalizados para busca de similaridade

### 4. **vector_store.py** - Armazenamento Vetorial
Armazena e busca documentos por similaridade:
- Armazenamento em memória (dicionário Python)
- Busca por similaridade de cosseno
- Operações: add, search, delete, clear

```python
# Criar vector store
vector_store = VectorStore(config=config.vector_store)

# Adicionar documentos
vector_store.add([(documento, vetor)])

# Buscar documentos similares
results = vector_store.search(query_vector, top_k=5)
```

**Como funciona a busca:**
1. Recebe um vetor de consulta
2. Calcula similaridade de cosseno com todos os vetores armazenados
3. Retorna os top_k documentos mais similares

### 5. **retrieval.py** - Recuperação de Documentos
Carrega documentos do disco e os prepara para indexação:

**DocumentIngestor**: Carrega arquivos do disco
- Varre diretórios recursivamente
- Filtra por extensões (.txt, .md)
- Divide textos longos em chunks

```python
# Criar ingestor
ingestor = DocumentIngestor(config=config.retrieval)

# Carregar documentos em chunks
chunks = ingestor.load_chunks()
```

**FileSystemRetriever**: Combina ingestão + embeddings + busca
```python
# Criar retriever
retriever = FileSystemRetriever(
    config=config.retrieval,
    embeddings=embeddings,
    vector_store=vector_store
)

# Ingerir documentos
retriever.ingest()

# Buscar por query
results = retriever.search("minha pergunta")
```

**Chunking de Texto:**
- Divide documentos em pedaços menores (chunks)
- Usa `chunk_size` palavras por chunk
- Mantém `chunk_overlap` palavras entre chunks para preservar contexto

### 6. **memory.py** - Armazenamento de Memória
Salva histórico de tarefas em SQLite:
- Armazena queries e respostas
- Registra passos de raciocínio
- Permite consultar histórico
- Limpeza automática de dados antigos

```python
# Criar memória
memory = MemoryStore(config=config.memory)

# Salvar uma tarefa
log = TaskLog(task_id="task1", query="pergunta")
memory.store(log)

# Consultar histórico recente
recent_tasks = memory.recent(limit=10)
```

### 7. **pipeline.py** - Orquestração do Pipeline
Coordena o fluxo de execução:

**ExecutionContext**: Agrupa todas as dependências
```python
context = ExecutionContext(
    config=config,
    embeddings=embeddings,
    retriever=retriever,
    vector_store=vector_store,
    memory=memory
)
```

**Pipeline**: Utilitários de recuperação e registro em memória
1. Recupera documentos relevantes
2. Registra logs de tarefas na memória

```python
pipeline = Pipeline(context)
pipeline.initialize()  # Prepara recursos

# Processar query
documents = pipeline.retrieve_documents("pergunta")
response = pipeline.process("pergunta", "resposta", documents)
```

### 8. **agent.py + loop.py + tools.py + llm.py** - O Agente

O coração do sistema é um loop de tool use: o modelo recebe a tarefa e as
ferramentas disponíveis, decide o que chamar, lê os resultados e itera até
produzir a resposta final (ou atingir `agent.max_iterations`). Depois, a
resposta é verificada contra as evidências coletadas pelas ferramentas
(`verification.min_confidence`) — respostas que se afastam dos documentos
recuperados recebem um aviso explícito.

Três backends implementam a mesma interface (`config.llm.provider`) — os
detalhes de configuração estão em
[Trocando de modelo ou de provedor](../README.md#trocando-de-modelo-ou-de-provedor):

- `anthropic` — Claude com tool use nativo (requer `ANTHROPIC_API_KEY`)
- `openai` — qualquer endpoint OpenAI-compatible (OpenRouter, Groq, vLLM, Ollama…)
- `scripted` — modelo determinístico offline: busca uma vez e responde
  extrativamente; mantém loop, ferramentas, memória e verificação
  genuinamente exercitados sem rede (é o que a CI roda)
- `null`/`auto` (padrão) — Claude, depois OpenAI-compatible, depois scripted

```python
# Criar agente
agent = AgenticRAG(context)
agent.initialize()

# Fazer pergunta
response = agent.query("O que é IA?")

# Acessar resposta
print(response.answer)
print(response.references)      # Documentos usados
print(response.steps)           # Chamadas de ferramenta + resposta + verificação
print(response.metadata)        # agent_mode, iterations, tool_calls, confidence
```

**Fluxo de execução:**
1. Recebe a query do usuário
2. Envia a query e as definições das ferramentas ao modelo
3. O modelo chama `search_documents` (possivelmente várias vezes, reformulando
   a consulta entre as chamadas) e lê os resultados
4. Os passos 2–3 se repetem até o modelo responder ou atingir
   `agent.max_iterations`
5. A resposta é conferida contra as evidências acumuladas e sinalizada se
   divergir
6. Resposta e metadados são salvos na memória
7. Retorna resposta estruturada

### 9. **main.py** - Interface CLI
Ponto de entrada do sistema:

**Modo Interativo:**
- Loop de perguntas e respostas
- Comandos: stats, history, clear, quit
- Exibe respostas formatadas

**Modo de Tarefa Única:**
- Executa uma pergunta e sai
- Útil para scripts

```bash
# Interativo
python main.py

# Tarefa única
python main.py --task "sua pergunta aqui"

# Com configuração customizada
python main.py --config meu_config.yaml
```
