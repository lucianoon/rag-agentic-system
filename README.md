# Sistema RAG Agêntico 🤖

*[English version](README.en.md)*

Um sistema completo de **Retrieval-Augmented Generation (RAG) com comportamento agêntico** para recuperação e processamento inteligente de informações.

A diferença para o [Enterprise RAG System](https://github.com/lucianoon/enterprise-rag-system)
está no **número de passos**: lá a recuperação acontece uma vez e o foco é a qualidade da
lista ranqueada (fusão híbrida BM25 + vetorial, Recall@K / MRR). Aqui a recuperação é uma
**ferramenta** que o modelo chama quantas vezes precisar, reformulando a consulta entre as
chamadas — o foco é responder perguntas que uma única query não resolve.

## ✨ Funcionalidades

- **Loop Agêntico com Tool Use**: o modelo decide quais ferramentas chamar
  (`search_documents`, `get_task_history`), lê os resultados e itera até
  responder — limitado por `agent.max_iterations`
- **Dois cérebros intercambiáveis**: Claude (tool use nativo, via
  `ANTHROPIC_API_KEY`) ou um modelo scripted determinístico que mantém o mesmo
  loop rodando offline (CI, demos sem chave)
- **Verificação de Groundedness**: a resposta final é conferida contra as
  evidências que as ferramentas retornaram (`verification.min_confidence`);
  respostas não confirmadas são sinalizadas
- **Recuperação Multi-Fonte**: Ingestão de documentos do sistema de arquivos
- **Embeddings Flexíveis**: Sentence-Transformers com fallback automático para TF-IDF
- **Busca Vetorial**: Similaridade de cosseno em memória
- **Memória de Tarefas**: Armazenamento SQLite para histórico de tarefas
- **CLI Interativa**: Interface de linha de comando amigável
- **Pipeline Configurável**: Configuração baseada em YAML

## 🚀 Instalação

### Pré-requisitos
- Python 3.8 ou superior
- Git

### Início Rápido

```bash
# Clone o repositório
git clone https://github.com/lucianoon/rag-agentic-system.git
cd rag-agentic-system

# Instale as dependências
pip install -r requirements.txt

# Execute o sistema
python main.py
```

## 📖 Como Usar

### Modo Interativo (Padrão)

```bash
python main.py
```

Isso inicia o agente RAG interativo:

```
🤖 RAG Agentic System - Interactive Mode
Type 'help' for commands, 'quit' to exit

RAG> O que é machine learning?
🔍 Processing: O que é machine learning?

📄 Response:
Baseado nos documentos recuperados...
```

### Modo de Tarefa Única

```bash
python main.py --task "Explique computação quântica"
```

### Adicionando Documentos

1. Coloque seus arquivos `.txt` ou `.md` no diretório `data/processed/`
2. Execute o sistema - ele irá indexá-los automaticamente na inicialização

### Configuração

Edite `config/default.yaml` para personalizar:
- Modelos de embedding
- Configurações do vector store
- Parâmetros de recuperação
- Configurações de memória

### Comandos Disponíveis

No modo interativo:
- `<pergunta>` - Faça uma pergunta
- `stats` - Exibe estatísticas do sistema
- `history` - Mostra histórico de tarefas recentes
- `clear` - Limpa o vector store
- `quit`/`exit`/`q` - Sai do sistema

## 📁 Estrutura do Projeto

```
rag-agentic-system/
├── main.py                     # Ponto de entrada CLI
├── requirements.txt            # Dependências
├── README.md                   # Este arquivo
├── config/
│   └── default.yaml           # Arquivo de configuração
├── src/rag_agent/             # Código principal da aplicação
│   ├── __init__.py            # Inicialização do pacote
│   ├── agent.py               # Agente RAG principal
│   ├── loop.py                # Loop agêntico: turnos, tool use, verificação
│   ├── tools.py               # Definição e execução das ferramentas
│   ├── llm.py                 # Backends de modelo (Claude / scripted)
│   ├── context.py             # Contexto de execução compartilhado
│   ├── config.py              # Gerenciamento de configuração
│   ├── embeddings.py          # Backends de embedding
│   ├── memory.py              # Armazenamento de memória
│   ├── pipeline.py            # Orquestração do pipeline
│   ├── retrieval.py           # Recuperação de documentos
│   ├── types.py               # Modelos de dados
│   └── vector_store.py        # Armazenamento vetorial
└── data/
    └── processed/             # Coloque documentos aqui
```

## 🔧 Explicação do Código

### Arquitetura do Sistema

O sistema é organizado em módulos independentes que trabalham juntos:

#### 1. **types.py** - Modelos de Dados
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

#### 2. **config.py** - Gerenciamento de Configuração
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

#### 3. **embeddings.py** - Backend de Embeddings
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

#### 4. **vector_store.py** - Armazenamento Vetorial
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

#### 5. **retrieval.py** - Recuperação de Documentos
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

#### 6. **memory.py** - Armazenamento de Memória
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

#### 7. **pipeline.py** - Orquestração do Pipeline
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

#### 8. **agent.py + loop.py + tools.py + llm.py** - O Agente

O coração do sistema é um loop de tool use: o modelo recebe a tarefa e as
ferramentas disponíveis, decide o que chamar, lê os resultados e itera até
produzir a resposta final (ou atingir `agent.max_iterations`). Depois, a
resposta é verificada contra as evidências coletadas pelas ferramentas
(`verification.min_confidence`) — respostas que se afastam dos documentos
recuperados recebem um aviso explícito.

Dois modelos implementam a mesma interface (`config.llm.provider`):

- `anthropic` — Claude com tool use nativo (requer `ANTHROPIC_API_KEY`)
- `scripted` — modelo determinístico offline: busca uma vez e responde
  extrativamente; mantém loop, ferramentas, memória e verificação
  genuinamente exercitados sem rede (é o que a CI roda)
- `null`/`auto` (padrão) — Claude quando a chave existe, senão scripted

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

#### 9. **main.py** - Interface CLI
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

### Trocando de modelo ou de provedor

O loop do agente depende só do protocolo `AgentModel`, então três backends
rodam a mesma máquina de loop:

| `llm.provider` | Backend |
|---|---|
| `null` / `auto` (padrão) | Claude se houver chave; senão OpenAI-compatible se houver endpoint; senão o scripted |
| `anthropic` | Claude com tool use nativo |
| `openai` | Qualquer endpoint OpenAI-compatible — OpenAI, OpenRouter, Groq, Together, vLLM, Ollama, LM Studio |
| `scripted` | Modelo determinístico offline (é o que a CI roda) |

O backend OpenAI-compatible traduz nos dois sentidos: schemas de ferramenta e
histórico de mensagens na ida, tool calls na volta — o loop continua falando
nos formatos da Anthropic.

```bash
# OpenRouter, Groq, Together…
export RAG_LLM_BASE_URL=https://openrouter.ai/api/v1
export RAG_LLM_API_KEY=sk-or-v1-...
export RAG_LLM_MODEL=meta-llama/llama-3.3-70b-instruct

# Ollama local — sem credencial nenhuma
export RAG_LLM_BASE_URL=http://localhost:11434/v1
export RAG_LLM_MODEL=llama3.1
```

Esse backend exige `pip install openai`.

## ⚙️ Configuração Detalhada

### Embeddings
```yaml
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: null  # null = auto, "cpu" ou "cuda"
  use_tfidf_fallback: true  # Usar TF-IDF se transformers indisponível
  vector_dimension: 384  # Dimensão dos vetores
```

### Vector Store
```yaml
vector_store:
  backend: "simple"  # Atualmente apenas "simple" suportado
  embedding_dimension: 384  # Deve coincidir com embeddings
  similarity_metric: "cosine"  # Métrica de similaridade
  top_k: 5  # Número de documentos a recuperar
```

### Retrieval
```yaml
retrieval:
  sources:
    - "data/processed"  # Diretórios para escanear
  file_extensions:
    - ".txt"  # Extensões de arquivo permitidas
    - ".md"
  chunk_size: 512  # Máximo de palavras por chunk
  chunk_overlap: 64  # Palavras de sobreposição entre chunks
```

### Memória
```yaml
memory:
  enabled: true  # Ativar/desativar memória
  database_path: "data/memory.db"  # Caminho do banco SQLite
  cleanup_days: 30  # Deletar logs mais antigos que X dias
  importance_threshold: 0.3  # Limiar para salvar tarefas
```

## 🔄 Fluxo de Dados

```
1. Usuário faz pergunta
   ↓
2. Pergunta e definições das ferramentas vão para o modelo
   ↓
3. O modelo decide chamar search_documents
   ↓
4. A query vira vetor e o vector store retorna os chunks mais próximos
   ↓
5. O modelo lê os resultados e ou reformula a consulta e busca de novo
   (volta ao passo 3) ou escreve a resposta
   ↓
6. A resposta é conferida contra tudo que as ferramentas retornaram
   ↓
7. Resposta + metadados salvos na memória
   ↓
8. Resposta exibida ao usuário, sinalizada se a confiança ficar abaixo do limiar
```

Os passos 3–5 são o loop: quantas vezes eles rodam é decisão do modelo,
limitada por `agent.max_iterations`.

## 🧪 Desenvolvimento

### Testando o Sistema

O projeto possui uma suíte de testes em `tests/` cobrindo chunking e ingestão de documentos, busca vetorial por similaridade de cosseno, fallback de embeddings para TF-IDF, memória de tarefas em SQLite, carregamento de configuração YAML e o fluxo completo do agente.

Os testes **não** precisam de LLM, chaves de API nem acesso à rede: o caminho de fallback TF-IDF é forçado, então nenhum modelo do sentence-transformers é baixado.

```bash
# Instalar dependências mínimas para os testes
pip install numpy scikit-learn pyyaml pytest

# Executar a suíte completa
pytest

# Executar um módulo específico
pytest tests/test_vector_store.py -v

# Formatar código
black src/ tests/

# Verificar qualidade do código
flake8 src/ tests/
```

A mesma suíte é executada automaticamente no CI (GitHub Actions) a cada push e pull request na branch `main` — veja `.github/workflows/ci.yml`.

### Adicionando Novos Retrievers

```python
from src.rag_agent.retrieval import DocumentIngestor

class MeuRetriever:
    def load_documents(self):
        # Sua lógica aqui
        pass
```

## 📊 Exemplos de Uso

### Exemplo 1: Perguntas e Respostas Simples

```python
from src.rag_agent import AgenticRAG, load_config, create_context

# Configurar
config = load_config()
context = create_context(config)
agent = AgenticRAG(context)
agent.initialize()

# Perguntar
response = agent.query("O que é Python?")
print(response.answer)
```

### Exemplo 2: Adicionar Documentos Manualmente

```python
# Adicionar documentos
agent.add_documents(["doc1.txt", "doc2.txt"])

# Buscar
response = agent.query("Busca nos documentos adicionados")
```

### Exemplo 3: Ver Estatísticas

```python
stats = agent.get_stats()
print(f"Documentos: {stats['total_documents']}")
print(f"Embeddings: {stats['embeddings_stored']}")
```

## 🤝 Contribuindo

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT.

## 📚 Próximos Passos

Melhorias planejadas:
- Backend FAISS/Qdrant para vector store com persistência
- Novas ferramentas para o agente (leitura de documento completo, anotações)
- Verificação via LLM-judge além da heurística lexical
- Empacotamento com pyproject.toml e Dockerfile

## 🆘 Suporte

Para problemas ou questões:
- Email: lucianomevam@outlook.com
- GitHub Issues: https://github.com/lucianoon/rag-agentic-system/issues
