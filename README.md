# Sistema RAG Agêntico 🤖

*[English version](README.en.md)*

[![CI](https://github.com/lucianoon/rag-agentic-system/actions/workflows/ci.yml/badge.svg)](https://github.com/lucianoon/rag-agentic-system/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Licença: MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-green.svg)](LICENSE)

Um sistema completo de **Retrieval-Augmented Generation (RAG) com comportamento agêntico** para recuperação e processamento inteligente de informações.

> **Não é o mesmo projeto que o [Enterprise RAG System](https://github.com/lucianoon/enterprise-rag-system).**
> A diferença está no **número de passos**. Lá a recuperação acontece **uma vez**
> e o objetivo é a qualidade da lista ranqueada (fusão híbrida BM25 + vetorial,
> Recall@K / MRR). Aqui a recuperação é uma **ferramenta** que o modelo chama
> quantas vezes precisar, reformulando a consulta entre as chamadas. Problemas
> diferentes: aquele repo otimiza qualidade de ranqueamento, este otimiza
> raciocínio em múltiplos passos sobre um corpus.

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

### Docker (alternativa)

```bash
# Modo interativo (funciona sem chave de API: cai no modelo offline)
docker compose run --rm rag-agent

# Tarefa única
docker compose run --rm rag-agent --task "Resuma o documento X"

# Adicionar documentos ao corpus
docker compose run --rm rag-agent --add-docs data/processed/meu-doc.md
```

O corpus e a memória persistem em `./data` (volume montado).

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

## Trocando de modelo ou de provedor

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

## Arquitetura do código

Passeio módulo a módulo pelos componentes (`types`, `config`, `embeddings`,
`vector_store`, `retriever`, `tools`, `agent`): [docs/ARQUITETURA.md](docs/ARQUITETURA.md).

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
ruff format .

# Lint e checagem de tipos
ruff check .
mypy
```

Os mesmos três gates — `ruff check`, `mypy` e `pytest` — rodam automaticamente no CI (GitHub Actions) a cada push e pull request na branch `main`; veja `.github/workflows/ci.yml`.

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

## 📚 Próximos Passos

Melhorias planejadas:
- Backend FAISS/Qdrant para vector store com persistência
- Novas ferramentas para o agente (leitura de documento completo, anotações)
- Verificação via LLM-judge além da heurística lexical
- Empacotamento com `pyproject.toml` (o Dockerfile e o compose já existem)

## 📄 Licença

Este projeto está licenciado sob a Licença MIT.
