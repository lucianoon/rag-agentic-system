# Changelog

Todas as mudanças relevantes deste projeto são registradas aqui. O formato segue
o espírito do [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/); as
versões seguem [SemVer](https://semver.org/lang/pt-BR/).

## Unreleased

### Adicionado
- Backend OpenAI-compatible, selecionado por configuração, ao lado do backend Claude e do modelo determinístico offline.
- Aplicação containerizada: `Dockerfile` e `docker-compose.yml`.

### Alterado
- CI: `actions/checkout` v4 para v7 e `actions/setup-python` v5 para v7.
- CI passa a bloquear em `ruff` e `mypy`, substituindo `black` e `flake8` (#6).
- CI instala o SDK `openai`; a seleção desse backend fica condicional à presença do SDK.
- Passeio de código movido do README para `docs/`; lista de backends corrigida.
- Distinção em relação ao Enterprise RAG System destacada em callout no README.
- README em português volta a ser o principal; a versão em inglês fica em `README.en.md`.
- Documentação expõe sinais verificáveis do projeto (testes, CI, evidências).
- Atualização automática de versões pelo Dependabot desativada; bumps passam por revisão manual.

### Corrigido
- Parâmetro `temperature` quebrava toda chamada real ao provedor.

## 0.1.0 — 2026-07-25

Primeira versão marcada. O sistema passa a ter um loop agêntico de verdade.

### Adicionado
- Loop agêntico com tool use do Claude sobre a recuperação, modelo determinístico
  offline com roteiro fixo (para CI e demonstração sem chave) e verificação de
  embasamento da resposta nos trechos recuperados (#2).
- Testes unitários para configuração, embeddings, vector store, memória,
  recuperação, pipeline e script de ingestão de documentos.
- Testes de ingestão de documentos e licença MIT.
- Workflow de testes em Python no GitHub Actions.
- README com resumo em inglês, contraste com o Enterprise RAG System e diagramas
  de fluxo atualizados para o loop agêntico.

### Alterado
- Documentação consolidada em um único README (#1).

### Corrigido
- TF-IDF devolve vetores nulos quando o texto só contém stop words, em vez de falhar.
- Caminhos desatualizados na documentação.

## Origem — 2025-10-14

- Implementação inicial do pipeline RAG, script de gestão de documentos e README em português com explicação do código.
