# Tech Challenge — Fases 1 e 2

Projeto acadêmico da **Pós-graduação IA para Devs (FIAP)**:

- **Fase 1:** classificação de câncer de mama e SOP (PCOS), avaliação e explicabilidade.
- **Fase 2 — Projeto 1:** otimização genética dos modelos e interpretação agregada por LLM.
- **Fase 2 — Projeto 2:** otimização de rotas para distribuição simulada de medicamentos e insumos.

## Guia de execução para o avaliador

Este README contém a sequência de preparação, geração dos resultados e
demonstração. O [Manual Técnico](./MANUAL_TECNICO.md) descreve a arquitetura,
os algoritmos, os contratos da API, a persistência e a infraestrutura.

### 1. Requisitos e preparação

- Windows com PowerShell, Git e Python 3 disponíveis no terminal.
- Internet para instalar dependências, carregar os recursos externos dos
  gráficos e utilizar um provedor LLM online.
- Chave própria do Google AI Studio para demonstrar o Gemini.
- Podman/Ollama e Kubernetes **não são necessários** para a execução com Gemini;
  são alternativas ou demonstrações opcionais.

Execute no PowerShell:

```powershell
git clone https://github.com/rsjtecnologia/tech-challenge-fase2.git
Set-Location .\tech-challenge-fase2
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\pre_execucao.ps1
```

O [preparador](./pre_execucao.ps1) cria/ativa `.venv` e instala
[requirements.txt](./requirements.txt). Ele seleciona a versão mais recente
do Python 3 detectada; não há ambiente de dependências travado por lockfile.
Se já clonou o projeto, entre na pasta existente em vez de clonar novamente.

### 2. Gerar os resultados das Fases 1 e 2

Antes de iniciar a web, execute:

```powershell
.\run_notebooks.ps1 -All
```

Esse comando executa os pipelines clínicos, os experimentos genéticos,
as validações de integração e os cenários de rotas registrados. Não exige
API ativa nem chave Gemini. O treinamento e a otimização são locais.

**Aguarde a conclusão:** são dois estudos × cinco classificadores × três
configurações genéticas, totalizando 30 experimentos. O tempo depende da máquina.
O aviso periódico de espera do kernel não significa falha nem comprova avanço.

Os datasets clínicos, Olist e inventário estão incluídos; o CSV CNES é opcional.
Os resultados são gerados em `src\db\outputs`, ignorado pelo Git, e não devem
ser presumidos presentes em um clone novo. Os notebooks executados são salvos
em `src\db\outputs\executed-notebooks`, sem sobrescrever os originais.

Para acompanhar a otimização, em outro terminal na mesma cópia:

```powershell
Get-Content .\src\db\outputs\phase2-genetic\optimization.log -Tail 10 -Wait
```

O log fica disponível após o início dessa etapa. Ctrl+C nesse terminal apenas
encerra a visualização do log.

Também é possível usar `.\run_notebooks.ps1` e escolher:

| Opção | Execução |
|---|---|
| `T` | Todos os notebooks, na ordem registrada. |
| `1` / `2` | Pipeline de câncer de mama / PCOS. |
| `3` | Otimização genética via CLI, por estudo. |
| `4` | Notebooks interativos no Jupyter; requer API ativa para as consultas. |
| `5` | Experimentos e integração da Fase 2. |
| `6` | Rotas e distribuição simulada. |

### 3. Preparar o acesso ao Gemini

1. Acesse [Google AI Studio — chaves](https://aistudio.google.com/apikey).
2. Crie uma chave no seu projeto e confira modelo e cotas da conta.
3. Use a chave somente na entrada oculta do launcher. Não edite o código e
   não coloque a credencial em notebooks, commits, vídeos ou capturas.

O padrão configurado é `gemini-3.1-flash-lite`. É possível informar outro
modelo com `-Model`; por exemplo, `gemini-3.5-flash-lite`, se disponível na
sua conta. Gratuidade, cotas e disponibilidade dependem do provedor:
consulte [preços](https://ai.google.dev/gemini-api/docs/pricing) e
[limites](https://ai.google.dev/gemini-api/docs/rate-limits).
Não há garantia de disponibilidade contínua nem troca automática de modelo.

### 4. Iniciar API e dashboards

Na raiz, após gerar os resultados:

```powershell
.\run.ps1
```

Informe a chave quando aparecer **Chave API de Gemini (entrada oculta; nao
sera salva)**. Se `GEMINI_API_KEY` já estiver no ambiente do terminal, ela
será reutilizada.

O launcher:

1. prepara o ambiente e verifica os artefatos locais;
2. testa uma geração LLM curta com mensagem sintética, que pode ter custo;
3. inicia o FastAPI/Uvicorn;
4. verifica saúde e prontidão e abre o navegador.

**Confirmações diferentes:**

- **Geracao LLM: OK neste momento**: a chamada ao provedor funcionou nesse teste.
- **LLM nao validada**: a API/web pode iniciar, mas as consultas LLM podem falhar.
  O erro é exibido; não há resposta simulada para esconder a falha.
- **API local: OK. Dados locais: OK**: `/healthz` e `/readyz` responderam;
  isso não comprova disponibilidade do Gemini.
- Dados ausentes, configuração inválida ou falha interna interrompem a
  inicialização com diagnóstico. Iniciar a web não gera artefatos faltantes.

Mantenha o terminal aberto. Acesse:

| Página | URL |
|---|---|
| Câncer de mama | http://127.0.0.1:8020/ |
| SOP/PCOS | http://127.0.0.1:8020/pcos |
| Distribuição e rotas | http://127.0.0.1:8020/rotas |
| API interativa | http://127.0.0.1:8020/docs |

Variações úteis:

```powershell
.\run.ps1 -Provider Gemini -Model gemini-3.5-flash-lite
.\run.ps1 -NoBrowser
.\run.ps1 -Port 8021
.\run.ps1 -TimeoutSeconds 120
```

Use apenas uma execução por porta. Encerre com **Ctrl+C**; para reiniciar,
execute o launcher novamente. Aumentar o timeout não corrige indisponibilidade
persistente do provedor.

### 5. Sequência sugerida de avaliação

1. **Fase 1:** nos dois dashboards clínicos, examine métricas, ranking,
   matrizes de confusão, gráficos, seletor de modelos e filtros de amostras.
   São resultados pré-computados, não diagnóstico online de novos pacientes.
2. **Fase 2 — modelos:** compare baseline e otimizado; observe recall,
   precisão e F1. Ganhos não são universais: uma métrica pode cair no teste.
3. **LLM:** clique em **Gerar interpretação**. Selecione um exemplo e envie
   com **Perguntar à LLM** ou Enter; o exemplo sozinho apenas preenche o campo.
4. **Rotas:** selecione **Demonstração fictícia — capital**, confira/execute a
   otimização e compare distância, atrasos e entregas não atendidas.
   Examine mapa, veículos e manifesto; teste instruções ou relatório diário.
5. **Fontes alternativas:** Olist e híbrido usam dados incluídos. CNES exige
   CSV adicional; a ausência dessa fonte opcional não invalida a demonstração.
6. **API:** em `/docs`, examine os contratos e execute consultas locais.
   Chamadas de interpretação/assistente podem consumir cota LLM.

Para notebooks com widgets, mantenha a API ativa e, em outro terminal:

```powershell
.\run_notebooks.ps1
```

Escolha `4` e o notebook desejado. Se mudou a porta, use
`.\run_notebooks.ps1 -ApiPort 8021`. Os notebooks usam a API local; não copie
a chave para eles. O batch pula células interativas.

### 6. Diagnóstico de execução

| Sintoma | Ação |
|---|---|
| Script bloqueado | Use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` nesse terminal. |
| Arquivo de resultados ausente | Execute `.\run_notebooks.ps1 -All` e confira a conclusão sem erros. |
| Porta ocupada | Encerre sua execução anterior com Ctrl+C ou escolha `-Port 8021`; nenhum processo é encerrado automaticamente. |
| Navegador não abriu | Confira as mensagens e abra a URL manualmente; `-NoBrowser` mantém as verificações locais. |
| HTTP 401/403 do provedor | Confira chave, projeto e permissões; reinicie com a chave correta. |
| HTTP 404 do provedor | Confira o identificador do modelo e o acesso da conta. |
| HTTP 429 do provedor | Confira cotas e eventual faturamento; evite repetir chamadas continuamente. |
| HTTP 503 do provedor | Há até três tentativas com espera progressiva. Se persistir, verifique o provedor; não significa que a API local caiu. |
| Timeout | Verifique rede/provedor; se necessário, ajuste `-TimeoutSeconds`. |
| Gráficos não carregam | Confira os artefatos e o acesso ao CDN do Chart.js; a interface não é totalmente independente de internet. |

Para verificar a API sem chamar o Gemini:

```powershell
Invoke-RestMethod http://127.0.0.1:8020/healthz
Invoke-RestMethod http://127.0.0.1:8020/readyz
```

Para testar **somente o provedor**, sem abrir navegador, sem depender de
artefatos e sem iniciar outro servidor (funciona com a porta já ocupada):

```powershell
.\run.ps1 -Provider Gemini -CheckLLMOnly
```

O teste usa uma mensagem sintética e a chave na entrada oculta; pode consumir
cota. Exibe modelo, host, timeout e tempo até a falha. Para outro modelo, inclua
`-Model IDENTIFICADOR_DO_MODELO`. Compartilhe apenas o diagnóstico, nunca a chave.
Quando disponível, o erro inclui um status Gemini validado, como
`RESOURCE_EXHAUSTED` (cota) ou `UNAVAILABLE` (indisponibilidade).

`/healthz` verifica vida do processo; `/readyz` verifica a existência dos
arquivos exigidos pelo perfil, incluindo gráficos e tabelas dos dois dashboards.
Não verifica a validade integral dos dados nem o Gemini.

### Testes da aplicação

Para preparar as dependências de desenvolvimento e executar a suíte:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Os testes de integração LLM usam mocks ou servidor local; não comprovam que
uma conta Gemini real tenha acesso, cota ou disponibilidade.

### 7. Alternativas opcionais

- **Ollama local:** `.\run.ps1 -Provider Ollama`. Requer Podman e preparação
  do modelo local; veja [execução técnica](./MANUAL_TECNICO.md#1-iniciar-o-projeto).
- **Sem provedor:** com os artefatos gerados, execute em um terminal sem
  variáveis `LLM_*` configuradas:

  ```powershell
  .\.venv\Scripts\python.exe -m uvicorn src.web.app:app --host 127.0.0.1 --port 8020
  ```

  Resumos gerais e rotas podem usar regras locais identificadas como tal;
  perguntas clínicas livres exigem LLM. Isso **não demonstra Gemini**.
- **Kubernetes/HPA:** opcional, não iniciado pelo launcher; veja a
  [infraestrutura](./MANUAL_TECNICO.md#9-infraestrutura-opcional-e-escalabilidade).

## Documentação e limites

- [Manual Técnico](./MANUAL_TECNICO.md): descrição técnica completa e reprodução.

> Uso acadêmico: não substitui diagnóstico médico nem valida uma operação real
> de distribuição hospitalar. As rotas usam distâncias Haversine, não ruas ou
> trânsito real; demandas e associações médicas podem ser simuladas. Não envie
> dados identificáveis de pacientes à LLM. Se uma chave for exposta, revogue-a
> no provedor e gere outra.
