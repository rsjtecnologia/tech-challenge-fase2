# Manual técnico — Tech Challenge

Projeto acadêmico FIAP com estudos de classificação clínica, otimização por
algoritmos genéticos e distribuição simulada de medicamentos e insumos.

**Organização da documentação:** o [README](./README.md) é o guia de execução
do avaliador. Este manual mantém os comandos de reprodução e descreve a
implementação técnica: arquitetura, dados, algoritmos, APIs, integração LLM,
persistência, interface, testes e infraestrutura.

> **Uso acadêmico:** os modelos não são dispositivos médicos e não substituem
> avaliação profissional. As rotas não representam uma operação real.

## Sumário

1. [Execução e reprodução](#1-iniciar-o-projeto)
2. [Arquitetura](#2-arquitetura-e-responsabilidades)
3. [Pipelines clínicos](#3-fase-1--pipelines-clínicos-e-contratos-web)
4. [Otimização dos classificadores](#4-fase-2--otimização-genética-dos-classificadores)
5. [LLM e cache](#5-integração-llm-qualidade-e-cache)
6. [Logística](#6-logística--cenários-algoritmo-e-persistência)
7. [API e interface](#7-api-http-e-interface)
8. [Inicialização e observabilidade](#8-inicialização-execução-de-notebooks-e-observabilidade)
9. [Infraestrutura opcional](#9-infraestrutura-opcional-e-escalabilidade)
10. [Dados e artefatos](#10-datasets-artefatos-e-atribuição)
11. [Testes](#11-testes-e-reprodução)
12. [Segurança e limitações](#12-segurança-limitações-e-próximos-passos)

## 1. Iniciar o projeto

Requisitos: Windows, PowerShell, Git, Python 3 e internet para instalar
dependências. Clone o repositório e prepare o ambiente na raiz:

```powershell
git clone https://github.com/rsjtecnologia/tech-challenge-fase2.git
Set-Location .\tech-challenge-fase2
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\pre_execucao.ps1
```

O script cria/ativa `.venv` e instala `requirements.txt`. Os datasets clínicos,
Olist e o inventário Hospital Supply Chain estão incluídos no repositório;
resultados e artefatos são gerados localmente e não estão no Git.

### Gerar os resultados

```powershell
.\run_notebooks.ps1
```

No menu:

1. `T` executa **todos os notebooks** na ordem: EDA e pipelines clínicos,
   experimentos genéticos dos dois estudos, validações de integração e rotas
   demo/Olist/híbridas. Não exige API nem chave.
2. As células com widgets são puladas no batch; para mapa e consultas Gemini,
   inicie a API e use `4` para abrir os notebooks no Jupyter.
3. Para executar tudo sem menus: `.\run_notebooks.ps1 -All`.

O launcher verifica se todos os `.ipynb` estão registrados; notebooks novos não
registrados geram erro explícito. Execuções são salvas em
`src\db\outputs\executed-notebooks`, preservando os notebooks originais.
Os grupos `5` (experimentos/integração) e `6` (rotas) também podem ser executados
separadamente; `3` mantém a otimização via CLI por estudo.

| Notebook da Fase 2 | Cobertura |
|---|---|
| `notebooks\phase2\01_otimizacao_modelos.ipynb` | Três experimentos por classificador/estudo, persistência, comparação baseline/GA e interpretação interativa. |
| `notebooks\phase2\02_integracao_validacao.ipynb` | Contratos API, contexto agregado, interpretação local, cache simulado e inspeção do HPA; não implanta cluster. |
| `notebooks\medical-routes\01_otimizacao_rotas.ipynb` | Otimização demo/Olist/híbrida, persistência/histórico, relatório local e interface com mapa, manifesto, filtros e assistente. |

As interfaces dos exports clínicos também demonstram métricas e perguntas.
Os notebooks reutilizam os módulos da aplicação; não duplicam algoritmos.
Gemini só é chamado pelas ações interativas solicitadas pelo avaliador.

Os resultados são salvos em `src\db\outputs`. A otimização pode levar vários
minutos. A célula `from src.optimization.cli import main` executa os 30
experimentos (dois estudos × cinco modelos × três configurações); ela só
libera o notebook seguinte quando termina. O executor exibe os logs em tempo
real e, a cada 30 segundos, informa que ainda aguarda o kernel. Esse aviso
não prova avanço do treinamento. O aviso Jupyter sobre TCP sem criptografia
não é um erro de treino.

Para acompanhar uma execução já iniciada com a versão anterior do executor,
abra outro terminal **na mesma cópia do projeto** e execute:

```powershell
Get-Content .\src\db\outputs\phase2-genetic\optimization.log -Tail 10 -Wait
```

Logs `model_start`, `experiment_complete` e `model_complete` mostram as etapas;
Ctrl+C nesse segundo terminal encerra apenas a visualização do log. Se o
arquivo não existir, confira a pasta da execução. Para executar diretamente, use:

```powershell
.\.venv\Scripts\python.exe -m src.optimization.cli
```

### Configurar a chave Google Gemini

O avaliador deve utilizar **uma chave própria**. Nenhuma chave acompanha o
repositório e não é necessário editar `run.ps1`.

1. Acesse o [Google AI Studio](https://aistudio.google.com/apikey) e entre com
   sua conta Google.
2. Na área de chaves de API, crie uma chave vinculada a um projeto Google.
   Se solicitado, selecione ou crie o projeto e aceite os termos aplicáveis.
3. Confira se a conta/projeto tem acesso ao modelo `gemini-3.1-flash-lite` e
   cota disponível. Acesso, gratuidade e cobrança dependem do modelo e da conta;
   consulte os [preços](https://ai.google.dev/gemini-api/docs/pricing) e
   [limites](https://ai.google.dev/gemini-api/docs/rate-limits).
4. Copie a chave para informá-la no terminal na próxima etapa. Não a coloque
   em arquivos Python, notebooks, `run.ps1`, commits ou capturas de tela.

A chave permite gerar explicações das métricas e textos do assistente de rotas.
Treinamento, classificadores, otimização genética, distâncias e métricas são
calculados localmente e não precisam dessa chave.

### Iniciar e conferir a aplicação

Depois de gerar os resultados, execute na raiz do projeto:

```powershell
.\run.ps1
```

1. O launcher prepara o ambiente e seleciona Gemini por padrão.
2. Quando aparecer **Chave API de Gemini (entrada oculta; nao sera salva)**,
   cole sua chave e pressione **Enter**. Os caracteres não serão exibidos.
3. Aguarde a validação dos arquivos locais e uma geração LLM curta com mensagem
   sintética, sem dados do projeto, usando a mesma integração da aplicação.
   Essa chamada pode ter custo no provedor online. HTTP 503 é repetido até três
   vezes. Se o provedor falhar, o launcher exibe o diagnóstico e inicia a
   API/web com o aviso **LLM nao validada**, sem alterar o modelo nem gerar uma
   resposta simulada. As ações LLM continuam sujeitas ao erro do provedor.
   Configuração inválida, falha interna ou arquivos ausentes interrompem a
   inicialização; a mensagem final inclui o diagnóstico, não apenas “confira
   o erro acima”. A geração é testada diretamente, sem depender de `/models`.
   Se faltarem artefatos, execute os notebooks; o launcher não gera resultados.
   O teste confirma disponibilidade naquele momento, não garante chamadas futuras.
4. Após **API/web iniciando**, aguarde **API local: OK. Dados locais: OK**:
   o verificador consulta `/healthz` e `/readyz`, inclusive com `-NoBrowser`.
   O navegador abre automaticamente somente quando ambas respondem. Mantenha esse
   terminal aberto. Para desativar a abertura automática, use
   `.\run.ps1 -NoBrowser`. Se o navegador não abrir, acesse manualmente:

- `http://127.0.0.1:8020/` — câncer de mama
- `http://127.0.0.1:8020/pcos` — SOP/PCOS
- `http://127.0.0.1:8020/rotas` — distribuição e rotas
- `http://127.0.0.1:8020/docs` — documentação da API

Nos painéis clínicos, confira as métricas e solicite a interpretação ou envie
uma pergunta sobre os resultados. Em `/rotas`, aguarde a otimização inicial e
teste **Instruções para motoristas**, **Relatório diário** ou **Perguntar**.
Selecionar um exemplo de pergunta apenas preenche o campo: é necessário enviar.
Essas ações usam a LLM configurada; resultados já disponíveis no cache podem
ser reutilizados. Iniciar a web não treina os classificadores nem gera seus
experimentos genéticos.

O script não salva a chave digitada. Se `GEMINI_API_KEY` já estiver definida no
ambiente do terminal, ela será utilizada sem solicitar outra chave. Para voltar
à entrada oculta, remova essa variável **do terminal atual** antes de iniciar:

```powershell
Remove-Item Env:\GEMINI_API_KEY -ErrorAction SilentlyContinue
.\run.ps1
```

Encerre com **Ctrl+C**. Ao iniciar novamente, informe a chave se ela não estiver
no ambiente. Os notebooks interativos usam a API local, portanto não é preciso
copiar a chave para eles: com a API ativa, abra outro terminal na raiz, execute
`.\run_notebooks.ps1`, escolha `4` e selecione o notebook desejado.

> **Privacidade:** somente contexto agregado/simulado e a pergunta são enviados
> ao provedor. Não informe dados pessoais ou clínicos. O uso do conteúdo pelo
> Google depende dos termos da conta. Se uma chave for exposta, revogue-a no
> AI Studio e crie outra; apagar o texto de um arquivo não revoga a credencial.

### Problemas comuns

| Sintoma | Procedimento |
|---|---|
| PowerShell bloqueia o script | Execute `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` nesse terminal. |
| Chave ausente, inválida ou sem permissão | Confira a chave e as permissões do projeto no AI Studio; reinicie e informe a chave correta. |
| Modelo não confirmado / HTTP 404 | Confira o identificador e o acesso na conta. Use `.\run.ps1 -Provider Gemini -Model IDENTIFICADOR_DO_MODELO` com um modelo compatível; não há troca automática. |
| HTTP 429 | Confira cota, limites e eventual faturamento do projeto. Evite repetir chamadas continuamente. |
| HTTP 503 do provedor | A aplicação faz até 3 tentativas no mesmo modelo, com espera progressiva e pequena variação aleatória, dentro do tempo configurado. Se persistir, confira a disponibilidade do Gemini e tente mais tarde; não há troca automática de modelo. No teste inicial, a API/web inicia com aviso de LLM não validada. O erro do provedor não significa que a API local caiu. Confira `http://127.0.0.1:8020/healthz` e `/readyz` (ajuste a porta se necessário). |
| Timeout | Confira rede e disponibilidade do provedor. Para espera maior, use `.\run.ps1 -TimeoutSeconds 120`; aumentar o tempo não corrige indisponibilidade HTTP 503. |
| Porta 8020 ocupada | Encerre sua execução anterior com Ctrl+C ou use `.\run.ps1 -Port 8021` e abra a nova URL. Para notebooks, use `.\run_notebooks.ps1 -ApiPort 8021`. |
| Métricas ou comparação genética ausentes | Execute os pipelines e experimentos descritos em **Gerar os resultados**; iniciar a web não produz esses arquivos. |

### Executar sem chave

Para usar a API sem provedor LLM, execute no terminal com o ambiente preparado
e sem variáveis `LLM_*` configuradas:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.web.app:app --host 127.0.0.1 --port 8020
```

Na página `/rotas`, selecione **Demonstração fictícia — capital** para testar sem
datasets extras. Os dados Olist/inventário incluídos também permitem o modo
híbrido. Sem provedor configurado, resumos gerais e o assistente de rotas usam
regras locais identificadas como tal, não geração Gemini; perguntas clínicas
livres exigem LLM. Essa alternativa não demonstra a integração com Gemini.

## 2. Arquitetura e responsabilidades

A aplicação separa processamento batch, serviços locais e geração de texto.
FastAPI serve páginas Jinja2 e APIs JSON; Uvicorn executa o servidor.
Não há um servidor web separado para cada painel nem um banco relacional:
a persistência desta entrega é baseada em arquivos.

```text
CSV de entrada
    -> notebooks / pipelines locais
    -> modelos, métricas, gráficos e tabelas JSON/Parquet
    -> loaders e APIs FastAPI
    -> dashboards e widgets dos notebooks

Dados clínicos -> experimentos GA -> comparação baseline/otimizado
Dados logísticos -> cenário -> GA/decoder -> plano e histórico
Métricas ou plano agregado + pergunta -> integração LLM -> texto validado
```

| Componente | Responsabilidade |
|---|---|
| [notebooks](./notebooks/) | Exploração, preparação, treino, avaliação, explicabilidade, exportação e demonstrações executáveis. |
| [optimization](./src/optimization/) | Busca genética clínica, experimentos, CLI, interpretação, cache e interface dos notebooks. |
| [logistics](./src/logistics/) | Fontes de dados, cenários, distribuição, VRP, persistência e assistente. |
| [app.py](./src/web/app.py) | Composição dos routers, templates, arquivos estáticos, logging e probes. |
| [services](./src/web/services/) | Leitura dos contratos clínicos JSON/Parquet e caches de dados. |
| [api](./src/web/api/) | Consultas dos dashboards clínicos, filtros, paginação e detalhes. |
| [static](./src/web/static/) | JavaScript/CSS dos dashboards e mapa SVG. |
| [infra](./infra/) | Configurações opcionais de container, Ollama e autoscaling. |

As dependências de [requirements.txt](./requirements.txt) cobrem:
Pandas/NumPy/PyArrow para dados; scikit-learn para ML; Matplotlib/Seaborn/SHAP
para análise; Jupyter/nbclient/nbformat/ipykernel para notebooks;
FastAPI/Uvicorn/Jinja2 para a aplicação. O transporte LLM usa a biblioteca
padrão Python, sem depender de um SDK Google.

### Perfis de execução

`APP_PROFILE` aceita `full` (padrão do launcher) e `diagnostics` (container).
O perfil completo inclui a API e a página de logística. O perfil de diagnóstico
mantém os painéis clínicos e a otimização, sem registrar as rotas logísticas.
Os arquivos exigidos por `/readyz` variam entre esses perfis.
Ambos exigem os contratos de gráficos e os Parquet dos dois dashboards;
o perfil `full` também exige o plano logístico demo.

## 3. Fase 1 — pipelines clínicos e contratos web

### Estudos e etapas

- **Câncer de mama:** atributos numéricos do Wisconsin Diagnostic, com alvo
  maligno/benigno. Não há processamento de imagens brutas.
- **PCOS:** estudo separado, com limpeza e engenharia de atributos próprias;
  os resultados não são intercambiáveis com os do câncer de mama.
- Os notebooks organizam exploração, pré-processamento, treino, avaliação e
  exportação. O estudo de câncer de mama também tem notebook específico de
  explicabilidade com importância de atributos e SHAP e discussão de limitações.

As métricas incluem medidas como recall, precisão, F1 e acurácia; as interfaces
também apresentam rankings, comparações e matrizes de confusão. Recall é
destacado por sua relação com falsos negativos, sem constituir validação clínica.
Importância e SHAP descrevem comportamento do modelo, não causalidade.

### Exportação e leitura

Os exports [câncer de mama](./notebooks/breast-cancer-wisconsin-data/08_export_web.ipynb)
e [PCOS](./notebooks/polycystic-ovary-syndrome-pcos/05_export_web.ipynb)
convertem os artefatos em contratos de consumo web:

- JSON: métricas, configurações e dados resumidos dos gráficos;
- Parquet: amostras e predições tabulares;
- formato largo para consulta por amostra e longo para comparação por modelo.

Os loaders [data_loader.py](./src/web/services/data_loader.py) e
[pcos_data_loader.py](./src/web/services/pcos_data_loader.py) usam Pandas e
`lru_cache(maxsize=1)` por função de carga. As APIs aplicam filtros/paginação e
serializam registros. Os endpoints de recarga limpam esse cache de dados;
ele não é o cache das interpretações LLM.

O navegador recebe JSON pela API, não acessa Parquet diretamente.
Os dashboards exibem resultados pré-computados; não recebem novos pacientes,
não treinam ao abrir a página e não são uma ferramenta de diagnóstico online.

## 4. Fase 2 — otimização genética dos classificadores

Implementação em [experiments.py](./src/optimization/experiments.py) e
[genetic.py](./src/optimization/genetic.py).

### Preparação e avaliação

O loader da Fase 2 reproduz transformações dos estudos:

- Wisconsin: mapeia `M/B` para `1/0` e remove identificadores/coluna vazia;
- PCOS: normaliza nomes, converte atributos hormonais para números, trata
  valores de beta-HCG `1.99` como ausentes e deriva máximo, diferença, razão,
  indicador AMH alto e indicadores de ausência.

O split é estratificado, com 80% de treino e 20% de teste, semente padrão 42.
O pipeline contém imputação pela mediana e padronização antes do classificador.
Na busca, o pipeline é ajustado em cada fold de validação cruzada estratificada,
sem ajustar o pré-processamento no holdout.

São otimizados separadamente:

1. Regressão Logística: regularização e balanceamento;
2. Árvore de Decisão: profundidade, divisões, folhas, critério e balanceamento;
3. SVM: `C`, `gamma`, kernel e balanceamento, com calibração;
4. Gradient Boosting: estimadores, taxa de aprendizagem, profundidade e amostragem;
5. Random Forest: estimadores, profundidade, folhas, atributos e balanceamento.

### Representação e evolução

Cada indivíduo é um dicionário de hiperparâmetros de um espaço discreto.
A seleção usa torneio; o cruzamento escolhe genes de cada pai; a mutação
substitui valores por alternativas do espaço. Há elitismo e cache de fitness
para não repetir a avaliação da mesma configuração dentro da busca.

A fitness é **maximizada**:

```text
fitness = 0,50 × recall + 0,30 × F1 + 0,20 × acurácia
```

| Experimento | População | Gerações | Mutação | Folds |
|---|---:|---:|---:|---:|
| `exploracao_conservadora` | 6 | 3 | 0,15 | 3 |
| `exploracao_ampla` | 10 | 3 | 0,30 | 3 |
| `refinamento` | 8 | 5 | 0,10 | 3 |

Dois estudos × cinco modelos × três configurações = 30 experimentos.
O vencedor é escolhido pela fitness de validação cruzada no treino; depois é
reajustado no treino completo e comparado com o baseline no mesmo holdout.
O baseline é treinado novamente com os padrões implementados na Fase 2,
não apenas copiado de uma tabela histórica da Fase 1.

O JSON registra configuração, melhores parâmetros, métricas, histórico de
gerações e quantidade de candidatos únicos. Melhor fitness no treino não
garante ganho em todas as métricas do teste nem ótimo global.

### CLI e persistência

```powershell
.\.venv\Scripts\python.exe -m src.optimization.cli
.\.venv\Scripts\python.exe -m src.optimization.cli --study pcos --output .\src\db\outputs\phase2-genetic\pcos_results.json
```

A [CLI](./src/optimization/cli.py) aceita `--study`, `--model`, `--output`
e `--interpret`. O segundo exemplo usa saída separada: o dashboard lê o
arquivo consolidado `optimization_results.json`; uma execução parcial não
deve ser confundida com a execução completa dos dois estudos.
`--interpret` solicita explicações após o treino e pode consumir cota LLM.

## 5. Integração LLM, qualidade e cache

### Provedores e transporte

[llm.py](./src/optimization/llm.py) compartilha o transporte entre clínica
e logística. O launcher permite Gemini, OpenAI, Groq, Qwen e Ollama.
Os quatro primeiros usam API compatível com OpenAI (`/chat/completions`);
Ollama usa `/api/chat`, sem streaming, com opções nativas e `keep_alive`.
A autenticação remota usa Bearer; a chave não vai no corpo dos prompts.

| Configuração | Uso |
|---|---|
| `LLM_PROVIDER` | `openai_compatible` ou `ollama`; ausência de configuração permite modo local limitado. |
| `LLM_BASE_URL` | Endpoint base; remoto exige HTTPS e Ollama exige host local. |
| `LLM_MODEL` | Identificador do modelo. |
| `LLM_API_KEY` | Credencial em memória/ambiente do processo. |
| `LLM_TIMEOUT_SECONDS` | Timeout; launcher online usa 45 s por padrão. |
| `LLM_NUM_PREDICT` | Sobrescreve o orçamento de tokens da tarefa. |
| `LLM_TEMPERATURE` | Temperatura validada entre 0 e 2. |
| `LLM_REASONING_EFFORT` | Opção compatível de raciocínio; Gemini no launcher usa `minimal`. |
| `LLM_ENABLE_THINKING` | Opção de APIs compatíveis; Qwen no launcher usa `false`. |

O launcher usa `GEMINI_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY` ou
`DASHSCOPE_API_KEY` quando disponíveis, ou solicita entrada oculta.
Ele define a configuração para o processo servidor e restaura as variáveis
`LLM_*`/perfil anteriores ao encerrar. Não grava a chave em arquivo.
As opções de raciocínio dependem da compatibilidade do modelo escolhido.

### Contexto e validação

Na clínica, o contexto contém métricas agregadas, identificação do estudo,
tamanho dos splits e comparação baseline/otimizado. Não inclui linhas nem
atributos individuais de pacientes. A resposta JSON contém `summary`,
`findings`, `limitations` e `disclaimer`.

A avaliação verifica estrutura, aviso clínico, ausência de certezas proibidas,
tamanho, idioma, referência a métricas e recomendação de validação.
Há resumos verificados para perguntas exemplificadas. Isso reduz erros,
mas não constitui prova de correção factual de todo texto livre.

Na logística, [llm.py](./src/logistics/llm.py) monta contexto reduzido por
ação, recupera destinos pertinentes à pergunta e usa planos/histórico locais.
Instruções, métricas e atribuições verificadas são preservadas pela aplicação;
o texto da LLM não pode recriar arbitrariamente as paradas do plano.

### Cache e falhas

Para reduzir a latência, a interpretação clínica solicita até 80 palavras
(60 nas perguntas), sem repetir toda a tabela já exibida. Os limites de saída
são 280 tokens para interpretação geral e 240 para perguntas. Na logística,
o texto é limitado a 50 palavras, com 240 tokens para perguntas/instruções e
280 para relatórios. Métricas e paradas verificadas continuam anexadas pelo
sistema. `LLM_NUM_PREDICT`, quando definido diretamente no ambiente do servidor,
sobrescreve esses orçamentos; um limite muito baixo pode truncar o JSON.
O launcher Gemini já usa raciocínio `minimal`. Esses limites reduzem o
trabalho de geração, mas não garantem tempo de resposta de um provedor externo.
Não aumentam o timeout e não convertem timeout em resposta local.

[cache.py](./src/optimization/cache.py) mantém cache em memória por processo:
64 entradas por padrão, TTL de 3.600 segundos, remoção LRU e locks para
agrupar chamadas concorrentes de mesma chave. A chave SHA-256 considera
contexto, pergunta, configuração LLM e revisão do prompt.
O namespace logístico também considera ação/plano e histórico no relatório semanal.

Somente respostas LLM elegíveis são armazenadas; na clínica, exige aprovação
da rubrica. Exceções não se tornam sucesso em cache. Não existe cache distribuído
entre workers/pods e reiniciar o processo perde as entradas.

HTTP 503 remoto tem até três tentativas com espera exponencial e jitter,
reduzindo o timeout disponível nas tentativas seguintes. O código não faz
troca automática de modelo nem repete indiscriminadamente erros de autenticação
ou cota. Truncamento, timeout e resposta inválida são explícitos.

Sem provedor configurado, resumos gerais e ações logísticas podem usar
regras locais com `source: local`; perguntas clínicas livres exigem LLM.
**Falha de um provedor configurado não aciona um falso sucesso local.**

## 6. Logística — cenários, algoritmo e persistência

### Domínio e fontes

[scenario.py](./src/logistics/scenario.py) define depósito, veículos,
entregas e prioridades. Veículos incluem capacidade, autonomia, velocidade,
refrigeração e início de jornada; entregas incluem demanda, prazo, tempo de
serviço, itens e necessidade de refrigeração.

| Fonte | Significado |
|---|---|
| `demo` | Cenário fictício, sem CSV adicional. |
| `olist_sp` | Geografia e agregações do Olist; não representa hospitais ou demanda médica real. |
| `cnes_sp` | Estabelecimentos do CSV CNES opcional. |
| `hybrid_sp` | Geografia Olist/CNES e perfil de inventário, com associação e demanda médicas simuladas. |

Os loaders [datasets.py](./src/logistics/datasets.py),
[cnes_dataset.py](./src/logistics/cnes_dataset.py) e
[distribution.py](./src/logistics/distribution.py) transformam as fontes
em cenários. [item_types.py](./src/logistics/item_types.py) centraliza tipos
de itens usados pela distribuição.

### Algoritmo de rotas

[genetic.py](./src/logistics/genetic.py) resolve uma variante de VRP:

- indivíduo: permutação em que cada entrega aparece exatamente uma vez;
- seleção por torneio, crossover OX e mutação por inversão de segmento;
- decoder por inserção: avalia veículo/posição e escolhe incremento de custo;
- restrições de capacidade, refrigeração e autonomia, incluindo retorno ao depósito;
- simulação de horários com velocidade e tempo de serviço;
- distância Haversine, não distância por malha viária.

A fitness logística é **minimizada**:

```text
custo = distância total + atraso ponderado + 1.000.000 × entregas não atendidas
```

Os prazos são tratados por penalidade de atraso; não são todos restrições
rígidas que inviabilizam uma rota. A API usa por padrão população 64,
160 gerações, mutação 0,18, torneio 4 e semente 42, com limites validados.
O resultado compara plano base e otimizado, rotas por veículo, distâncias,
atrasos, cargas, entregas não atendidas e evolução da busca.

### Planos e histórico

[service.py](./src/logistics/service.py) orquestra cenário, otimização e
persistência. Salva JSON em `src\db\outputs\phase2-logistics`:

- `latest_route_plan.json` e `route_plan_history.json` para `demo`;
- sufixo da fonte para os demais cenários, evitando misturar planos;
- histórico limitado a 500 registros e resumo com janela de sete dias.

A escrita usa arquivo temporário e substituição atômica para evitar plano
parcialmente gravado. Isso não equivale a transações multiusuário ou locks
distribuídos; a persistência local não é um banco compartilhado entre réplicas.

## 7. API HTTP e interface

### Endpoints

Os modelos Pydantic validam corpos, enums e limites; entrada inválida usa
HTTP 422. A documentação gerada fica em `/docs` e o contrato em `/openapi.json`.

| Método | Endpoint | Responsabilidade |
|---|---|---|
| GET | `/`, `/pcos`, `/rotas` | Páginas; logística somente no perfil `full`. |
| GET | `/healthz` | Vida do processo, sem consultar LLM. |
| GET | `/readyz` | Existência de artefatos exigidos pelo perfil. |
| GET | `/api/health`, `/api/pcos/health` | Diagnóstico dos dados clínicos. |
| GET | `/api/metricas`, `/api/pcos/metricas` | Métricas exportadas. |
| GET | `/api/modelos`, `/api/pcos/modelos` | Modelos e rankings. |
| GET | `/api/graficos`, `/api/pcos/graficos` | Dados para gráficos. |
| GET | `/api/amostras`, `/api/pcos/amostras` | Filtros e paginação. |
| GET | `/api/amostras/{sample_id}`, `/api/pcos/amostras/{sample_id}` | Detalhe de amostra existente. |
| GET | `/api/predicoes/resumo` | Resumo de predições do estudo de mama. |
| POST | `/api/cache/recarregar`, `/api/pcos/cache/recarregar` | Limpeza dos caches de dados clínicos. |
| GET | `/api/otimizacao/{study_name}` | Baseline, otimizado e configurações dos experimentos. |
| POST | `/api/interpretacao` | Explicação agregada para `breast_cancer` ou `pcos`. |
| GET | `/api/rotas/cenario` | Cenário de uma fonte. |
| POST | `/api/rotas/otimizar` | Executa GA e persiste plano. |
| GET | `/api/rotas/resultado`, `/api/rotas/historico` | Plano salvo e histórico por fonte. |
| POST | `/api/rotas/assistente` | Instruções, relatório diário/semanal ou pergunta. |

Exemplos de corpos JSON (ações LLM podem consumir cota):

```json
{"estudo": "pcos", "pergunta": "Em quais modelos o recall caiu após a otimização?"}
```

```json
{"source": "demo", "population_size": 64, "generations": 160, "mutation_rate": 0.18, "tournament_size": 4, "seed": 42}
```

```json
{"source": "demo", "action": "daily_report"}
```

A interpretação clínica limita a pergunta a 500 caracteres; o assistente
logístico a 1.000. As ações logísticas são `instructions`, `daily_report`,
`weekly_report` e `question`.

Dados ausentes/indisponíveis ou configuração LLM inválida podem produzir
HTTP 503 local; plano/estudo não encontrado pode produzir 404. Uma
`LLMProviderError` é traduzida para **HTTP 502 da API**, com o status original
do provedor no `detail`. Portanto, “provedor respondeu HTTP 503” no texto
não significa necessariamente que a resposta HTTP local tenha status 503.

### Front-end

Jinja2 entrega HTML com CSS/JavaScript próprios. As páginas clínicas usam
Chart.js por CDN e fontes Google externas; não há garantia de interface
completa offline. Os scripts consultam a API via HTTP para atualizar rankings,
filtros, amostras e interpretações. O componente de interpretação compartilha
tratamento de perguntas entre telas.

O mapa logístico é SVG local, com filtros de veículos/destinos, sequência
de paradas e manifesto. Não usa serviço de navegação por ruas. Os exemplos
de perguntas apenas preenchem o campo; o envio depende do botão ou Enter.

## 8. Inicialização, execução de notebooks e observabilidade

### Launcher

O parâmetro `-CheckLLMOnly` executa apenas o teste sintético da integração
selecionada, sem exigir arquivos gerados, sem abrir navegador nem iniciar
Uvicorn. A verificação de porta ocupada é dispensada nesse modo; as variáveis
de ambiente ainda são restauradas ao encerrar. A mensagem de diagnóstico
exibe modelo, host, timeout e duração da falha, sem imprimir a chave.
Erros Gemini podem incluir um identificador de status de uma lista permitida;
o corpo e a mensagem arbitrária do provedor não são expostos.

[run.ps1](./run.ps1) valida opções/porta, prepara `.venv`, configura provedor,
executa preflight e inicia Uvicorn em `127.0.0.1`, porta 8020, com `--reload`.
É um launcher local de desenvolvimento/demonstração, não uma implantação de
produção. Ollama pode usar [start_ollama.ps1](./scripts/start_ollama.ps1);
essa preparação é distinta do serviço externo Gemini.

[preflight.py](./src/web/preflight.py) separa os resultados:

| Código | Significado | Ação do launcher |
|---|---|---|
| `0` | Arquivos presentes e geração sintética válida. | Inicia API/web. |
| `1` | Arquivos ausentes ou configuração inválida. | Interrompe com diagnóstico. |
| `2` | Falha do provedor ou conteúdo diferente do JSON esperado. | Avisa `LLM nao validada` e inicia API/web. |

Outras falhas do processo também interrompem a inicialização. O launcher
captura o diagnóstico nativo para incluí-lo na mensagem final.
O preflight confirma uma geração curta, não todas as ações ou disponibilidade
futura. Ele consulta a existência dos arquivos de prontidão, não valida
integralmente todos os contratos exportados.

[open_browser.py](./scripts/open_browser.py) roda como auxiliar e espera
`/healthz` e `/readyz`, inclusive com `-NoBrowser`. Só então informa prontidão
e, se habilitado, abre o navegador. Se expirar ou não houver navegador,
emite instrução explícita; o auxiliar não substitui a supervisão do servidor.
Ao encerrar, o launcher limpa o auxiliar que iniciou, sem encerrar serviços alheios.

### Notebooks e logs

[run_notebooks.ps1](./run_notebooks.ps1) registra os notebooks, verifica
cobertura e respeita sua ordem de execução. [run_notebook.py](./run_notebook.py)
usa execução de notebook e salva cópias; widgets são separados do batch.
As interfaces em [notebook_ui.py](./src/optimization/notebook_ui.py)
consomem a mesma API local.

Os experimentos registram `model_start`, `experiment_complete` e
`model_complete` em `optimization.log`. Logística registra persistência e
otimização; a camada LLM registra retries/cache e estatísticas disponíveis do
Ollama. O middleware HTTP registra método, caminho, status e duração quando
o logger está habilitado em INFO. A
[configuração do container](./infra/autoscaling/logging.json) envia logs
formatados para stdout. Não há dashboard de monitoramento central embutido.

## 9. Infraestrutura opcional e escalabilidade

O launcher local não cria cluster nem faz autoscaling.

- [Containerfile](./infra/autoscaling/Containerfile): Python 3.12 slim,
  perfil `diagnostics`, um worker Uvicorn, porta 8000, usuário não root.
- [kubernetes.json](./infra/autoscaling/kubernetes.json): namespace,
  Deployment, Service ClusterIP e HPA.
- Probes de startup/liveness usam `/healthz`; readiness usa `/readyz`.
- Requests: CPU `100m`, memória `384Mi`; limites: CPU `500m`, memória `1Gi`.
- HPA: mínimo 1, máximo 4 réplicas, alvo CPU 60%; configurações de estabilização
  e políticas de aumento/redução estão no manifesto.
- Pod com filesystem somente leitura, sem elevação de privilégios,
  capabilities removidas e sem token de ServiceAccount montado automaticamente.

É necessário gerar os artefatos **antes** da construção da imagem:
o build copia `src`, incluindo os outputs locais disponíveis. Confira o
[Containerfile.dockerignore](./infra/autoscaling/Containerfile.dockerignore).
O manifesto usa `localhost/tech-challenge-api:phase2`; a imagem precisa estar
disponível nos nós do cluster ou ser adaptada para um registry acessível.
Também são necessários contexto Kubernetes válido e métricas de CPU
disponíveis ao HPA (por exemplo, Metrics Server).

Com a imagem e o contexto preparados, substitua `SEU_CONTEXTO`:

```powershell
kubectl --context SEU_CONTEXTO apply -f .\infra\autoscaling\kubernetes.json
.\scripts\test_autoscaling.ps1 -Context SEU_CONTEXTO
```

O [teste](./scripts/test_autoscaling.ps1) exige estabilização no mínimo,
cria o [Job de carga](./infra/autoscaling/load-test-job.json), observa aumento
de réplicas prontas e retorno ao mínimo, e remove o Job ao finalizar.
O Job faz 180 segundos de carga, concorrência 16, em um endpoint de
resultados clínicos, sem acionar LLM. O
[cliente de carga](./scripts/load_autoscaling.py) mede respostas e erros.

Esses arquivos são preparação e procedimento de validação, não evidência de
que um cluster já foi implantado. O manifesto não configura credenciais LLM,
ingress público, banco compartilhado nem cache distribuído. Não trate o HPA
como escalabilidade de escrita do módulo logístico.

## 10. Datasets, artefatos e atribuição

### Fontes de rotas

O dataset Olist já está em `src\db\models\dataset_rotasBR\`:

```text
olist_customers_dataset.csv
olist_geolocation_dataset.csv
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_products_dataset.csv
```

O inventário `inventory_data.csv` do Hospital Supply Chain também está em
`src\db\models\dataset_cnes\hospital_supply_chain\`. O script de download
permanece disponível para obter uma cópia atualizada. O CSV CNES não está
incluído: essa fonte é opcional e deve ser obtida pelo avaliador e colocada em
`src\db\models\dataset_cnes\cnes_estabelecimentos_sp.csv`.

Os dados de demanda, medicamentos, pesos e associação entre inventário e
destinos podem ser simulados; não os interprete como dados médicos reais.

**Fontes e atribuição:** o dataset Olist foi publicado por Olist no Kaggle sob
CC BY-NC-SA 4.0; uso não comercial, atribuição e compartilhamento de derivados
sob a mesma licença. O inventário Hospital Supply Chain, de Van Patangan, tem
licença MIT. Consulte as páginas de origem para os termos completos:
[Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) e
[Hospital Supply Chain](https://www.kaggle.com/datasets/vanpatangan/hospital-supply-chain).

### Localização dos resultados

| Diretório em `src\db\outputs` | Conteúdo |
|---|---|
| `breast-cancer-wisconsin-data` | Artefatos e export web do estudo de mama (`08-export-web`). |
| `polycystic-ovary-syndrome-pcos` | Artefatos e export web PCOS (`05-export-web`). |
| `phase2-genetic` | `optimization_results.json`, log e interpretações opcionais. |
| `phase2-logistics` | Planos e históricos separados por fonte. |
| `executed-notebooks` | Cópias executadas pelo launcher batch. |

Os dados de entrada ficam em `src\db\models`. `.venv`, credenciais de ambiente
em arquivos `.env` e outputs são ignorados pelo Git. Ignorar não revoga uma
credencial previamente exposta. Não versione arquivos sensíveis.

## 11. Testes e reprodução

Os testes em [tests](./tests/) usam `unittest` e incluem algoritmo genético,
logística, fontes, distribuição, contratos API, cache, transporte LLM,
notebooks, interfaces e launchers. Há testes de PowerShell que dependem do
Windows e testes HTTP com servidor local simulado.

Após preparar o ambiente, instale as dependências de desenvolvimento:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-dev.txt
```

[requirements-dev.txt](./requirements-dev.txt) inclui as dependências da
aplicação e o cliente HTTP usado pelo `fastapi.testclient`.

Exemplo de validação direcionada:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_startup_preflight.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_unified_launcher.py
```

Para a suíte completa:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

As dependências têm faixas de versão, não um lockfile; preserve o registro
das versões usadas ao comparar execuções em ambientes diferentes.

Mocks/servidores simulados não validam a conta Gemini real. Para isso, observe
o resultado da geração do preflight e faça uma consulta no painel.
A suíte local também não substitui o teste de carga no cluster. Não há
contagem fixa ou promessa de aprovação de testes nesta documentação.

## 12. Segurança, limitações e próximos passos

- Os datasets e resultados não validam uso clínico nem generalização externa.
- Os dashboards clínicos exibem resultados pré-computados.
- As buscas genéticas são heurísticas; não garantem ótimo global nem ganho
  universal no holdout. Semente ajuda a reproduzir uma execução, mas mudanças
  nas bibliotecas/dados podem alterar resultados.
- As rotas, demandas, prioridades e cargas podem ser simuladas; distâncias não
  consideram ruas, trânsito ou horários reais.
- Não envie dados pessoais ou clínicos ao Gemini ou a outro provedor.
- A aplicação local não implementa uma camada de autenticação de usuários
  nem autorização por perfil. Não a exponha publicamente sem controles
  adicionais, proteção de rede e revisão apropriada.
- O cache é local ao processo e a persistência de planos é local ao filesystem;
  operação distribuída exige coordenação e armazenamento adequados.
- O Free Tier depende de cotas/termos do provedor e não é garantia de
  disponibilidade. O uso do conteúdo pelo Google depende da conta/termos.
- Kubernetes/HPA é opcional; a execução local não possui escalabilidade
  automática.

Evoluções possíveis incluem validação clínica externa, avaliação com outras
divisões/coortes, distâncias por malha viária, dados logísticos operacionais,
armazenamento transacional e implantação monitorada com testes de carga reais.
