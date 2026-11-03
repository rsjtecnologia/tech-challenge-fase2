# Manual técnico — Tech Challenge

Projeto acadêmico FIAP com estudos de classificação clínica, otimização por
algoritmos genéticos e distribuição simulada de medicamentos e insumos.

> **Uso acadêmico:** os modelos não são dispositivos médicos e não substituem
> avaliação profissional. As rotas não representam uma operação real.

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

## 2. O que foi implementado

### Fase 1 — estudos clínicos

Pipelines de notebooks para câncer de mama (Wisconsin) e SOP/PCOS: exploração,
pré-processamento, treinamento, avaliação e exportação de resultados para os
dashboards. As páginas exibem dados pré-computados; não recebem pacientes nem
fazem diagnóstico online.

### Fase 2 — Projeto 1: otimização dos modelos

Um algoritmo genético otimiza separadamente os hiperparâmetros de cinco
classificadores nos dois estudos. São executadas três configurações genéticas,
com avaliação por validação cruzada estratificada no treino. O melhor
experimento é comparado ao baseline em um holdout separado. A função fitness é
`0,50 × recall + 0,30 × F1 + 0,20 × acurácia`.

A LLM pode explicar métricas agregadas em linguagem natural. Ela não recebe
registros individuais. A otimização funciona sem chave; a interpretação no
Gemini requer provedor configurado.

### Fase 2 — Projeto 2: otimização de rotas

Problema VRP com múltiplos veículos. O algoritmo genético representa a ordem
das entregas como uma permutação, usa seleção por torneio, crossover OX e
mutação por inversão de segmento. O decoder respeita capacidade, autonomia e
refrigeração; o fitness considera distância, atrasos ponderados por prioridade
e entregas não atendidas.

Fontes disponíveis:

- `demo`: cenário fictício, não requer dados adicionais.
- `olist_sp`: geografia e dados agregados do dataset Olist; não representa
  hospitais ou demanda médica.
- `cnes_sp`: estabelecimentos do CSV CNES local.
- `hybrid_sp`: geografia Olist ou CNES e perfil de inventário Kaggle, com
  associação, carga e demanda médicas simuladas.

O mapa é SVG local e as distâncias são Haversine em linha reta, não rotas por
ruas. O assistente Gemini gera instruções e relatórios; não altera o plano.

## 3. Dados opcionais de rotas

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

## 4. Estrutura e artefatos

- `notebooks\` — pipelines clínicos e notebook interativo de rotas.
- `src\optimization\` — experimentos genéticos e interpretação de métricas.
- `src\logistics\` — cenários, VRP, loaders, serviço e API de rotas.
- `src\web\` — FastAPI, páginas e arquivos estáticos.
- `src\db\models\` — datasets de entrada.
- `src\db\outputs\` — resultados gerados localmente, ignorados pelo Git.

Planos e históricos de rotas também são armazenados localmente em
`src\db\outputs\phase2-logistics\`, separados por fonte.

## 5. Limitações

- Os datasets e resultados não validam uso clínico nem generalização externa.
- Os dashboards clínicos exibem resultados pré-computados.
- As rotas, demandas, prioridades e cargas podem ser simuladas; distâncias não
  consideram ruas, trânsito ou horários reais.
- Não envie dados pessoais ou clínicos ao Gemini ou a outro provedor.
- Kubernetes/HPA é opcional; a execução local não possui escalabilidade
  automática.
