const state = {
  metrics: null,
  models: [],
  page: 1,
  pages: 1,
  size: 15,
  sort: "id",
  order: "asc",
  chart: null,
  chartsData: null,
  classChart: null,
  featureChart: null,
  importanceChart: null,
};

const $ = (selector) => document.querySelector(selector);
const percent = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const probability = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

async function request(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || "Não foi possível carregar os dados.");
  return body;
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 4300);
}

async function generateInterpretation(question = null) {
  const button = $("#interpretButton");
  const output = $("#interpretationOutput");
  $("#interpretationComparisonSection").hidden = false;
  const finishRequest = beginInterpretationRequest(question);
  button.textContent = "Gerando…";
  try {
    const result = await request("/api/interpretacao", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ estudo: "breast_cancer", pergunta: question }),
    });
    const data = result.interpretation;
    output.innerHTML = `
      <p class="interpretation-source">${result.source === "llm" ? "Gerado por provedor LLM" : "Interpretação local por regras (sem LLM)"}</p>
      ${result.verified_summary ? '<p class="llm-quality">Resumo calculado diretamente das métricas; explicação complementar pela LLM.</p>' : ""}
      ${result.cache?.hit ? '<p class="llm-quality">Resposta reutilizada do cache; não foi necessário consultar a LLM novamente.</p>' : ""}
      ${question ? `<p><strong>Pergunta:</strong> ${escapeHtml(question)}</p>` : ""}
      <p>${escapeHtml(data.summary)}</p>
      <h4>Principais observações</h4>
      <ul>${data.findings.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <h4>Limitações</h4>
      <ul>${data.limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <p><strong>${escapeHtml(data.disclaimer)}</strong></p>
      <p class="llm-quality">Rubrica de qualidade: ${result.quality_evaluation.score}/${result.quality_evaluation.maximum}</p>
      <p class="llm-quality">Verificação automática de forma e segurança; não comprova factualidade nem validade clínica.</p>
      ${result.quality_evaluation.passed ? "" : "<p><strong>Atenção: a resposta não passou em todos os critérios automáticos. Revise o texto e confira as afirmações nas métricas originais antes de utilizá-lo.</strong></p>"}
    `;
  } catch (error) {
    output.textContent = `Não foi possível gerar a interpretação: ${error.message}`;
    showToast(error.message);
  } finally {
    finishRequest();
    button.textContent = "Gerar interpretação";
  }
}

async function loadGeneticComparison() {
  const container = $("#geneticComparison");
  try {
    const result = await request("/api/otimizacao/breast_cancer");
    renderInterpretationComparison(result.models);
    container.innerHTML = Object.entries(result.models).map(([name, model]) => `
      <article class="model-card">
        <h4>${escapeHtml(name)}</h4>
        <span class="model-score">${percent(model.baseline_metrics.recall)} → ${percent(model.optimized_metrics.recall)}</span>
        <small>Recall baseline → GA · F1 ${percent(model.baseline_metrics.f1)} → ${percent(model.optimized_metrics.f1)}</small>
        <small>Acurácia ${percent(model.baseline_metrics.accuracy)} → ${percent(model.optimized_metrics.accuracy)}</small>
        <div class="model-tags"><span class="tag good">GA: ${escapeHtml(model.selected_experiment)}</span></div>
      </article>
    `).join("");
  } catch (error) {
    container.textContent = error.message;
    $("#interpretationComparison").textContent = error.message;
  }
}

function renderOverview(metrics) {
  const select = $("#overviewModelSelect");
  const modelName = metrics.por_modelo[select.value]
    ? select.value
    : metrics.melhor_modelo;
  const selectedModel = metrics.por_modelo[modelName];
  const isRecommended = modelName === metrics.melhor_modelo;
  $("#contractVersion").textContent = metrics.versao_contrato;
  $("#sampleCount").textContent = metrics.quantidade_amostras;
  $("#modelCount").textContent = metrics.quantidade_modelos;
  $("#featureCount").textContent = metrics.quantidade_atributos;
  $("#overviewModelTitle").textContent = isRecommended
    ? "Modelo recomendado"
    : "Modelo selecionado";
  $("#bestModel").textContent = modelName;
  $("#overviewModelBadge").textContent = isRecommended
    ? "Maior recall maligno"
    : `Recall maligno ${percent(selectedModel.recall_maligno)}`;
  $("#metricAccuracy").textContent = percent(selectedModel.acuracia);
  $("#metricRecall").textContent = percent(selectedModel.recall_maligno);
  $("#metricF1").textContent = percent(selectedModel.f1_maligno);
  $("#metricPrecision").textContent = percent(selectedModel.precisao_maligno);
}

function renderRanking(metrics) {
  const ranking = metrics.ranking.recall_maligno;
  $("#rankingList").innerHTML = ranking.map((item, index) => `
    <div class="rank-item">
      <span class="rank-number">${index + 1}</span>
      <div>
        <strong title="${escapeHtml(item.modelo)}">${escapeHtml(item.modelo)}</strong>
        <small>${index === 0 ? "Melhor sensibilidade clínica" : "Recall da classe positiva"}</small>
      </div>
      <span class="rank-value">${percent(item.valor)}</span>
    </div>
  `).join("");
}

function metricLabel(metric) {
  return {
    acuracia: "Acurácia",
    recall_maligno: "Recall",
    f1_maligno: "F1",
    precisao_maligno: "Precisão",
  }[metric] || metric;
}

function renderModelCards(metrics) {
  $("#modelCards").innerHTML = state.models.map((model) => {
    const data = metrics.por_modelo[model];
    const tags = [
      ...data.melhor_em.map((metric) => `<span class="tag good">Melhor ${metricLabel(metric)}</span>`),
      ...data.pior_em.map((metric) => `<span class="tag bad">Menor ${metricLabel(metric)}</span>`),
    ].join("");
    return `
      <article class="model-card ${model === metrics.melhor_modelo ? "best" : ""}">
        <h4>${escapeHtml(model)}</h4>
        <span class="model-score">${percent(data.recall_maligno)}</span>
        <small>recall maligno · ${data.fn} falso${data.fn === 1 ? "" : "s"} negativo${data.fn === 1 ? "" : "s"}</small>
        <div class="model-tags">${tags}</div>
      </article>
    `;
  }).join("");
}

function renderChart(metrics) {
  if (state.chart) state.chart.destroy();
  const context = $("#metricsChart");
  const datasets = [
    ["Acurácia", "acuracia", "#60a5fa"],
    ["Recall maligno", "recall_maligno", "#7dd3fc"],
    ["F1 maligno", "f1_maligno", "#bfdbfe"],
  ].map(([label, field, color]) => ({
    label,
    data: state.models.map((model) => Number(metrics.por_modelo[model][field]) * 100),
    backgroundColor: color,
    borderRadius: 4,
    borderSkipped: false,
    barPercentage: .7,
    categoryPercentage: .72,
    yAxisID: "y",
  }));
  datasets.push({
    label: "Falsos negativos",
    data: state.models.map((model) => Number(metrics.por_modelo[model].fn)),
    backgroundColor: "#ff7b91",
    borderRadius: 4,
    borderSkipped: false,
    barPercentage: .7,
    categoryPercentage: .72,
    yAxisID: "yErrors",
  });

  state.chart = new Chart(context, {
    type: "bar",
    data: { labels: state.models, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#dbeafe", boxWidth: 8, boxHeight: 8, usePointStyle: true, padding: 20, font: { size: 12 } },
        },
        tooltip: {
          backgroundColor: "#0b294b",
          borderColor: "rgba(255,255,255,.1)",
          borderWidth: 1,
          callbacks: {
            label: (context) => context.dataset.yAxisID === "yErrors"
              ? ` ${context.dataset.label}: ${context.raw}`
              : ` ${context.dataset.label}: ${context.raw.toFixed(2)}%`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "#c5d9f0", maxRotation: 0, callback(value) {
            const label = this.getLabelForValue(value);
            return label.length > 14 ? `${label.slice(0, 12)}…` : label;
          }, font: { size: 11 } },
          border: { color: "rgba(255,255,255,.06)" },
        },
        y: {
          min: 80,
          max: 100,
          grid: { color: "rgba(255,255,255,.045)" },
          ticks: { color: "#c5d9f0", callback: (value) => `${value}%`, font: { size: 11 } },
          border: { display: false },
        },
        yErrors: {
          position: "right",
          beginAtZero: true,
          suggestedMax: 5,
          grid: { drawOnChartArea: false },
          title: {
            display: true,
            text: "Falsos negativos",
            color: "#ff9aad",
            font: { size: 11, weight: "600" },
          },
          ticks: {
            color: "#ff9aad",
            precision: 0,
            stepSize: 1,
            font: { size: 11 },
          },
          border: { color: "rgba(255,123,145,.2)" },
        },
      },
    },
  });
}

function formatParameter(value) {
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null) return "null";
  return String(value);
}

function populateModelSelectors() {
  ["configModel", "confusionModel"].forEach((id) => {
    const select = $(`#${id}`);
    const previous = select.value;
    select.innerHTML = state.models
      .map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`)
      .join("");
    select.value = state.models.includes(previous) ? previous : state.metrics.melhor_modelo;
  });
}

function populateOverviewModelSelector(metrics) {
  const select = $("#overviewModelSelect");
  const previous = select.value;
  select.innerHTML = state.models
    .map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`)
    .join("");
  select.value = state.models.includes(previous) ? previous : metrics.melhor_modelo;
}

function renderModelConfig() {
  const name = $("#configModel").value;
  const configuration = state.metrics.por_modelo[name].configuracao;
  const preprocessing = state.metrics.preprocessamento;
  $("#modelConfig").innerHTML = `
    <div class="config-block">
      <span>Algoritmo</span>
      <strong>${escapeHtml(configuration.algoritmo)}</strong>
      <small>${escapeHtml(configuration.justificativa)}</small>
    </div>
    <div class="config-block">
      <span>Hiperparâmetros definidos</span>
      <div class="parameter-list">
        ${Object.entries(configuration.hiperparametros).map(([key, value]) =>
          `<span>${escapeHtml(key)} = ${escapeHtml(formatParameter(value))}</span>`
        ).join("")}
      </div>
    </div>
    <div class="config-block">
      <span>Dados e preparação</span>
      <strong>${configuration.amostras_treino} treino · ${configuration.amostras_avaliacao} avaliação</strong>
      <small>Imputação: ${escapeHtml(preprocessing.imputacao)} · Escala: ${escapeHtml(preprocessing.padronizacao)} · ${escapeHtml(preprocessing.ajuste)}</small>
    </div>
  `;
}

function renderInventory() {
  const inventory = state.chartsData.inventario;
  const splits = state.chartsData.splits;
  $("#rawCount").textContent = inventory.registros_brutos.toLocaleString("pt-BR");
  $("#usedCount").textContent = inventory.registros_usados.toLocaleString("pt-BR");
  $("#trainCount").textContent = splits.treino.total.toLocaleString("pt-BR");
  $("#testCount").textContent = splits.teste.total.toLocaleString("pt-BR");
}

function renderClassChart() {
  const split = state.chartsData.splits[$("#classSplit").value];
  if (state.classChart) state.classChart.destroy();
  state.classChart = new Chart($("#classChart"), {
    type: "doughnut",
    data: {
      labels: ["Benigno", "Maligno"],
      datasets: [{
        data: [split.benigno.n, split.maligno.n],
        backgroundColor: ["#55d6a5", "#ff7b91"],
        borderColor: ["#55d6a5", "#ff7b91"],
        borderWidth: 1,
        hoverOffset: 5,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "67%",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => ` ${context.label}: ${context.raw} (${percent(context.raw / split.total)})`,
          },
        },
      },
    },
  });
  $("#classSummary").innerHTML = `
    <div class="class-stat benigno"><span>Benigno</span><strong>${split.benigno.n} · ${percent(split.benigno.pct)}</strong></div>
    <div class="class-stat maligno"><span>Maligno</span><strong>${split.maligno.n} · ${percent(split.maligno.pct)}</strong></div>
    <div class="class-stat"><span>Total do conjunto</span><strong>${split.total}</strong></div>
  `;
}

function populateFeatureSelect() {
  $("#featureSelect").innerHTML = state.metrics.atributos
    .map((feature) => `<option value="${escapeHtml(feature)}">${escapeHtml(feature)}</option>`)
    .join("");
}

function renderFeatureChart() {
  const splitName = $("#featureSplit").value;
  const feature = $("#featureSelect").value;
  const histogram = state.chartsData.histogramas[splitName][feature];
  if (state.featureChart) state.featureChart.destroy();
  state.featureChart = new Chart($("#featureChart"), {
    type: "line",
    data: {
      labels: histogram.centros,
      datasets: [
        {
          label: "Benigno",
          data: histogram.benigno,
          borderColor: "#55d6a5",
          backgroundColor: "rgba(85,214,165,.13)",
          fill: true,
          tension: .25,
        },
        {
          label: "Maligno",
          data: histogram.maligno,
          borderColor: "#ff7b91",
          backgroundColor: "rgba(255,123,145,.11)",
          fill: true,
          tension: .25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "bottom", labels: { color: "#dbeafe", usePointStyle: true, boxWidth: 8 } },
      },
      scales: {
        x: {
          title: { display: true, text: feature, color: "#c5d9f0" },
          grid: { color: "rgba(255,255,255,.035)" },
          ticks: { color: "#c5d9f0", maxTicksLimit: 8 },
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: "Amostras", color: "#c5d9f0" },
          grid: { color: "rgba(255,255,255,.045)" },
          ticks: { color: "#c5d9f0", precision: 0 },
        },
      },
    },
  });
}

function lerpChannel(start, end, ratio) {
  return Math.round(start + (end - start) * ratio);
}

function coolwarmColor(value) {
  // Equivalente visual ao cmap="coolwarm" do Matplotlib/Seaborn (center=0).
  const stops = [
    { stop: -1, rgb: [59, 76, 192] },
    { stop: -0.5, rgb: [110, 166, 205] },
    { stop: 0, rgb: [247, 247, 247] },
    { stop: 0.5, rgb: [226, 102, 90] },
    { stop: 1, rgb: [180, 4, 38] },
  ];
  const clamped = Math.max(-1, Math.min(1, Number(value) || 0));
  let lower = stops[0];
  let upper = stops[stops.length - 1];
  for (let index = 0; index < stops.length - 1; index += 1) {
    if (clamped >= stops[index].stop && clamped <= stops[index + 1].stop) {
      lower = stops[index];
      upper = stops[index + 1];
      break;
    }
  }
  const span = upper.stop - lower.stop || 1;
  const ratio = (clamped - lower.stop) / span;
  const red = lerpChannel(lower.rgb[0], upper.rgb[0], ratio);
  const green = lerpChannel(lower.rgb[1], upper.rgb[1], ratio);
  const blue = lerpChannel(lower.rgb[2], upper.rgb[2], ratio);
  return `rgb(${red}, ${green}, ${blue})`;
}

function renderCorrelation() {
  const data = state.chartsData.correlacao;
  const labels = data.atributos;
  const heatmap = $("#correlationHeatmap");
  heatmap.style.gridTemplateColumns = `var(--heatmap-label) repeat(${labels.length}, var(--heatmap-cell))`;

  const body = labels.map((rowName, rowIndex) => `
    <span class="heatmap-row-label" title="${escapeHtml(rowName)}">${escapeHtml(rowName)}</span>
    ${labels.map((columnName, columnIndex) => {
      if (columnIndex > rowIndex) {
        return `<span class="heatmap-cell masked" aria-hidden="true"></span>`;
      }
      const value = Number(data.matriz[rowIndex][columnIndex]);
      return `
        <span
          class="heatmap-cell"
          style="background:${coolwarmColor(value)}"
          title="${escapeHtml(rowName)} × ${escapeHtml(columnName)}: ${value.toFixed(3)}"
        ></span>
      `;
    }).join("")}
  `).join("");

  const footer = `
    <span class="heatmap-corner"></span>
    ${labels.map((name) => `
      <span class="heatmap-col-label" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
    `).join("")}
  `;

  heatmap.innerHTML = body + footer;
}

function renderConfusionMatrix() {
  const model = $("#confusionModel").value;
  const data = state.metrics.por_modelo[model];
  $("#confusionMatrix").innerHTML = `
    <div class="matrix-axis corner">Real ↓<br>Previsto →</div>
    <div class="matrix-axis">Benigno</div>
    <div class="matrix-axis">Maligno</div>
    <div class="matrix-axis">Benigno</div>
    <div class="matrix-cell"><strong>${data.tn}</strong><small>verdadeiro negativo</small></div>
    <div class="matrix-cell error"><strong>${data.fp}</strong><small>falso positivo</small></div>
    <div class="matrix-axis">Maligno</div>
    <div class="matrix-cell error"><strong>${data.fn}</strong><small>falso negativo</small></div>
    <div class="matrix-cell"><strong>${data.tp}</strong><small>verdadeiro positivo</small></div>
  `;
}

function renderImportanceChart() {
  const data = state.chartsData.importancia_atributos;
  const limit = Number($("#importanceLimit").value);
  const items = data.itens.slice(0, limit).reverse();
  const chartWrap = $("#importanceChart").closest(".importance-chart");
  chartWrap.style.height = `${Math.max(320, items.length * 28 + 48)}px`;
  $("#importanceDescription").textContent = `${data.modelo} · ${data.metodo}`;
  if (state.importanceChart) state.importanceChart.destroy();
  state.importanceChart = new Chart($("#importanceChart"), {
    type: "bar",
    data: {
      labels: items.map((item) => item.atributo),
      datasets: [{
        label: "Importância",
        data: items.map((item) => item.valor),
        backgroundColor: "#60a5fa",
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 8, bottom: 8, left: 4, right: 8 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (points) => points[0]?.label || "",
            label: (context) => ` Importância: ${Number(context.raw).toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          grid: { color: "rgba(255,255,255,.045)" },
          ticks: { color: "#c5d9f0" },
        },
        y: {
          grid: { display: false },
          ticks: {
            color: "#e2edfa",
            font: { size: 11 },
            autoSkip: false,
            crossAlign: "far",
          },
        },
      },
    },
  });
}

function renderChartsSection() {
  renderInventory();
  populateFeatureSelect();
  renderClassChart();
  renderFeatureChart();
  renderCorrelation();
  renderConfusionMatrix();
  renderImportanceChart();
}

function populateModelFilter() {
  const select = $("#modelFilter");
  const previous = select.value;
  select.innerHTML = `<option value="">Todos os modelos</option>${state.models
    .map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`)
    .join("")}`;
  select.value = previous;
}

function sampleQuery() {
  const params = new URLSearchParams({ page: state.page, size: state.size });
  const diagnosis = $("#diagnosisFilter").value;
  const model = $("#modelFilter").value;
  const result = $("#resultFilter").value;
  const consensus = $("#consensusFilter").value;
  if (diagnosis) params.set("diagnostico", diagnosis);
  if (model) params.set("modelo", model);
  if (model && result) params.set("acerto", result);
  if (consensus) params.set("consenso", consensus);
  params.set("sort", state.sort);
  params.set("order", state.order);
  return params;
}

function sortIcon(column) {
  if (state.sort !== column) return "↕";
  return state.order === "asc" ? "↑" : "↓";
}

function renderTableHeader(models) {
  $("#samplesHead").innerHTML = `
    <tr>
      <th class="col-id"><button class="sort-button ${state.sort === "id" ? "active" : ""}" data-sort="id">Amostra <i>${sortIcon("id")}</i></button></th>
      <th class="col-diagnosis"><button class="sort-button ${state.sort === "diagnostico_real" ? "active" : ""}" data-sort="diagnostico_real"><span class="label-full">Diagnóstico real</span><span class="label-short">Diagnóstico</span> <i>${sortIcon("diagnostico_real")}</i></button></th>
      ${models.map((model) => {
        const column = `confianca_${model}`;
        return `<th class="col-model"><button class="sort-button ${state.sort === column ? "active" : ""}" data-sort="${escapeHtml(column)}">${escapeHtml(model)} <i>${sortIcon(column)}</i></button></th>`;
      }).join("")}
      <th class="consensus-cell"><button class="sort-button ${state.sort === "n_modelos_acertaram" ? "active" : ""}" data-sort="n_modelos_acertaram"><span class="label-full">Concordância</span><span class="label-short">Conc.</span> <i>${sortIcon("n_modelos_acertaram")}</i></button></th>
      <th class="amplitude-cell"><button class="sort-button ${state.sort === "amplitude_probabilidade" ? "active" : ""}" data-sort="amplitude_probabilidade"><span class="label-full">Amplitude</span><span class="label-short">Ampl.</span> <i>${sortIcon("amplitude_probabilidade")}</i></button></th>
    </tr>
  `;
  $("#samplesHead").querySelectorAll("[data-sort]").forEach((button) => {
    button.addEventListener("click", () => {
      const column = button.dataset.sort;
      state.order = state.sort === column && state.order === "asc" ? "desc" : "asc";
      state.sort = column;
      state.page = 1;
      loadSamples();
    });
  });
}

function renderSamples(items, models) {
  const body = $("#samplesBody");
  if (!items.length) {
    body.innerHTML = `<tr><td colspan="${models.length + 4}"><div class="empty-state">Nenhuma amostra corresponde aos filtros.</div></td></tr>`;
    return;
  }
  body.innerHTML = items.map((sample) => `
    <tr>
      <td class="col-id"><button class="sample-id" data-sample-id="${sample.id}" aria-label="Abrir detalhes da amostra ${sample.id}">#${String(sample.id).padStart(3, "0")}</button></td>
      <td class="col-diagnosis"><span class="diagnosis ${sample.diagnostico_real.toLowerCase()}">${sample.diagnostico_real}</span></td>
      ${models.map((model) => {
        const value = Number(sample[`confianca_${model}`]);
        const hit = sample[`acerto_${model}`];
        const prediction = Number(sample[`pred_${model}`]) === 1 ? "Maligno" : "Benigno";
        return `
          <td class="col-model prediction-cell ${hit ? "prediction-hit" : "prediction-miss"}">
            <div class="prediction-top">
              <span class="prediction-value">
                <span class="diagnosis prediction-diagnosis ${prediction.toLowerCase()}">${prediction}</span>
                <span class="prediction-probability">${probability(value)}</span>
              </span>
            </div>
            <div class="probability-track"><i style="width:${value * 100}%"></i></div>
          </td>
        `;
      }).join("")}
      <td class="consensus-cell ${sample.consenso ? "consensus" : "divergence"}" title="${sample.n_modelos_acertaram}/${models.length} · ${sample.consenso ? "consenso" : "divergência"}">
        <span class="consensus-score">${sample.n_modelos_acertaram}/${models.length}</span>
        <span class="consensus-kind">${sample.consenso ? "consenso" : "divergência"}</span>
      </td>
      <td class="amplitude-cell">
        <button
          type="button"
          class="amplitude-trigger"
          aria-describedby="ampTooltip"
          data-amplitude="${escapeHtml(JSON.stringify(models.map((model) => ({
            modelo: model,
            prob: Number(sample[`prob_${model}`]),
          }))))}"
        >${probability(sample.amplitude_probabilidade)}</button>
      </td>
    </tr>
  `).join("");

  body.querySelectorAll("[data-sample-id]").forEach((button) => {
    button.addEventListener("click", () => openDrawer(button.dataset.sampleId));
  });
  body.querySelectorAll(".amplitude-trigger").forEach((button) => {
    button.addEventListener("pointerenter", () => showAmplitudeTooltip(button));
    button.addEventListener("focus", () => showAmplitudeTooltip(button));
    button.addEventListener("pointerleave", hideAmplitudeTooltip);
    button.addEventListener("blur", hideAmplitudeTooltip);
  });
}

function showAmplitudeTooltip(trigger) {
  const tooltip = $("#ampTooltip");
  const body = $("#ampTooltipBody");
  let entries = [];
  try {
    entries = JSON.parse(trigger.dataset.amplitude || "[]");
  } catch {
    hideAmplitudeTooltip();
    return;
  }
  if (!entries.length) {
    hideAmplitudeTooltip();
    return;
  }

  const probs = entries.map((entry) => entry.prob);
  const maxProb = Math.max(...probs);
  const minProb = Math.min(...probs);
  const amplitude = maxProb - minProb;

  body.innerHTML = `
    ${entries.map((entry) => {
      const isMax = entry.prob === maxProb && amplitude > 0;
      const isMin = entry.prob === minProb && amplitude > 0;
      const flags = [isMax ? "is-max" : "", isMin ? "is-min" : ""].filter(Boolean).join(" ");
      return `
        <div class="amp-tooltip-row ${flags}">
          <strong title="${escapeHtml(entry.modelo)}">${escapeHtml(entry.modelo)}</strong>
          <span>${probability(entry.prob)}</span>
          <div class="amp-bar"><i style="width:${Math.max(0, Math.min(1, entry.prob)) * 100}%"></i></div>
        </div>
      `;
    }).join("")}
    <div class="amp-tooltip-footer">
      Amplitude = máx <b>${probability(maxProb)}</b> − mín <b>${probability(minProb)}</b>
    </div>
  `;

  tooltip.hidden = false;
  requestAnimationFrame(() => {
    const rect = trigger.getBoundingClientRect();
    const tipRect = tooltip.getBoundingClientRect();
    const gap = 10;
    let left = rect.left + rect.width / 2 - tipRect.width / 2;
    left = Math.max(12, Math.min(left, window.innerWidth - tipRect.width - 12));
    let top = rect.top - tipRect.height - gap;
    if (top < 12) top = rect.bottom + gap;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
    tooltip.classList.add("visible");
  });
}

function hideAmplitudeTooltip() {
  const tooltip = $("#ampTooltip");
  tooltip.classList.remove("visible");
  window.clearTimeout(hideAmplitudeTooltip.timer);
  hideAmplitudeTooltip.timer = window.setTimeout(() => {
    if (!tooltip.classList.contains("visible")) tooltip.hidden = true;
  }, 180);
}

async function loadSamples() {
  $("#samplesBody").innerHTML = `<tr><td colspan="10"><div class="loading-state"><i></i>Atualizando amostras…</div></td></tr>`;
  try {
    const data = await request(`/api/amostras?${sampleQuery()}`);
    state.pages = data.pagination.pages;
    renderTableHeader(data.modelos);
    renderSamples(data.items, data.modelos);
    $("#filteredCount").textContent = data.pagination.total;
    $("#pageInfo").textContent = data.pagination.all
      ? `Todos os ${data.pagination.total} registros`
      : `Página ${data.pagination.page} de ${data.pagination.pages}`;
    $("#prevPage").disabled = state.page <= 1;
    $("#nextPage").disabled = data.pagination.all || state.page >= state.pages;
  } catch (error) {
    $("#samplesBody").innerHTML = `<tr><td colspan="10"><div class="empty-state">${escapeHtml(error.message)}</div></td></tr>`;
    showToast(error.message);
  }
}

function formatFeature(value) {
  if (typeof value !== "number") return escapeHtml(value);
  return value.toLocaleString("pt-BR", { maximumFractionDigits: 5 });
}

async function openDrawer(sampleId) {
  const drawer = $("#sampleDrawer");
  const backdrop = $("#drawerBackdrop");
  $("#drawerId").textContent = `#${String(sampleId).padStart(3, "0")}`;
  $("#drawerContent").innerHTML = `<div class="loading-state"><i></i>Carregando detalhes…</div>`;
  drawer.classList.add("open");
  backdrop.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
  $("#closeDrawer").focus();

  try {
    const sample = await request(`/api/amostras/${sampleId}`);
    const sortedPredictions = [...sample.predicoes].sort((first, second) =>
      first.modelo.localeCompare(second.modelo, "pt-BR", { sensitivity: "base" })
    );
    $("#drawerContent").innerHTML = `
      <div class="detail-summary">
        <div class="detail-stat"><span>Diagnóstico real</span><strong><span class="diagnosis ${sample.diagnostico_real.toLowerCase()}">${sample.diagnostico_real}</span></strong></div>
        <div class="detail-stat"><span>Modelos corretos</span><strong>${sample.n_modelos_acertaram}/${state.models.length}</strong></div>
        <div class="detail-stat"><span>Amplitude</span><strong>${probability(sample.amplitude_probabilidade)}</strong></div>
      </div>
      <h3 class="detail-title">Predição de cada modelo</h3>
      <div class="detail-models">
        ${sortedPredictions.map((prediction) => {
          const predictedDiagnosis = prediction.rotulo_previsto === 1 ? "Maligno" : "Benigno";
          return `
          <div class="detail-model ${prediction.acerto ? "model-hit" : "model-miss"}">
            <strong>${escapeHtml(prediction.modelo)}</strong>
            <div class="detail-model-result">
              <span class="diagnosis ${predictedDiagnosis.toLowerCase()}">${predictedDiagnosis}</span>
              <span class="detail-confidence">${probability(prediction.confianca)}</span>
            </div>
            <small>Confiança na classe prevista</small>
            <span class="model-result-status">${prediction.acerto ? "✓ Resultado correto" : "× Resultado incorreto"}</span>
          </div>
        `;
        }).join("")}
      </div>
      <h3 class="detail-title">Atributos da amostra</h3>
      <div class="feature-grid">
        ${Object.entries(sample.atributos).map(([name, value]) => `
          <div class="feature-item"><span title="${escapeHtml(name)}">${escapeHtml(name)}</span><strong>${formatFeature(value)}</strong></div>
        `).join("")}
      </div>
    `;
  } catch (error) {
    $("#drawerContent").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

function closeDrawer() {
  $("#sampleDrawer").classList.remove("open");
  $("#drawerBackdrop").classList.remove("open");
  $("#sampleDrawer").setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
}

async function loadDashboard() {
  try {
    const [metrics, chartsData] = await Promise.all([
      request("/api/metricas"),
      request("/api/graficos"),
    ]);
    state.metrics = metrics;
    state.chartsData = chartsData;
    state.models = Object.keys(metrics.por_modelo).sort((first, second) =>
      first.localeCompare(second, "pt-BR", { sensitivity: "base" })
    );
    renderOverview(metrics);
    renderRanking(metrics);
    renderModelCards(metrics);
    renderChart(metrics);
    populateOverviewModelSelector(metrics);
    populateModelSelectors();
    renderModelConfig();
    renderChartsSection();
    populateModelFilter();
    await loadGeneticComparison();
    await loadSamples();
  } catch (error) {
    showToast(error.message);
    $("#bestModel").textContent = "Execute o notebook 08";
  }
}

function bindEvents() {
  $("#interpretButton").addEventListener("click", () => generateInterpretation());
  initializeInterpretationQuestions(generateInterpretation);
  $("#configModel").addEventListener("change", renderModelConfig);
  $("#overviewModelSelect").addEventListener("change", () => renderOverview(state.metrics));
  $("#classSplit").addEventListener("change", renderClassChart);
  $("#featureSelect").addEventListener("change", renderFeatureChart);
  $("#featureSplit").addEventListener("change", renderFeatureChart);
  $("#confusionModel").addEventListener("change", renderConfusionMatrix);
  $("#importanceLimit").addEventListener("change", renderImportanceChart);
  ["diagnosisFilter", "modelFilter", "resultFilter", "consensusFilter"].forEach((id) => {
    $(`#${id}`).addEventListener("change", () => {
      state.page = 1;
      const hasModel = Boolean($("#modelFilter").value);
      $("#resultFilter").disabled = !hasModel;
      if (!hasModel) $("#resultFilter").value = "";
      loadSamples();
    });
  });
  $("#resultFilter").disabled = true;
  $("#pageSize").addEventListener("change", () => {
    state.size = Number($("#pageSize").value);
    state.page = 1;
    loadSamples();
  });
  $("#clearFilters").addEventListener("click", () => {
    ["diagnosisFilter", "modelFilter", "resultFilter", "consensusFilter"].forEach((id) => { $(`#${id}`).value = ""; });
    $("#resultFilter").disabled = true;
    state.page = 1;
    state.sort = "id";
    state.order = "asc";
    loadSamples();
  });
  $("#prevPage").addEventListener("click", () => { if (state.page > 1) { state.page -= 1; loadSamples(); } });
  $("#nextPage").addEventListener("click", () => { if (state.page < state.pages) { state.page += 1; loadSamples(); } });
  $("#closeDrawer").addEventListener("click", closeDrawer);
  $("#drawerBackdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDrawer(); });
  $("#menuButton").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  document.querySelectorAll(".nav-link").forEach((link) => link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active"));
    link.classList.add("active");
    $("#sidebar").classList.remove("open");
  }));
  $("#refreshButton").addEventListener("click", async () => {
    try {
      await request("/api/cache/recarregar", { method: "POST" });
      await loadDashboard();
      showToast("Dados recarregados.");
    } catch (error) {
      showToast(error.message);
    }
  });
  window.addEventListener("scroll", hideAmplitudeTooltip, true);
  window.addEventListener("resize", hideAmplitudeTooltip);
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadDashboard();
});
