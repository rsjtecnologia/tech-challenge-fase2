const API = "/api/pcos";

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
      body: JSON.stringify({ estudo: "pcos", pergunta: question }),
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
    const result = await request("/api/otimizacao/pcos");
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

function diagnosisClass(label) {
  return label === "Com PCOS" ? "com-pcos" : "sem-pcos";
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
    ? "Maior recall PCOS"
    : `Recall PCOS ${percent(selectedModel.recall_pcos)}`;
  $("#metricAccuracy").textContent = percent(selectedModel.acuracia);
  $("#metricRecall").textContent = percent(selectedModel.recall_pcos);
  $("#metricF1").textContent = percent(selectedModel.f1_pcos);
  $("#metricPrecision").textContent = percent(selectedModel.precisao_pcos);
}

function renderRanking(metrics) {
  const ranking = metrics.ranking.recall_pcos;
  $("#rankingList").innerHTML = ranking.map((item, index) => `
    <div class="rank-item">
      <span class="rank-number">${index + 1}</span>
      <div>
        <strong>${escapeHtml(item.modelo)}</strong>
        <small>${index === 0 ? "Melhor sensibilidade" : "Recall da classe PCOS"}</small>
      </div>
      <span class="rank-value">${percent(item.valor)}</span>
    </div>
  `).join("");
}

function renderModelCards(metrics) {
  $("#modelCards").innerHTML = state.models.map((model) => {
    const data = metrics.por_modelo[model];
    return `
      <article class="model-card ${model === metrics.melhor_modelo ? "best" : ""}">
        <h4>${escapeHtml(model)}</h4>
        <span class="model-score">${percent(data.recall_pcos)}</span>
        <small>recall PCOS · ${data.fn} falso${data.fn === 1 ? "" : "s"} negativo${data.fn === 1 ? "" : "s"}</small>
      </article>
    `;
  }).join("");
}

function consolidatedModelFields(data) {
  const probPcos = data.media_prob_pcos ?? data.acuracia;
  const probSem = data.media_prob_sem_pcos ?? (1 - probPcos);
  const conf = data.media_confianca ?? data.acuracia;
  const acertos = data.acertos ?? Math.round(data.acuracia * (data.total_avaliacao || 0));
  const total = data.total_avaliacao ?? (data.tn + data.fp + data.fn + data.tp);
  return { probPcos, probSem, conf, acertos, total };
}

function renderModelSummaryTable(metrics) {
  const body = $("#modelSummaryBody");
  if (!body) return;

  body.innerHTML = state.models.map((model) => {
    const data = metrics.por_modelo[model];
    const { probPcos, probSem, conf, acertos, total } = consolidatedModelFields(data);
    const isBest = model === metrics.melhor_modelo;
    return `
      <tr class="${isBest ? "best-model-row" : ""}">
        <td><strong>${escapeHtml(model)}</strong>${isBest ? ' <span class="subtle-pill">melhor</span>' : ""}</td>
        <td class="prob-cell">${percent(probPcos)}</td>
        <td class="prob-cell">${percent(probSem)}</td>
        <td class="prob-cell">${percent(conf)}</td>
        <td class="diag-summary">
          ${acertos}/${total} acertos
          <small>${percent(data.acuracia)} de acurácia</small>
        </td>
        <td class="prob-cell">${percent(data.recall_pcos)}</td>
        <td class="prob-cell">${percent(data.f1_pcos)}</td>
      </tr>
    `;
  }).join("");
}

function renderChart(metrics) {
  if (state.chart) state.chart.destroy();
  const values = state.models.flatMap((model) => [
    metrics.por_modelo[model].acuracia,
    metrics.por_modelo[model].recall_pcos,
    metrics.por_modelo[model].f1_pcos,
  ]);
  const minPct = Math.floor(Math.min(...values) * 100) - 5;
  const maxPct = Math.ceil(Math.max(...values) * 100) + 5;

  state.chart = new Chart($("#metricsChart"), {
    type: "bar",
    data: {
      labels: state.models,
      datasets: [
        {
          label: "Acurácia",
          data: state.models.map((m) => metrics.por_modelo[m].acuracia * 100),
          backgroundColor: "#60a5fa",
          borderRadius: 4,
        },
        {
          label: "Recall PCOS",
          data: state.models.map((m) => metrics.por_modelo[m].recall_pcos * 100),
          backgroundColor: "#7dd3fc",
          borderRadius: 4,
        },
        {
          label: "F1 PCOS",
          data: state.models.map((m) => metrics.por_modelo[m].f1_pcos * 100),
          backgroundColor: "#bfdbfe",
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { color: "#dbeafe", usePointStyle: true, boxWidth: 8 } },
      },
      scales: {
        x: { ticks: { color: "#c5d9f0" }, grid: { display: false } },
        y: {
          min: Math.max(0, minPct),
          max: Math.min(100, maxPct),
          ticks: { color: "#c5d9f0", callback: (v) => `${v}%` },
          grid: { color: "rgba(255,255,255,.045)" },
        },
      },
    },
  });
}

function populateModelSelectors() {
  const select = $("#confusionModel");
  select.innerHTML = state.models
    .map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`)
    .join("");
  select.value = state.metrics.melhor_modelo;
}

function populateOverviewModelSelector(metrics) {
  const select = $("#overviewModelSelect");
  const previous = select.value;
  select.innerHTML = state.models
    .map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`)
    .join("");
  select.value = state.models.includes(previous) ? previous : metrics.melhor_modelo;
}

function renderEdaHighlights() {
  const h = state.chartsData.destaques_eda || {};
  $("#edaHighlights").innerHTML = `
    <article><span>AMH médio (sem PCOS)</span><strong>${h.amh_media_sem_pcos ?? "—"} ng/mL</strong></article>
    <article><span>AMH médio (com PCOS)</span><strong>${h.amh_media_com_pcos ?? "—"} ng/mL</strong></article>
    <article><span>Censura beta-HCG I (1.99)</span><strong>${h.censura_beta_i ?? "—"}</strong></article>
    <article><span>Censura beta-HCG II (1.99)</span><strong>${h.censura_beta_ii ?? "—"}</strong></article>
  `;
}

function renderInventory() {
  const inventory = state.chartsData.inventario;
  const splits = state.chartsData.splits;
  $("#rawCount").textContent = inventory.registros_brutos;
  $("#usedCount").textContent = inventory.registros_usados;
  $("#trainCount").textContent = splits.treino.total;
  $("#testCount").textContent = splits.teste.total;
}

function renderClassChart() {
  const split = state.chartsData.splits[$("#classSplit").value];
  if (state.classChart) state.classChart.destroy();
  state.classChart = new Chart($("#classChart"), {
    type: "doughnut",
    data: {
      labels: ["Sem PCOS", "Com PCOS"],
      datasets: [{
        data: [split.sem_pcos.n, split.com_pcos.n],
        backgroundColor: ["#55d6a5", "#f8bd68"],
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "67%",
      plugins: { legend: { display: false } },
    },
  });
  $("#classSummary").innerHTML = `
    <div class="class-stat benigno"><span>Sem PCOS</span><strong>${split.sem_pcos.n} · ${percent(split.sem_pcos.pct)}</strong></div>
    <div class="class-stat maligno"><span>Com PCOS</span><strong>${split.com_pcos.n} · ${percent(split.com_pcos.pct)}</strong></div>
    <div class="class-stat"><span>Total</span><strong>${split.total}</strong></div>
  `;
}

function populateFeatureSelect() {
  $("#featureSelect").innerHTML = state.metrics.atributos
    .map((f) => `<option value="${escapeHtml(f)}">${escapeHtml(f)}</option>`)
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
        { label: "Sem PCOS", data: histogram.sem_pcos, borderColor: "#55d6a5", backgroundColor: "rgba(85,214,165,.13)", fill: true, tension: .25 },
        { label: "Com PCOS", data: histogram.com_pcos, borderColor: "#f8bd68", backgroundColor: "rgba(248,189,104,.11)", fill: true, tension: .25 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { color: "#dbeafe", usePointStyle: true, boxWidth: 8 } } },
      scales: {
        x: { title: { display: true, text: feature, color: "#c5d9f0" }, ticks: { color: "#c5d9f0" } },
        y: { beginAtZero: true, ticks: { precision: 0, color: "#c5d9f0" } },
      },
    },
  });
}

function coolwarmColor(value) {
  const stops = [
    { stop: -1, rgb: [59, 76, 192] },
    { stop: 0, rgb: [247, 247, 247] },
    { stop: 1, rgb: [180, 4, 38] },
  ];
  const clamped = Math.max(-1, Math.min(1, Number(value) || 0));
  let lower = stops[0];
  let upper = stops[2];
  for (let i = 0; i < stops.length - 1; i += 1) {
    if (clamped >= stops[i].stop && clamped <= stops[i + 1].stop) {
      lower = stops[i];
      upper = stops[i + 1];
      break;
    }
  }
  const ratio = (clamped - lower.stop) / (upper.stop - lower.stop || 1);
  const rgb = lower.rgb.map((start, i) => Math.round(start + (upper.rgb[i] - start) * ratio));
  return `rgb(${rgb.join(",")})`;
}

function renderCorrelation() {
  const data = state.chartsData.correlacao;
  const labels = data.atributos;
  const heatmap = $("#correlationHeatmap");
  heatmap.style.gridTemplateColumns = `var(--heatmap-label) repeat(${labels.length}, var(--heatmap-cell))`;
  heatmap.innerHTML = labels.map((rowName, rowIndex) => `
    <span class="heatmap-row-label" title="${escapeHtml(rowName)}">${escapeHtml(rowName)}</span>
    ${labels.map((columnName, columnIndex) => {
      if (columnIndex > rowIndex) return `<span class="heatmap-cell masked"></span>`;
      const value = Number(data.matriz[rowIndex][columnIndex]);
      return `<span class="heatmap-cell" style="background:${coolwarmColor(value)}" title="${escapeHtml(rowName)} × ${escapeHtml(columnName)}: ${value.toFixed(3)}"></span>`;
    }).join("")}
  `).join("") + `<span class="heatmap-corner"></span>${labels.map((name) => `<span class="heatmap-col-label" title="${escapeHtml(name)}">${escapeHtml(name)}</span>`).join("")}`;
}

function renderConfusionMatrix() {
  const model = $("#confusionModel").value;
  const data = state.metrics.por_modelo[model];
  $("#confusionMatrix").innerHTML = `
    <div class="matrix-axis corner">Real ↓<br>Previsto →</div>
    <div class="matrix-axis">Sem PCOS</div>
    <div class="matrix-axis">Com PCOS</div>
    <div class="matrix-axis">Sem PCOS</div>
    <div class="matrix-cell"><strong>${data.tn}</strong><small>verdadeiro negativo</small></div>
    <div class="matrix-cell error"><strong>${data.fp}</strong><small>falso positivo</small></div>
    <div class="matrix-axis">Com PCOS</div>
    <div class="matrix-cell error"><strong>${data.fn}</strong><small>falso negativo</small></div>
    <div class="matrix-cell"><strong>${data.tp}</strong><small>verdadeiro positivo</small></div>
  `;
}

function renderImportanceChart() {
  const data = state.chartsData.importancia_atributos;
  const items = [...data.itens].reverse();
  const chartWrap = $("#importanceChart").closest(".importance-chart");
  chartWrap.style.height = `${Math.max(280, items.length * 32 + 48)}px`;
  if (state.importanceChart) state.importanceChart.destroy();
  state.importanceChart = new Chart($("#importanceChart"), {
    type: "bar",
    data: {
      labels: items.map((item) => item.atributo),
      datasets: [{ label: "Importância", data: items.map((item) => item.valor), backgroundColor: "#7dd3fc", borderRadius: 4 }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, ticks: { color: "#c5d9f0" } },
        y: { ticks: { color: "#e2edfa", autoSkip: false } },
      },
    },
  });
}

function populateModelFilter() {
  const select = $("#modelFilter");
  select.innerHTML = `<option value="">Todos</option>${state.models.map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("")}`;
}

function sampleQuery() {
  const params = new URLSearchParams({ page: state.page, size: state.size, sort: state.sort, order: state.order });
  const diagnosis = $("#diagnosisFilter").value;
  const model = $("#modelFilter").value;
  const result = $("#resultFilter").value;
  if (diagnosis) params.set("diagnostico", diagnosis);
  if (model) params.set("modelo", model);
  if (model && result) params.set("acerto", result);
  return params;
}

function renderTableHeader(models) {
  $("#samplesHead").innerHTML = `
    <tr>
      <th class="col-id">Amostra</th>
      <th class="col-diagnosis">Diagnóstico</th>
      ${models.map((model) => `<th class="col-model">${escapeHtml(model)}</th>`).join("")}
      <th>Acertos</th>
    </tr>
  `;
}

function renderSamples(items, models) {
  const body = $("#samplesBody");
  const colspan = models.length + 3;
  if (!items.length) {
    body.innerHTML = `<tr><td colspan="${colspan}"><div class="empty-state">Nenhuma amostra encontrada.</div></td></tr>`;
    return;
  }
  body.innerHTML = items.map((sample) => `
    <tr>
      <td class="col-id"><button class="sample-id" data-sample-id="${sample.id}">#${String(sample.id).padStart(3, "0")}</button></td>
      <td class="col-diagnosis"><span class="diagnosis ${diagnosisClass(sample.diagnostico_real)}">${sample.diagnostico_real}</span></td>
      ${models.map((model) => {
        const pred = Number(sample[`pred_${model}`]) === 1 ? "Com PCOS" : "Sem PCOS";
        const hit = sample[`acerto_${model}`];
        const conf = Number(sample[`confianca_${model}`]);
        return `<td class="col-model prediction-cell ${hit ? "prediction-hit" : "prediction-miss"}">
          <span class="diagnosis ${diagnosisClass(pred)}">${pred}</span>
          <small>${percent(conf)}</small>
        </td>`;
      }).join("")}
      <td>${sample.n_modelos_acertaram}/${models.length}</td>
    </tr>
  `).join("");
  body.querySelectorAll("[data-sample-id]").forEach((btn) => {
    btn.addEventListener("click", () => openDrawer(btn.dataset.sampleId));
  });
}

async function loadSamples() {
  const colspan = state.models.length ? state.models.length + 3 : 8;
  $("#samplesBody").innerHTML = `<tr><td colspan="${colspan}"><div class="loading-state"><i></i>Carregando…</div></td></tr>`;
  try {
    const data = await request(`${API}/amostras?${sampleQuery()}`);
    state.pages = data.pagination.pages;
    renderTableHeader(data.modelos);
    renderSamples(data.items, data.modelos);
    $("#filteredCount").textContent = data.pagination.total;
    $("#pageInfo").textContent = data.pagination.all
      ? `Todos (${data.pagination.total})`
      : `Página ${data.pagination.page} de ${data.pagination.pages}`;
    $("#prevPage").disabled = state.page <= 1;
    $("#nextPage").disabled = data.pagination.all || state.page >= state.pages;
  } catch (error) {
    showToast(error.message);
  }
}

async function openDrawer(sampleId) {
  $("#drawerId").textContent = `#${String(sampleId).padStart(3, "0")}`;
  $("#drawerContent").innerHTML = `<div class="loading-state"><i></i>Carregando…</div>`;
  $("#sampleDrawer").classList.add("open");
  $("#drawerBackdrop").classList.add("open");
  try {
    const sample = await request(`${API}/amostras/${sampleId}`);
    $("#drawerContent").innerHTML = `
      <div class="detail-summary">
        <div class="detail-stat"><span>Real</span><strong><span class="diagnosis ${diagnosisClass(sample.diagnostico_real)}">${sample.diagnostico_real}</span></strong></div>
        <div class="detail-stat"><span>Acertos</span><strong>${sample.n_modelos_acertaram}/${state.models.length}</strong></div>
      </div>
      <h3 class="detail-title">Predições</h3>
      <div class="detail-models">
        ${sample.predicoes.map((p) => {
          const pred = p.rotulo_previsto === 1 ? "Com PCOS" : "Sem PCOS";
          return `<div class="detail-model ${p.acerto ? "model-hit" : "model-miss"}">
            <strong>${escapeHtml(p.modelo)}</strong>
            <span class="diagnosis ${diagnosisClass(pred)}">${pred}</span>
            <small>${percent(p.confianca)}</small>
          </div>`;
        }).join("")}
      </div>
      <h3 class="detail-title">Biomarcadores</h3>
      <div class="feature-grid">
        ${Object.entries(sample.atributos).map(([name, value]) => `
          <div class="feature-item"><span>${escapeHtml(name)}</span><strong>${Number(value).toLocaleString("pt-BR", { maximumFractionDigits: 4 })}</strong></div>
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
}

async function loadDashboard() {
  try {
    const [metrics, chartsData] = await Promise.all([
      request(`${API}/metricas`),
      request(`${API}/graficos`),
    ]);
    state.metrics = metrics;
    state.chartsData = chartsData;
    state.models = Object.keys(metrics.por_modelo).sort((a, b) => a.localeCompare(b, "pt-BR"));
    populateOverviewModelSelector(metrics);
    renderOverview(metrics);
    renderRanking(metrics);
    renderModelCards(metrics);
    renderModelSummaryTable(metrics);
    renderChart(metrics);
    populateModelSelectors();
    renderEdaHighlights();
    renderInventory();
    populateFeatureSelect();
    renderClassChart();
    renderFeatureChart();
    renderCorrelation();
    renderConfusionMatrix();
    renderImportanceChart();
    populateModelFilter();
    await loadGeneticComparison();
    await loadSamples();
  } catch (error) {
    showToast(error.message);
    $("#bestModel").textContent = "Execute o notebook 05_export_web";
  }
}

function bindEvents() {
  $("#interpretButton").addEventListener("click", () => generateInterpretation());
  initializeInterpretationQuestions(generateInterpretation);
  $("#overviewModelSelect").addEventListener("change", () => renderOverview(state.metrics));
  $("#classSplit").addEventListener("change", renderClassChart);
  $("#featureSelect").addEventListener("change", renderFeatureChart);
  $("#featureSplit").addEventListener("change", renderFeatureChart);
  $("#confusionModel").addEventListener("change", renderConfusionMatrix);
  ["diagnosisFilter", "modelFilter", "resultFilter"].forEach((id) => {
    $(`#${id}`).addEventListener("change", () => {
      state.page = 1;
      $("#resultFilter").disabled = !$("#modelFilter").value;
      if (!$("#modelFilter").value) $("#resultFilter").value = "";
      loadSamples();
    });
  });
  $("#resultFilter").disabled = true;
  $("#pageSize").addEventListener("change", () => { state.size = Number($("#pageSize").value); state.page = 1; loadSamples(); });
  $("#clearFilters").addEventListener("click", () => {
    ["diagnosisFilter", "modelFilter", "resultFilter"].forEach((id) => { $(`#${id}`).value = ""; });
    $("#resultFilter").disabled = true;
    state.page = 1;
    loadSamples();
  });
  $("#prevPage").addEventListener("click", () => { if (state.page > 1) { state.page -= 1; loadSamples(); } });
  $("#nextPage").addEventListener("click", () => { if (state.page < state.pages) { state.page += 1; loadSamples(); } });
  $("#closeDrawer").addEventListener("click", closeDrawer);
  $("#drawerBackdrop").addEventListener("click", closeDrawer);
  $("#menuButton").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  document.querySelectorAll(".nav-link").forEach((link) => link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active"));
    link.classList.add("active");
    $("#sidebar").classList.remove("open");
  }));
  $("#refreshButton").addEventListener("click", async () => {
    await request(`${API}/cache/recarregar`, { method: "POST" });
    await loadDashboard();
    showToast("Dados recarregados.");
  });
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadDashboard();
});
