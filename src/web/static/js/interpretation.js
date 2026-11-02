const initializedQuestionForms = new WeakSet();

function initializeQuestionExamples(form, input) {
  if (initializedQuestionForms.has(form)) return;
  initializedQuestionForms.add(form);
  const badges = form.querySelectorAll("[data-question]");
  const updateSelection = () => {
    badges.forEach((badge) => {
      badge.setAttribute("aria-pressed", String(input.value === badge.dataset.question));
    });
  };
  input.addEventListener("input", () => {
    input.setCustomValidity("");
    updateSelection();
  });
  badges.forEach((badge) => {
    badge.setAttribute("aria-controls", input.id);
    badge.addEventListener("click", () => {
      if (input.disabled) return;
      input.value = badge.dataset.question;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.focus();
    });
  });
  updateSelection();
}

document.querySelectorAll(".interpretation-question-form, .route-question-form").forEach((form) => {
  initializeQuestionExamples(form, form.querySelector("input"));
});

function initializeInterpretationQuestions(generate) {
  const form = document.querySelector("#interpretationQuestionForm");
  const input = document.querySelector("#interpretationQuestion");
  initializeQuestionExamples(form, input);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!input.value.trim()) {
      input.setCustomValidity("Digite uma pergunta sobre as métricas do estudo.");
      input.reportValidity();
      return;
    }
    generate(input.value.trim());
  });
}

function beginInterpretationRequest(question) {
  const output = document.querySelector("#interpretationOutput");
  const controls = document.querySelectorAll("#interpretButton, #interpretationQuestionForm button, #interpretationQuestion");
  controls.forEach((control) => { control.disabled = true; });
  output.setAttribute("aria-busy", "true");
  document.querySelector("#interpretationQuestionButton").textContent = question ? "Respondendo…" : "Perguntar à LLM";
  const started = Date.now();
  const update = () => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    output.textContent = `${question ? `Respondendo: ${question}` : "Gerando interpretação das métricas agregadas."} Tempo de espera: ${seconds}s. Consultas repetidas podem usar o cache; a primeira depende da CPU e do modelo.`;
  };
  update();
  const timer = setInterval(update, 1000);
  return () => {
    clearInterval(timer);
    controls.forEach((control) => { control.disabled = false; });
    output.setAttribute("aria-busy", "false");
    document.querySelector("#interpretationQuestionButton").textContent = "Perguntar à LLM";
  };
}

function renderInterpretationComparison(models) {
  const container = document.querySelector("#interpretationComparison");
  const metrics = [
    ["recall", "Recall"],
    ["precision", "Precisão"],
    ["f1", "F1"],
    ["accuracy", "Acurácia"],
  ];
  const table = document.createElement("table");
  const caption = table.createCaption();
  caption.textContent = "Resultados reais no mesmo conjunto de teste (holdout), não gerados pela LLM. Δ = otimizado − baseline em pontos percentuais (p.p.); valores negativos indicam queda.";
  const head = table.createTHead().insertRow();
  for (const label of ["Modelo / experimento GA", "Métrica", "Baseline", "Otimizado", "Δ (p.p.)"]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    head.appendChild(cell);
  }
  const body = table.createTBody();
  for (const [name, model] of Object.entries(models)) {
    for (const [metric, label] of metrics) {
      const row = body.insertRow();
      const baseline = model.baseline_metrics[metric];
      const optimized = model.optimized_metrics[metric];
      const difference = (optimized - baseline) * 100;
      const roundedDifference = Number(difference.toFixed(2));
      const values = [
        `${name} · ${model.selected_experiment}`,
        label,
        `${(baseline * 100).toFixed(2)}%`,
        `${(optimized * 100).toFixed(2)}%`,
        `${roundedDifference > 0 ? "+" : ""}${roundedDifference.toFixed(2)}`,
      ];
      for (const value of values) {
        row.insertCell().textContent = value;
      }
    }
  }
  container.replaceChildren(table);
}
