const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const state = {
  scenario: null,
  result: null,
  history: null,
  assistantBusy: false,
  selectedMapDeliveryId: null,
};
const ROUTE_COLORS = ["#7dd3fc", "#60a5fa", "#bfdbfe", "#38bdf8", "#ffffff"];
const ROUTE_DASHES = ["none", "12 5", "4 5", "16 5 4 5", "2 6"];

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.detail || "Não foi possível carregar o plano.");
    error.status = response.status;
    throw error;
  }
  return body;
}

function sourcePath(path) {
  return `${path}?source=${encodeURIComponent($("#scenarioSource").value)}`;
}

function renderDatasetDetails() {
  const data = state.scenario.data_source;
  $("#datasetDetails").hidden = !data || !["olist_sp", "cnes_sp", "hybrid_sp"].includes(data.kind);
  if (!data || !["olist_sp", "cnes_sp", "hybrid_sp"].includes(data.kind)) return;
  if (data.kind === "hybrid_sp") {
    $("#datasetSummary").textContent = `Distribuição simulada de medicamentos e insumos em ${state.scenario.deliveries.length} destinos; coordenadas: ${data.geography_source}. Inventário de referência: Hospital Supply Chain / Kaggle.`;
    $("#datasetMethod").textContent = `${data.demand_source} ${data.distribution_method.priority_rule} Estoque, consumo e kits são escalados para demonstração; não representam uma operação real. As categorias históricas Olist continuam separadas da demanda médica.`;
  } else if (data.kind === "cnes_sp") {
    const profile = data.kaggle_inventory_profile;
    const profileSummary = profile
      ? ` Perfil Kaggle: ${profile.records} registros de inventário; maiores médias diárias: ${profile.items.slice().sort((a, b) => b.average_usage_per_day - a.average_usage_per_day).slice(0, 3).map((item) => `${item.item_name} (${item.item_type}): ${item.average_usage_per_day}/dia`).join(" · ")}. Referência global, sem vínculo a estabelecimento.`
      : " Perfil Kaggle opcional ausente; o cenário continua com a demanda definida pelo CSV CNES ou valores simulados.";
    $("#datasetSummary").textContent = `${Number(data.establishments_sp_with_coordinates).toLocaleString("pt-BR")} estabelecimentos CNES em SP com coordenadas válidas · ${data.selected_establishments} destinos selecionados.${profileSummary}`;
    $("#datasetMethod").textContent = `${data.coordinates} ${data.demand_source}. ${data.depot} Coordenadas e cadastro CNES são reais conforme o CSV; itens, cargas, prazos e frota são simulados; demanda e prioridade só deixam de ser simuladas quando informadas no próprio arquivo. O perfil Kaggle não é cruzado com as unidades CNES e não altera a carga em kg.`;
  } else {
    const profile = data.kaggle_inventory_profile;
    const profileSummary = profile
      ? ` Perfil Kaggle Hospital Supply Chain: ${profile.records} registros; maiores médias/dia: ${profile.items.slice().sort((a, b) => b.average_usage_per_day - a.average_usage_per_day).slice(0, 3).map((item) => `${item.item_name} (${item.item_type}): ${item.average_usage_per_day}/dia`).join(" · ")}.`
      : "";
    $("#datasetSummary").textContent = `${data.delivered_orders_sp.toLocaleString("pt-BR")} pedidos entregues em SP · ${data.municipalities_with_coordinates} municípios com coordenadas · ${data.selected_municipalities} destinos selecionados. Período histórico: ${data.purchase_period.start} a ${data.purchase_period.end}.${profileSummary}`;
    $("#datasetMethod").textContent = `${data.selection} ${data.coordinates} ${data.discarded_geolocation_rows} linhas de geolocalização descartadas por coordenadas/prefixos inválidos. Carga médica, prioridades, frota e prazos são simulados; pesos históricos de produtos não são carga médica. Categorias Olist só são contadas como medicamento/insumo quando o nome da categoria é explícito; as demais ficam sem classificação. ${profile ? "O perfil Kaggle é global, sem ligação a CNES; é apenas referência e não altera a carga da rota." : ""}`;
  }
  $("#datasetFiles").replaceChildren(...Object.entries(data.files_sha256).map(([name, hash]) => {
    const item = document.createElement("li");
    item.textContent = `${name} · SHA-256 ${hash}`;
    return item;
  }));
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 4500);
}

function formatKm(value) {
  return `${Number(value).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} km`;
}

function formatKg(value) {
  return `${Number(value).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} kg`;
}

function formatDuration(seconds) {
  return `${Number(seconds).toFixed(2)} s`;
}

function classifyMedicalItem(value) {
  const words = String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLocaleLowerCase("pt-BR")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
  if (/\b(medicamento|medicamentos|remedio|remedios|farmacia|farmaceutico|farmaceutica|medicine|medicines|medication|medications|drug|drugs|pharmaceutical|pharmaceuticals)\b/.test(words)) {
    return "medicamento";
  }
  if (/\b(insumo|insumos|suprimento|suprimentos|curativo|curativos|material medico|materiais medicos|material hospitalar|materiais hospitalares|medical supply|medical supplies|hospital supply|hospital supplies|surgical supply|surgical supplies|material de emergencia)\b/.test(words)) {
    return "insumo";
  }
  if (["beleza saude", "saude beleza", "health beauty", "beauty health"].includes(words)) {
    return "saude_beleza_nao_confirmado";
  }
  return "nao_classificado";
}

function distributionFor(deliveryId) {
  return state.scenario.data_source?.distribution_manifest?.find(
    (manifest) => manifest.delivery_id === deliveryId,
  );
}

function distributionItemsHtml(manifest) {
  return manifest.lines.map((line) => (
    `<div><strong>${escapeHtml(line.name)} · ${line.quantity} kits · ${formatKg(line.load_kg)}</strong>`
    + `<small>Estoque ${line.stock_units} · mínimo ${line.minimum_stock_units} · uso ${line.daily_usage_units}/dia · cobertura ${line.stock_coverage_days} dia(s) · reposição ${line.restock_lead_time_days} dia(s) · ${escapeHtml(line.priority)} (simulado)</small></div>`
  )).join("");
}

function deliveryMatchesItemType(delivery, itemType) {
  if (!itemType || itemType === "all") return true;
  const simulatedMatch = (delivery.items || []).some(
    (item) => classifyMedicalItem(item) === itemType,
  );
  if (
    state.scenario.data_source?.kind === "hybrid_sp"
    && ["medicamento", "insumo"].includes(itemType)
  ) return simulatedMatch;
  const aggregate = state.scenario.data_source?.municipal_aggregates?.find(
    (item) => item.delivery_id === delivery.id,
  );
  const historicalCount = {
    medicamento: aggregate?.historical_medicine_items,
    insumo: aggregate?.historical_supply_items,
    saude_beleza_nao_confirmado: aggregate?.historical_health_beauty_items,
  }[itemType];
  return simulatedMatch || Number(historicalCount || 0) > 0;
}

function itemStatistics(itemType) {
  const deliveryIds = state.scenario.deliveries
    .filter((delivery) => deliveryMatchesItemType(delivery, itemType))
    .map((delivery) => delivery.id);
  const aggregates = state.scenario.data_source?.municipal_aggregates || [];
  const historicalField = itemType === "medicamento"
    ? "historical_medicine_items"
    : itemType === "insumo"
      ? "historical_supply_items"
      : "historical_health_beauty_items";
  const historicalCount = aggregates.reduce(
    (sum, aggregate) => sum + Number(aggregate[historicalField] || 0),
    0,
  );
  const simulatedCount = state.scenario.deliveries.filter(
    (delivery) => (delivery.items || []).some(
      (item) => classifyMedicalItem(item) === itemType,
    ),
  ).length;
  const lines = (state.scenario.data_source?.distribution_manifest || [])
    .flatMap((manifest) => manifest.lines)
    .filter((line) => line.category === itemType);
  return {
    destinations: deliveryIds.length,
    simulatedDestinations: simulatedCount,
    historicalItems: historicalCount,
    quantity: lines.reduce((sum, line) => sum + line.quantity, 0),
    loadKg: lines.reduce((sum, line) => sum + line.load_kg, 0),
  };
}

function currentSolution() {
  return state.result.comparison[$("#routeView").value];
}

function visibleRoutes(solution = currentSolution()) {
  const selectedVehicle = $("#routeVehicleSelect").value;
  if (!selectedVehicle || selectedVehicle === "all") return solution.routes;
  return solution.routes.filter((route) => route.vehicle_id === selectedVehicle);
}

function populateRouteVehicleSelector() {
  const select = $("#routeVehicleSelect");
  const previous = select.value;
  const routes = currentSolution().routes;
  select.innerHTML = [
    '<option value="all">Todos os veículos</option>',
    ...routes.map((route) => (
      `<option value="${escapeHtml(route.vehicle_id)}">${escapeHtml(route.vehicle_name)}</option>`
    )),
  ].join("");
  select.value = routes.some((route) => route.vehicle_id === previous)
    ? previous
    : "all";
}

function renderSummary() {
  const baseline = state.result.comparison.baseline;
  const optimized = state.result.comparison.optimized;
  $("#deliveryCount").textContent = state.scenario.deliveries.length;
  $("#vehicleCount").textContent = state.scenario.vehicles.length;
  $("#routeDistance").textContent = Number(optimized.total_distance_km).toFixed(1);
  $("#baselineDistance").textContent = Number(baseline.total_distance_km).toFixed(1);
  $("#optimizedDistance").textContent = Number(optimized.total_distance_km).toFixed(1);
  $("#distanceSaved").textContent = formatKm(state.result.comparison.distance_saved_km);
  $("#distanceSavedPercent").textContent = `${state.result.comparison.distance_saved_percent}% comparado à base`;
  $("#servedCount").textContent = `${optimized.served_delivery_count}/${optimized.delivery_count}`;
  $("#lateCount").textContent = `${optimized.late_delivery_count} entrega(s) após o prazo estimado`;
  for (const [itemType, prefix] of [
    ["medicamento", "medicine"],
    ["insumo", "supply"],
    ["saude_beleza_nao_confirmado", "healthBeauty"],
  ]) {
    const statistics = itemStatistics(itemType);
    if (itemType === "saude_beleza_nao_confirmado" && state.scenario.data_source?.kind === "cnes_sp") {
      $(`#${prefix}Count`).textContent = "—";
      $(`#${prefix}CountNote`).textContent = "Categoria disponível na fonte Olist";
    } else {
      $(`#${prefix}Count`).textContent = statistics.destinations;
      $(`#${prefix}CountNote`).textContent = (
        itemType === "saude_beleza_nao_confirmado"
          ? `${statistics.historicalItems} item(ns) Olist nesta categoria; não confirma itens médicos`
          : state.scenario.data_source?.kind === "hybrid_sp"
            ? `${statistics.quantity} kits · ${formatKg(statistics.loadKg)} · demanda simulada`
          : state.scenario.data_source?.kind === "cnes_sp"
            ? `${statistics.simulatedDestinations} destino(s) com itens médicos simulados`
          : `${statistics.simulatedDestinations} destino(s) simulados · `
            + `${statistics.historicalItems} item(ns) Olist explicitamente classificados`
      );
    }
  }
  $("#scenarioNotice").textContent = state.result.scenario_description;
}

function projectPoint(point, bounds, width, height) {
  const margin = 55;
  const longitudeFactor = Math.cos(((bounds.minLat + bounds.maxLat) / 2) * Math.PI / 180);
  const longitudeRange = Math.max(bounds.maxLon - bounds.minLon, 0.001);
  const latitudeRange = Math.max(bounds.maxLat - bounds.minLat, 0.001);
  const scale = Math.min(
    (width - margin * 2) / (longitudeRange * longitudeFactor),
    (height - margin * 2) / latitudeRange,
  );
  return {
    x: width / 2 + (point.longitude - (bounds.minLon + bounds.maxLon) / 2) * longitudeFactor * scale,
    y: height / 2 - (point.latitude - (bounds.minLat + bounds.maxLat) / 2) * scale,
  };
}

function declutterMapStops(stops) {
  const remaining = new Set(stops.map((_, index) => index));
  const positions = stops.map((stop) => ({ ...stop.point }));
  const clusterDistance = 36;

  while (remaining.size) {
    const seedIndex = remaining.values().next().value;
    remaining.delete(seedIndex);
    const cluster = [seedIndex];
    for (const index of [...remaining]) {
      const dx = stops[index].point.x - stops[seedIndex].point.x;
      const dy = stops[index].point.y - stops[seedIndex].point.y;
      if (Math.hypot(dx, dy) <= clusterDistance) {
        cluster.push(index);
        remaining.delete(index);
      }
    }
    if (cluster.length < 2) continue;

    const center = cluster.reduce((sum, index) => ({
      x: sum.x + stops[index].point.x / cluster.length,
      y: sum.y + stops[index].point.y / cluster.length,
    }), { x: 0, y: 0 });
    const radius = Math.max(24, (cluster.length * 18) / (2 * Math.PI));
    cluster.forEach((index, offset) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * offset) / cluster.length;
      positions[index] = {
        x: center.x + Math.cos(angle) * radius,
        y: center.y + Math.sin(angle) * radius,
      };
    });
  }
  return positions;
}

function findDepotMarkerPosition(depot, stopPoints, width, height) {
  const candidates = Array.from({ length: 16 }, (_, index) => {
    const angle = (2 * Math.PI * index) / 16;
    return {
      x: Math.min(width - 24, Math.max(24, depot.x + Math.cos(angle) * 34)),
      y: Math.min(height - 24, Math.max(24, depot.y + Math.sin(angle) * 34)),
    };
  });
  return candidates.reduce((best, candidate) => {
    const score = stopPoints.reduce((minimum, point) => Math.min(
      minimum,
      Math.hypot(candidate.x - point.x, candidate.y - point.y),
    ), Infinity);
    return score > best.score ? { point: candidate, score } : best;
  }, { point: candidates[0], score: -Infinity }).point;
}

function renderMapStopDetails(deliveryId) {
  const container = $("#routeMapDetails");
  const delivery = state.scenario.deliveries.find((item) => item.id === deliveryId);
  const itemType = $("#routeItemTypeSelect").value;
  if (!delivery || !deliveryMatchesItemType(delivery, itemType)) {
    container.textContent = "Selecione uma parada no mapa para consultar os itens e as condições da entrega.";
    return;
  }
  let assignment = null;
  for (const route of currentSolution().routes) {
    if (
      $("#routeVehicleSelect").value !== "all"
      && route.vehicle_id !== $("#routeVehicleSelect").value
    ) continue;
    const stopIndex = route.stops.findIndex((stop) => stop.delivery_id === deliveryId);
    if (stopIndex >= 0) {
      assignment = { route, stop: route.stops[stopIndex], sequence: stopIndex + 1 };
      break;
    }
  }
  const aggregate = state.scenario.data_source?.municipal_aggregates?.find(
    (item) => item.delivery_id === deliveryId,
  );
  const facility = state.scenario.data_source?.facility_aggregates?.find(
    (item) => item.delivery_id === deliveryId,
  );
  const items = delivery.items?.length ? delivery.items : ["Itens não informados"];
  const distribution = distributionFor(deliveryId);
  container.innerHTML = `
    <div class="route-map-detail-heading">
      <div>
        <span>${assignment
          ? `${escapeHtml(assignment.route.vehicle_name)} · parada ${assignment.sequence}`
          : "Sem veículo atribuído nesta estratégia"}</span>
        <h4>${escapeHtml(delivery.name)}</h4>
      </div>
      <span class="priority-badge priority-${escapeHtml(delivery.priority)}">${escapeHtml(delivery.priority)}</span>
    </div>
    <div class="route-map-detail-grid">
      <div><span>${facility ? "Itens médicos (origem indicada)" : "Itens simulados"}</span><strong>${items.map(escapeHtml).join(", ")}</strong></div>
      <div><span>Carga simulada</span><strong>${formatKg(delivery.demand_kg)}</strong></div>
      <div><span>Chegada estimada</span><strong>${escapeHtml(assignment?.stop.arrival_time || "Não atribuída")}</strong></div>
      <div><span>Prazo</span><strong>${escapeHtml(delivery.deadline_time)}</strong></div>
      <div><span>Cadeia fria</span><strong>${delivery.requires_refrigeration ? "Necessária" : "Não necessária"}</strong></div>
      ${distribution ? `<div><span>Distribuição e estoque simulados</span>${distributionItemsHtml(distribution)}</div>` : ""}
      ${facility ? `<div><span>Estabelecimento CNES</span><strong>${escapeHtml(facility.facility_type)} · prioridade ${facility.priority_score}/10${facility.priority_is_simulated ? " (simulada)" : ""} · ${escapeHtml(facility.cnes_id)}</strong></div>` : ""}
      ${facility ? `<div><span>Demanda por categoria</span><strong>${formatKg(facility.medicine_demand_kg)} medicamento · ${formatKg(facility.supply_demand_kg)} insumo${facility.demand_is_simulated ? " (simulados)" : ""}</strong></div>` : ""}
      ${aggregate ? `<div><span>Histórico agregado Olist</span><strong>${Number(aggregate.historical_delivered_orders).toLocaleString("pt-BR")} pedidos · ${Number(aggregate.historical_items).toLocaleString("pt-BR")} itens</strong></div>` : ""}
      ${aggregate ? `<div><span>Olist · itens por categoria</span><strong>${Number(aggregate.historical_medicine_items || 0)} medicamento(s) explícitos · ${Number(aggregate.historical_supply_items || 0)} insumo(s) explícitos · ${Number(aggregate.historical_health_beauty_items || 0)} saúde/beleza não confirmado(s) · ${Number(aggregate.historical_unclassified_items || 0)} em outras categorias</strong></div>` : ""}
    </div>
    <small class="route-map-detail-note">${facility ? "As coordenadas e o cadastro do estabelecimento vêm do CNES; demanda, itens, prioridade, depósito e frota podem ser simulados conforme a origem indicada." : "A carga e os itens médicos são simulados; o histórico Olist, quando disponível, é agregado e não representa carga médica real."}</small>
  `;
}

function selectMapStop(deliveryId) {
  state.selectedMapDeliveryId = deliveryId;
  renderMapStopDetails(deliveryId);
}

function renderMap() {
  const solution = currentSolution();
  const itemType = $("#routeItemTypeSelect").value;
  const routes = visibleRoutes(solution).map((route) => ({
    ...route,
    routeIndex: solution.routes.indexOf(route),
    stops: route.stops.filter((stop) => {
      const delivery = state.scenario.deliveries.find(
        (item) => item.id === stop.delivery_id,
      );
      return delivery && deliveryMatchesItemType(delivery, itemType);
    }),
  })).filter((route) => route.stops.length);
  const width = 1000;
  const height = 590;
  const points = [state.result.depot, ...state.scenario.deliveries];
  const bounds = {
    minLat: Math.min(...points.map((point) => point.latitude)),
    maxLat: Math.max(...points.map((point) => point.latitude)),
    minLon: Math.min(...points.map((point) => point.longitude)),
    maxLon: Math.max(...points.map((point) => point.longitude)),
  };
  const depot = projectPoint(state.result.depot, bounds, width, height);
  const mapStops = routes.flatMap((route) => route.stops.map(
    (stop) => ({
      stop,
      route,
      routeIndex: route.routeIndex,
      stopIndex: solution.routes[route.routeIndex].stops.findIndex(
        (candidate) => candidate.delivery_id === stop.delivery_id,
      ),
      point: projectPoint(stop, bounds, width, height),
    }),
  ));
  const displayPoints = declutterMapStops(mapStops);
  const assignedIds = new Set(solution.routes.flatMap((route) => (
    route.stops.map((stop) => stop.delivery_id)
  )));
  const unassignedPoints = state.scenario.deliveries
    .filter((delivery) => !assignedIds.has(delivery.id))
    .map((delivery) => projectPoint(delivery, bounds, width, height));
  const depotMarker = findDepotMarkerPosition(
    depot, [...displayPoints, ...unassignedPoints], width, height,
  );
  let paths = "";
  let legend = "";

  routes.forEach((route) => {
    const colorIndex = route.routeIndex;
    const color = ROUTE_COLORS[colorIndex % ROUTE_COLORS.length];
    const routePoints = itemType === "all"
      ? route.path
      : [
        state.result.depot,
        ...route.stops,
        state.result.depot,
      ];
    const coordinates = routePoints.map((point) => projectPoint(point, bounds, width, height));
    const pointsString = coordinates.map((point) => `${point.x},${point.y}`).join(" ");
    paths += `<polyline points="${pointsString}" fill="none" stroke="${color}" stroke-dasharray="${ROUTE_DASHES[colorIndex % ROUTE_DASHES.length]}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity=".95"/>`;
    const routeSummary = itemType === "all"
      ? formatKm(route.distance_km)
      : `${route.stops.length} entrega(s) deste tipo`;
    legend += `<span><i style="--route-color:${color}"></i>${escapeHtml(route.vehicle_name)} · ${routeSummary}</span>`;
  });

  const stopMarkers = mapStops.map(({ stop, route, stopIndex, routeIndex }, index) => {
    const actual = mapStops[index].point;
    const point = displayPoints[index];
    const color = ROUTE_COLORS[routeIndex % ROUTE_COLORS.length];
    const itemNames = stop.items?.length ? stop.items.join(", ") : "Não informado";
    const deadline = state.scenario.deliveries.find(
      (delivery) => delivery.id === stop.delivery_id,
    )?.deadline_time || "—";
    const title = [
      stop.name,
      `Veículo: ${route.vehicle_name} · parada ${stopIndex + 1}`,
      `Itens simulados: ${itemNames}`,
      `Prioridade: ${stop.priority} · carga simulada: ${formatKg(stop.demand_kg)}`,
      `Chegada: ${stop.arrival_time} · prazo: ${deadline}`,
    ].join("\n");
    const connector = Math.hypot(point.x - actual.x, point.y - actual.y) > 1
      ? `<line x1="${actual.x}" y1="${actual.y}" x2="${point.x}" y2="${point.y}" stroke="${color}" stroke-width="1.5" stroke-dasharray="2 3" opacity=".75"/>`
      : "";
    return `
      ${connector}
      <g class="route-map-stop" data-delivery-id="${escapeHtml(stop.delivery_id)}" tabindex="0" role="button" aria-label="${escapeHtml(title.replaceAll("\n", ". "))}">
        <title>${escapeHtml(title)}</title>
        <circle cx="${point.x}" cy="${point.y}" r="15" fill="#0b294b" stroke="${color}" stroke-width="4"/>
        <text x="${point.x}" y="${point.y + 5}" text-anchor="middle" fill="#fff" font-size="12" font-weight="700" pointer-events="none">${stopIndex + 1}</text>
      </g>
    `;
  }).join("");

  const deliveryMarkers = state.scenario.deliveries
    .filter((delivery) => (
      !assignedIds.has(delivery.id)
      && deliveryMatchesItemType(delivery, itemType)
    ))
    .map((delivery) => {
      const point = projectPoint(delivery, bounds, width, height);
      return `<g class="route-map-stop route-map-stop-unassigned" data-delivery-id="${escapeHtml(delivery.id)}" tabindex="0" role="button" aria-label="${escapeHtml(`Entrega não atribuída: ${delivery.name}. Itens: ${delivery.items?.join(", ") || "não informados"}`)}"><title>${escapeHtml(`${delivery.name} · não atribuída · itens simulados: ${delivery.items?.join(", ") || "não informados"}`)}</title><circle cx="${point.x}" cy="${point.y}" r="10" fill="#ff7b91" stroke="#fff" stroke-width="2"/></g>`;
    })
    .join("");
  const depotConnector = Math.hypot(depotMarker.x - depot.x, depotMarker.y - depot.y) > 1
    ? `<line x1="${depot.x}" y1="${depot.y}" x2="${depotMarker.x}" y2="${depotMarker.y}" stroke="#f8bd68" stroke-width="2" stroke-dasharray="3 4" opacity=".8"/>`
    : "";

  $("#routeMap").innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="group" aria-label="Rotas esquemáticas mostrando ${routes.length} de ${solution.routes.length} veículos">
      <defs><pattern id="routeGrid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,.04)" stroke-width="1"/></pattern></defs>
      <rect width="${width}" height="${height}" fill="url(#routeGrid)"/>
      <text x="25" y="32" fill="#c5d9f0" font-size="13">Esquema geográfico · não representa vias ou trânsito</text>
      ${paths}
      ${stopMarkers}
      ${deliveryMarkers}
      ${depotConnector}
      <circle cx="${depotMarker.x}" cy="${depotMarker.y}" r="17" fill="#0b294b" stroke="#f8bd68" stroke-width="2"/>
      <path d="M ${depotMarker.x} ${depotMarker.y - 11} L ${depotMarker.x + 10} ${depotMarker.y + 8} L ${depotMarker.x - 10} ${depotMarker.y + 8} Z" fill="#f8bd68" stroke="#0b294b" stroke-width="2"><title>Central de distribuição</title></path>
    </svg>
  `;
  $("#routeLegend").innerHTML = `${legend}<span><i class="route-depot-legend"></i>Central de distribuição</span>`;
  renderMapStopDetails(state.selectedMapDeliveryId);
}

function renderRoutes() {
  const solution = currentSolution();
  $("#vehicleRoutes").innerHTML = solution.routes.map((route, routeIndex) => `
    <article class="route-vehicle-card">
      <div class="route-vehicle-heading">
        <span class="route-color-dot" style="--route-color:${ROUTE_COLORS[routeIndex % ROUTE_COLORS.length]}"></span>
        <div><h4>${escapeHtml(route.vehicle_name)}</h4><small>${formatKm(route.distance_km)} · ${route.stops.length} parada(s)</small></div>
      </div>
      <div class="route-vehicle-capacity">
        <span>Carga ${formatKg(route.load_kg)} / ${formatKg(route.capacity_kg)}</span>
        <span>Autonomia restante ${formatKm(route.remaining_range_km)}</span>
      </div>
      <ol>${route.stops.map((stop) => `
        <li>
          <span class="priority-dot priority-${escapeHtml(stop.priority)}"></span>
          <div><strong>${escapeHtml(stop.name)}</strong><small>${escapeHtml(stop.arrival_time)} · prazo ${escapeHtml(state.scenario.deliveries.find((delivery) => delivery.id === stop.delivery_id)?.deadline_time || "—")} · ${formatKg(stop.demand_kg)}</small></div>
          ${stop.lateness_minutes > 0 ? `<span class="late-label">+${Math.round(stop.lateness_minutes)} min</span>` : ""}
        </li>
      `).join("")}</ol>
    </article>
  `).join("");
}

function renderDeliveries() {
  const solution = currentSolution();
  const assignments = new Map();
  solution.routes.forEach((route) => route.stops.forEach((stop) => assignments.set(stop.delivery_id, {
    vehicle: route.vehicle_name,
    arrival: stop.arrival_time,
    lateness: stop.lateness_minutes,
  })));
  const hybrid = state.scenario.data_source?.kind === "hybrid_sp";
  $("#manifestReferenceHeading").textContent = hybrid ? "Base geográfica" : "Referência Olist / CNES";
  $("#manifestItemsHeading").textContent = hybrid ? "Medicamentos, insumos e estoque simulado" : "Categorias históricas / demanda de referência";
  $("#deliveryRows").innerHTML = state.scenario.deliveries.map((delivery) => {
    const assignment = assignments.get(delivery.id);
    const aggregate = state.scenario.data_source?.municipal_aggregates?.find((item) => item.delivery_id === delivery.id);
    const facility = state.scenario.data_source?.facility_aggregates?.find((item) => item.delivery_id === delivery.id);
    const distribution = distributionFor(delivery.id);
    return `
      <tr>
        <td><strong>${escapeHtml(delivery.name)}</strong><small>${escapeHtml(delivery.items.join(", "))}</small></td>
        <td>${hybrid ? escapeHtml(state.scenario.data_source.geography_source) : aggregate ? Number(aggregate.historical_delivered_orders).toLocaleString("pt-BR") : (facility ? escapeHtml(facility.facility_type) : "Não se aplica")}</td>
        <td>${distribution ? distributionItemsHtml(distribution) : aggregate ? `${Number(aggregate.historical_medicine_items || 0)} medicamento(s) explícitos · ${Number(aggregate.historical_supply_items || 0)} insumo(s) explícitos · ${Number(aggregate.historical_health_beauty_items || 0)} saúde/beleza (não confirmado) · ${Number(aggregate.historical_unclassified_items || 0)} outras categorias` : (facility ? `${formatKg(facility.medicine_demand_kg)} medicamento · ${formatKg(facility.supply_demand_kg)} insumo · prioridade ${facility.priority_score}/10${facility.demand_is_simulated || facility.priority_is_simulated ? " (valores simulados)" : " (CSV local)"}` : "Não há categoria histórica")}</td>
        <td><span class="priority-badge priority-${escapeHtml(delivery.priority)}">${escapeHtml(delivery.priority)}</span></td>
        <td>${formatKg(delivery.demand_kg)}</td>
        <td>${escapeHtml(delivery.deadline_time)}</td>
        <td>${delivery.requires_refrigeration ? "Sim" : "Não"}</td>
        <td>${assignment ? `${escapeHtml(assignment.vehicle)} · ${escapeHtml(assignment.arrival)}` : "Não atribuída"}</td>
      </tr>
    `;
  }).join("");
}

function renderConvergence() {
  const history = state.result.algorithm.history;
  const width = 600;
  const height = 190;
  const pad = 35;
  const values = history.map((point) => point.best_fitness);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1);
  const points = values.map((value, index) => {
    const x = pad + (index / Math.max(values.length - 1, 1)) * (width - pad * 2);
    const y = height - pad - ((value - min) / span) * (height - pad * 2);
    return `${x},${y}`;
  }).join(" ");
  $("#gaChart").innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Convergência do fitness ao longo de ${history.length} gerações">
      <path d="M ${pad} ${pad} V ${height - pad} H ${width - pad}" fill="none" stroke="rgba(255,255,255,.18)"/>
      <polyline points="${points}" fill="none" stroke="#7dd3fc" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      <text x="${pad}" y="${height - 8}" fill="#c5d9f0" font-size="12">Geração 1</text>
      <text x="${width - pad}" y="${height - 8}" fill="#c5d9f0" font-size="12" text-anchor="end">Geração ${history.length}</text>
      <text x="${pad}" y="18" fill="#c5d9f0" font-size="12">Fitness menor é melhor · distância + atraso ponderado</text>
    </svg>
  `;
  const config = state.result.algorithm.config;
  $("#gaParameters").innerHTML = `
    <span>População: <strong>${config.population_size}</strong></span>
    <span>Gerações: <strong>${config.generations}</strong></span>
    <span>Mutação: <strong>${Math.round(config.mutation_rate * 100)}%</strong></span>
    <span>Avaliações únicas: <strong>${state.result.algorithm.evaluated_chromosomes}</strong></span>
    <span>Tempo: <strong>${formatDuration(state.result.elapsed_seconds)}</strong></span>
  `;
}

function renderAll() {
  renderSummary();
  populateRouteVehicleSelector();
  renderMap();
  renderRoutes();
  renderDeliveries();
  renderConvergence();
}

async function optimizeRoutes() {
  const button = $("#optimizeButton");
  const seedInput = $("#routeSeed");
  if (!seedInput.checkValidity()) {
    seedInput.reportValidity();
    return;
  }
  if (!state.scenario) {
    showToast("Carregue uma fonte de dados válida antes de otimizar.");
    return;
  }
  button.disabled = true;
  $("#scenarioSource").disabled = true;
  document.querySelectorAll("[data-action], #routeQuestionForm button").forEach((item) => { item.disabled = true; });
  $("#scenarioNotice").textContent = "Executando busca genética…";
  try {
    state.result = await request("/api/rotas/otimizar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed: Number(seedInput.value), source: $("#scenarioSource").value }),
    });
    renderAll();
    await refreshHistory();
    showToast("Rotas otimizadas e salvas localmente.");
  } catch (error) {
    $("#scenarioNotice").textContent = error.message;
    showToast(error.message);
  } finally {
    button.disabled = false;
    $("#scenarioSource").disabled = false;
    document.querySelectorAll("[data-action], #routeQuestionForm button").forEach((item) => { item.disabled = !state.result; });
  }
}

async function refreshHistory() {
  const notice = $("#routeHistoryNotice");
  try {
    state.history = await request(sourcePath("/api/rotas/historico"));
    const count = state.history.run_count;
    const runLabel = count === 1 ? "simulação sintética registrada" : "simulações sintéticas registradas";
    notice.textContent = count >= state.history.minimum_runs_for_trend
      ? `${count} ${runLabel} nos últimos ${state.history.window_days} dias. O comparativo mede execuções do algoritmo, não operações reais.`
      : `${count} ${count === 1 ? "simulação registrada" : "simulações registradas"} nos últimos ${state.history.window_days} dias. Registre ao menos mais uma execução para comparar variações; isso não representa uma tendência operacional.`;
  } catch (error) {
    notice.textContent = `Não foi possível carregar o histórico: ${error.message}`;
  }
}

function renderAssistantResult(result) {
  const content = result.content;
  const output = $("#assistantOutput");
  let mode = result.source === "local"
    ? "Resposta local, baseada nos dados do plano (sem LLM)."
    : "Resposta gerada pela LLM configurada.";
  if (result.verified_plan_fields) mode += " Sequências e métricas anexadas diretamente do plano, sem recálculo pela LLM.";
  if (result.cache?.hit) mode += " Reutilizada do cache deste plano.";
  if (result.action === "instructions") {
    output.innerHTML = `
      <p class="assistant-mode">${escapeHtml(mode)}</p>
      <p>${escapeHtml(content.summary)}</p>
      ${content.vehicle_instructions.map((vehicle) => `
        <h4>${escapeHtml(vehicle.vehicle)}</h4>
        <ol>${vehicle.ordered_stops.map((stop) => `<li>${escapeHtml(typeof stop === "string" ? stop : JSON.stringify(stop))}</li>`).join("")}</ol>
        <p>${escapeHtml(Array.isArray(vehicle.notes) ? vehicle.notes.join(" ") : vehicle.notes)}</p>
      `).join("")}
      <h4>Verificações operacionais</h4>
      <ul>${content.operational_checks.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <p><strong>${escapeHtml(content.disclaimer)}</strong></p>
    `;
    return;
  }
  if (result.action === "daily_report" || result.action === "weekly_report") {
    output.innerHTML = `
      <p class="assistant-mode">${escapeHtml(mode)}</p>
      <p><strong>${escapeHtml(content.period)}</strong></p>
      <p>${escapeHtml(content.summary)}</p>
      <h4>Métricas</h4>
      <ul>${Object.entries(content.key_metrics).map(([key, value]) => `<li><strong>${escapeHtml(key)}:</strong> ${escapeHtml(typeof value === "string" ? value : JSON.stringify(value))}</li>`).join("")}</ul>
      <h4>Melhorias propostas</h4>
      <ul>${content.improvements.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <h4>Limitações</h4>
      <ul>${content.caveats.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <p><strong>${escapeHtml(content.disclaimer)}</strong></p>
    `;
    return;
  }
  output.innerHTML = `
    <p class="assistant-mode">${escapeHtml(mode)}</p>
    <p>${escapeHtml(content.answer)}</p>
    <h4>Dados que sustentam a resposta</h4>
    <ul>${content.supporting_facts.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    <h4>Informações não disponíveis</h4>
    <ul>${content.missing_information.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    <p><strong>${escapeHtml(content.disclaimer)}</strong></p>
  `;
}

async function askAssistant(action, question = null, button = null) {
  if (state.assistantBusy) return;
  const output = $("#assistantOutput");
  if (!state.result) {
    output.textContent = "Otimize as rotas da fonte selecionada antes de consultar o assistente.";
    return;
  }
  state.assistantBusy = true;
  const controls = document.querySelectorAll("[data-action], #routeQuestionForm button, #routeQuestion, #optimizeButton, #scenarioSource");
  controls.forEach((item) => { item.disabled = true; });
  $("#scenarioSource").disabled = true;
  output.setAttribute("aria-busy", "true");
  const started = Date.now();
  const update = () => {
    output.textContent = `Gerando resposta fundamentada no plano… ${Math.floor((Date.now() - started) / 1000)}s. Consultas repetidas podem usar o cache.`;
  };
  update();
  const timer = setInterval(update, 1000);
  try {
    const result = await request("/api/rotas/assistente", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, question, source: $("#scenarioSource").value }),
    });
    renderAssistantResult(result);
  } catch (error) {
    output.textContent = error.message;
    showToast(error.message);
  } finally {
    clearInterval(timer);
    output.setAttribute("aria-busy", "false");
    state.assistantBusy = false;
    controls.forEach((item) => { item.disabled = false; });
  }
}

async function loadPage() {
  state.scenario = null;
  state.result = null;
  state.history = null;
  state.selectedMapDeliveryId = null;
  $("#scenarioSource").disabled = true;
  $("#optimizeButton").disabled = true;
  $("#datasetDetails").hidden = true;
  $("#scenarioNotice").textContent = "Preparando cenário e agregando dados locais de São Paulo…";
  $("#routeMapDetails").textContent = "Selecione uma parada no mapa para consultar os itens e as condições da entrega.";
  for (const id of ["routeMap", "routeLegend", "vehicleRoutes", "gaChart", "gaParameters"]) {
    $(`#${id}`).replaceChildren();
  }
  for (const id of ["deliveryCount", "vehicleCount", "routeDistance", "baselineDistance", "optimizedDistance", "distanceSaved", "distanceSavedPercent", "servedCount", "lateCount", "medicineCount", "supplyCount", "healthBeautyCount", "medicineCountNote", "supplyCountNote", "healthBeautyCountNote"]) {
    $(`#${id}`).textContent = "—";
  }
  $("#deliveryRows").innerHTML = '<tr><td colspan="8">Carregando a fonte selecionada…</td></tr>';
  $("#assistantOutput").textContent = "As respostas usarão somente o plano e o histórico da fonte selecionada.";
  $("#routeHistoryNotice").textContent = "Carregando histórico da fonte selecionada…";
  document.querySelectorAll("[data-action], #routeQuestionForm button").forEach((item) => { item.disabled = true; });
  try {
    state.scenario = await request(sourcePath("/api/rotas/cenario"));
    renderDatasetDetails();
    $("#scenarioNotice").textContent = state.scenario.description;
    try {
      state.result = await request(sourcePath("/api/rotas/resultado"));
    } catch (error) {
      if (error.status !== 404) throw error;
      await optimizeRoutes();
    }
    if (state.result) renderAll();
    await refreshHistory();
  } catch (error) {
    $("#scenarioNotice").textContent = error.message;
    showToast(error.message);
    $("#deliveryRows").innerHTML = '<tr><td colspan="8">Não foi possível carregar esta fonte. Consulte o aviso acima.</td></tr>';
  } finally {
    $("#scenarioSource").disabled = false;
    $("#optimizeButton").disabled = !state.scenario;
    document.querySelectorAll("[data-action], #routeQuestionForm button").forEach((item) => { item.disabled = !state.result; });
  }
}

function bindEvents() {
  $("#scenarioSource").addEventListener("change", loadPage);
  $("#optimizeButton").addEventListener("click", optimizeRoutes);
  $("#routeView").addEventListener("change", () => {
    if (!state.result) return;
    populateRouteVehicleSelector();
    state.selectedMapDeliveryId = null;
    renderMap();
    renderRoutes();
    renderDeliveries();
  });
  $("#routeVehicleSelect").addEventListener("change", () => {
    if (!state.result) return;
    state.selectedMapDeliveryId = null;
    renderMap();
  });
  $("#routeItemTypeSelect").addEventListener("change", () => {
    if (!state.result) return;
    state.selectedMapDeliveryId = null;
    renderMap();
  });
  $("#routeMap").addEventListener("click", (event) => {
    const marker = event.target.closest("[data-delivery-id]");
    if (!marker) return;
    selectMapStop(marker.dataset.deliveryId);
  });
  $("#routeMap").addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const marker = event.target.closest("[data-delivery-id]");
    if (!marker) return;
    event.preventDefault();
    selectMapStop(marker.dataset.deliveryId);
  });
  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => askAssistant(button.dataset.action, null, button));
  });
  const questionForm = $("#routeQuestionForm");
  const questionInput = $("#routeQuestion");
  initializeQuestionExamples(questionForm, questionInput);
  questionForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) {
      questionInput.setCustomValidity("Digite uma pergunta sobre o plano de rotas.");
      questionInput.reportValidity();
      return;
    }
    askAssistant("question", question, event.submitter);
  });
  $("#menuButton").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadPage();
});
