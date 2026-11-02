from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js required for route map behavior test")
class RouteMapVisualizationTests(unittest.TestCase):
    def test_map_exposes_items_and_spreads_coincident_stops_away_from_depot(self) -> None:
        script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
class Element {
  constructor() { this.innerHTML=''; this.textContent=''; this.value='all'; this.dataset={}; }
  set innerHTML(value) {
    this._html=value;
    const options=[...value.matchAll(/<option value="([^"]+)">/g)];
    if (options.length && !options.some(option=>option[1]===this.value)) this.value=options[0][1];
  }
  get innerHTML() { return this._html; }
  replaceChildren() { this.innerHTML=''; }
  addEventListener() {}
}
const elements=Object.fromEntries(
  ['routeView','routeVehicleSelect','routeItemTypeSelect','routeMap','routeLegend','routeMapDetails',
   'deliveryRows','manifestReferenceHeading','manifestItemsHeading'].map(id=>[id,new Element()])
);
elements.routeView.value='optimized';
elements.routeVehicleSelect.value='all';
global.document={
  querySelector(selector) { return elements[selector.slice(1)]; },
  querySelectorAll() { return []; },
  addEventListener() {},
};
const context=vm.createContext(global);
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const coordinate={latitude:-23.5,longitude:-46.6};
const deliveries=[
  {id:'d1',name:'Destino um',...coordinate,priority:'critica',demand_kg:4,
    deadline_time:'10:00',requires_refrigeration:true,items:['medicamento & insumo']},
  {id:'d2',name:'Destino dois',...coordinate,priority:'alta',demand_kg:3,
    deadline_time:'11:00',requires_refrigeration:false,items:['curativo']},
];
const stops=deliveries.map((delivery,index)=>({
  delivery_id:delivery.id,name:delivery.name,...coordinate,priority:delivery.priority,
  demand_kg:delivery.demand_kg,arrival_time:`08:0${index+1}`,lateness_minutes:0,
  items:delivery.items,
}));
const route={
  vehicle_id:'vehicle-1',vehicle_name:'Veículo 1',distance_km:10,stops:[stops[0]],
  path:[coordinate,coordinate],
};
const route2={
  vehicle_id:'vehicle-2',vehicle_name:'Veículo 2',distance_km:12,stops:[stops[1]],
  path:[coordinate,coordinate],
};
const scenario={
  deliveries,
  data_source:{municipal_aggregates:[
    {delivery_id:'d1',historical_delivered_orders:12,historical_items:16,
      historical_medicine_items:3,historical_supply_items:0,
      historical_health_beauty_items:0},
    {delivery_id:'d2',historical_delivered_orders:8,historical_items:10,
      historical_medicine_items:0,historical_supply_items:5,
      historical_health_beauty_items:6},
  ],
    facility_aggregates:[{
  delivery_id:'d1',cnes_id:'cnes-123',facility_type:'Hospital',
  priority_score:9,priority_is_simulated:true,
  medicine_demand_kg:2,supply_demand_kg:1,demand_is_simulated:true,
    }],
  },
};
const solution={routes:[route]};
solution.routes.push(route2);
elements.routeItemTypeSelect.value='all';
context.scenario=scenario;
context.result={
  depot:{name:'Central',...coordinate},
  comparison:{optimized:solution,baseline:solution},
};
vm.runInContext('state.scenario=scenario; state.result=result; renderMap()',context);
const svg=elements.routeMap.innerHTML;
vm.runInContext('populateRouteVehicleSelector()',context);
vm.runInContext('renderMap()',context);
const allVehiclesSvg=elements.routeMap.innerHTML;
assert.equal((allVehiclesSvg.match(/class="route-map-stop"/g)||[]).length,2);
assert.ok(allVehiclesSvg.includes('Itens simulados: medicamento &amp; insumo'));
assert.ok(allVehiclesSvg.includes('Prioridade: critica'));
assert.ok(allVehiclesSvg.includes('Central de distribuição'));
assert.ok(allVehiclesSvg.includes('stroke-dasharray="2 3"'),'Overlapping stops need visible position connectors');
assert.ok(!allVehiclesSvg.includes('>Central</text>'),'Do not place a depot label over the busy hub');
vm.runInContext('selectMapStop("d1")',context);
assert.equal(vm.runInContext('state.selectedMapDeliveryId',context),'d1');
const details=elements.routeMapDetails.innerHTML;
assert.ok(details.includes('Itens médicos (origem indicada)'));
assert.ok(details.includes('medicamento &amp; insumo'));
assert.ok(details.includes('12 pedidos · 16 itens'));
assert.ok(details.includes('Hospital · prioridade 9/10 (simulada) · cnes-123'));
assert.ok(details.includes('2 kg medicamento · 1 kg insumo (simulados)'));
assert.ok(details.includes('08:01'));
const medicineStats=vm.runInContext('itemStatistics("medicamento")',context);
const supplyStats=vm.runInContext('itemStatistics("insumo")',context);
const healthBeautyStats=vm.runInContext('itemStatistics("saude_beleza_nao_confirmado")',context);
assert.equal(medicineStats.destinations,1);
assert.equal(medicineStats.historicalItems,3);
assert.equal(supplyStats.destinations,1);
assert.equal(supplyStats.historicalItems,5);
assert.equal(healthBeautyStats.destinations,1);
assert.equal(healthBeautyStats.historicalItems,6);
elements.routeVehicleSelect.value='vehicle-2';
vm.runInContext('renderMap()',context);
const selectedVehicleSvg=elements.routeMap.innerHTML;
assert.equal((selectedVehicleSvg.match(/class="route-map-stop"/g)||[]).length,1);
assert.ok(selectedVehicleSvg.includes('Veículo 2 · parada 1'));
assert.ok(!selectedVehicleSvg.includes('Veículo 1 · parada 1'));
elements.routeVehicleSelect.value='all';
elements.routeItemTypeSelect.value='medicamento';
vm.runInContext('renderMap()',context);
const medicineOnlySvg=elements.routeMap.innerHTML;
assert.equal((medicineOnlySvg.match(/class="route-map-stop"/g)||[]).length,1);
assert.ok(medicineOnlySvg.includes('Veículo 1 · parada 1'));
assert.ok(!medicineOnlySvg.includes('Veículo 2 · parada 1'));
elements.routeItemTypeSelect.value='saude_beleza_nao_confirmado';
vm.runInContext('renderMap()',context);
const healthBeautySvg=elements.routeMap.innerHTML;
assert.equal((healthBeautySvg.match(/class="route-map-stop"/g)||[]).length,1);
assert.ok(healthBeautySvg.includes('Veículo 2 · parada 1'));
scenario.data_source.kind='hybrid_sp';
scenario.data_source.geography_source='olist_sp';
scenario.data_source.distribution_manifest=[{
  delivery_id:'d1',lines:[{
    category:'medicamento',name:'Kit fictício <teste>',quantity:10,load_kg:0.5,
    stock_units:0,minimum_stock_units:5,daily_usage_units:2,
    stock_coverage_days:0,restock_lead_time_days:5,priority:'critica',
  }],
}];
vm.runInContext('renderDeliveries()',context);
assert.ok(elements.deliveryRows.innerHTML.includes('Kit fictício &lt;teste&gt;'));
assert.ok(elements.deliveryRows.innerHTML.includes('Estoque 0 · mínimo 5'));
assert.ok(elements.deliveryRows.innerHTML.includes('cobertura 0 dia(s)'));
assert.equal(elements.manifestItemsHeading.textContent,'Medicamentos, insumos e estoque simulado');
const distributionStats=vm.runInContext('itemStatistics("medicamento")',context);
assert.equal(distributionStats.quantity,10);
assert.equal(distributionStats.loadKg,0.5);
elements.routeItemTypeSelect.value='medicamento';
vm.runInContext('renderMap(); selectMapStop("d1")',context);
assert.ok(elements.routeMapDetails.innerHTML.includes('Distribuição e estoque simulados'));
assert.ok(elements.routeMapDetails.innerHTML.includes('Kit fictício &lt;teste&gt;'));
"""
        result = subprocess.run(
            [
                "node", "-e", script,
                str(ROOT / "src" / "web" / "static" / "js" / "routes.js"),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
