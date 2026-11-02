from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js required for dashboard behavior test")
class OverviewModelSelectorTests(unittest.TestCase):
    def test_selector_defaults_to_best_model_and_displays_other_model_metrics(self) -> None:
        for name in ("index.html", "pcos.html"):
            template = (ROOT / "src" / "web" / "templates" / name).read_text(
                encoding="utf-8"
            )
            self.assertIn('id="overviewModelSelect"', template)
            self.assertIn('id="overviewModelBadge"', template)
        script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
class Element {
  constructor() { this.textContent=''; this.value=''; this._html=''; }
  set innerHTML(value) {
    this._html=value;
    const options=[...value.matchAll(/<option value="([^"]+)">/g)];
    if (!options.some(option=>option[1]===this.value)) this.value=options[0]?.[1] || '';
  }
  get innerHTML() { return this._html; }
}
const ids=['overviewModelSelect','contractVersion','sampleCount','modelCount','featureCount',
  'overviewModelTitle','bestModel','overviewModelBadge','metricAccuracy','metricRecall',
  'metricF1','metricPrecision'];
const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
global.document={
  querySelector(selector) { return elements[selector.slice(1)]; },
  addEventListener() {},
};
const context=vm.createContext(global);
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const metrics={
  versao_contrato:2, quantidade_amostras:100, quantidade_modelos:2,
  quantidade_atributos:10, melhor_modelo:'SVM',
  por_modelo:{
    'SVM':{acuracia:.98,__RECALL_FIELD__:.97,__F1_FIELD__:.96,__PRECISION_FIELD__:.95},
    'Random Forest':{acuracia:.94,__RECALL_FIELD__:.90,__F1_FIELD__:.89,__PRECISION_FIELD__:.88},
  },
};
context.metrics=metrics;
vm.runInContext('state.models=Object.keys(metrics.por_modelo)',context);
context.populateOverviewModelSelector(metrics);
context.renderOverview(metrics);
assert.equal(elements.overviewModelSelect.value,'SVM');
assert.equal(elements.bestModel.textContent,'SVM');
assert.equal(elements.overviewModelTitle.textContent,'Modelo recomendado');
assert.equal(elements.overviewModelBadge.textContent,'__RECOMMENDED_BADGE__');
assert.equal(elements.metricRecall.textContent,'97.0%');
elements.overviewModelSelect.value='Random Forest';
context.renderOverview(metrics);
assert.equal(elements.bestModel.textContent,'Random Forest');
assert.equal(elements.overviewModelTitle.textContent,'Modelo selecionado');
assert.equal(elements.overviewModelBadge.textContent,'__SELECTED_BADGE__ 90.0%');
assert.equal(elements.metricAccuracy.textContent,'94.0%');
assert.equal(elements.metricRecall.textContent,'90.0%');
assert.equal(elements.metricF1.textContent,'89.0%');
assert.equal(elements.metricPrecision.textContent,'88.0%');
assert.equal((elements.overviewModelSelect.innerHTML.match(/<option /g) || []).length,2);
"""
        for js_name, recall_field, f1_field, precision_field, class_name in (
            ("app.js", "recall_maligno", "f1_maligno", "precisao_maligno", "maligno"),
            ("pcos.js", "recall_pcos", "f1_pcos", "precisao_pcos", "PCOS"),
        ):
            with self.subTest(script=js_name):
                runnable_script = (
                    script.replace("__RECALL_FIELD__", recall_field)
                    .replace("__F1_FIELD__", f1_field)
                    .replace("__PRECISION_FIELD__", precision_field)
                    .replace("__RECOMMENDED_BADGE__", f"Maior recall {class_name}")
                    .replace("__SELECTED_BADGE__", f"Recall {class_name}")
                )
                result = subprocess.run(
                    [
                        "node", "-e", runnable_script,
                        str(ROOT / "src" / "web" / "static" / "js" / js_name),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
