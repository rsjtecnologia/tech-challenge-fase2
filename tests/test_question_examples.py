from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ExampleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.questions = []
        self.scripts = []
        self.inputs = {}

    def handle_starttag(self, tag, attrs) -> None:
        attrs = dict(attrs)
        if tag == "button" and "data-question" in attrs:
            self.questions.append(attrs)
        if tag == "script" and "src" in attrs:
            self.scripts.append(attrs["src"])
        if tag == "input":
            self.inputs[attrs.get("id")] = attrs


class QuestionExampleTests(unittest.TestCase):
    def test_all_pages_have_four_non_submitting_examples_and_shared_helper(self) -> None:
        for name, input_id in (
            ("index.html", "interpretationQuestion"),
            ("pcos.html", "interpretationQuestion"),
            ("routes.html", "routeQuestion"),
        ):
            with self.subTest(page=name):
                parser = ExampleParser()
                parser.feed((ROOT / "src" / "web" / "templates" / name).read_text(encoding="utf-8"))
                self.assertEqual(len(parser.questions), 4)
                for badge in parser.questions:
                    self.assertEqual(badge["type"], "button")
                    self.assertTrue(badge["data-question"].strip())
                    self.assertLessEqual(len(badge["data-question"]), int(parser.inputs[input_id]["maxlength"]))
                helper = next(i for i, src in enumerate(parser.scripts) if "interpretation.js" in src)
                page_script = next(i for i, src in enumerate(parser.scripts) if (
                    "/js/app.js" in src or "/js/pcos.js" in src or "/js/routes.js" in src
                ))
                self.assertLess(helper, page_script)
                self.assertIn("?v=", parser.scripts[helper])
                self.assertIn("?v=", parser.scripts[page_script])
                charts = [i for i, src in enumerate(parser.scripts) if "chart.umd" in src]
                if charts:
                    self.assertLess(helper, charts[0])

    @unittest.skipUnless(shutil.which("node"), "Node.js required for DOM event regression")
    def test_examples_fill_focus_and_only_explicit_submit_generates(self) -> None:
        script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
class Element extends EventTarget {
  constructor() { super(); this.attrs={}; this.value=''; this.disabled=false; this.id='question'; }
  setAttribute(key,value) { this.attrs[key]=value; }
  setCustomValidity(value) { this.error=value; }
  reportValidity() { this.reported=true; }
  focus() { this.focused=true; }
}
const input=new Element();
const form=new Element();
const badges=['Maior recall?', 'Recall caiu?', 'Recall e precisão?', 'F1 antes e depois?']
  .map(question=>{const el=new Element();el.dataset={question};return el;});
form.querySelectorAll=()=>badges;
form.querySelector=()=>input;
global.document={
  querySelector:s=>s==='#interpretationQuestionForm'?form:input,
  querySelectorAll:()=>[form],
};
vm.runInThisContext(fs.readFileSync(process.argv[2], 'utf8'));
badges[0].dispatchEvent(new Event('click'));
assert.equal(input.value,badges[0].dataset.question,'Must work before page initialization');
let inputEvents=0;
input.addEventListener('input',()=>inputEvents++);
initializeQuestionExamples(form,input);
badges[0].dispatchEvent(new Event('click'));
assert.equal(inputEvents,1,'Repeated initialization must not duplicate listeners');
const calls=[];
initializeInterpretationQuestions(question=>calls.push(question));
for(const badge of badges) {
  input.setCustomValidity('erro anterior');
  badge.dispatchEvent(new Event('click'));
  assert.equal(input.value,badge.dataset.question);
  assert.equal(input.error,'');
  assert.equal(input.focused,true);
  assert.equal(badge.attrs['aria-controls'],input.id);
  assert.equal(badge.attrs['aria-pressed'],'true');
  assert.equal(calls.length,0);
}
input.value='  Pergunta editada  ';
input.dispatchEvent(new Event('input'));
assert.ok(badges.every(b=>b.attrs['aria-pressed']==='false'));
form.dispatchEvent(new Event('submit',{cancelable:true}));
assert.deepEqual(calls,['Pergunta editada']);
input.value='   ';
form.dispatchEvent(new Event('submit',{cancelable:true}));
assert.equal(calls.length,1);
assert.equal(input.reported,true);
input.disabled=true;
badges[0].dispatchEvent(new Event('click'));
assert.equal(input.value,'   ');
console.log('Question examples: fill, focus, edit, validation and submit passed');
"""
        result = subprocess.run(
            ["node", "-", str(ROOT / "src" / "web" / "static" / "js" / "interpretation.js")],
            input=script, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
