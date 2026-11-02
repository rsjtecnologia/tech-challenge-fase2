import unittest
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nbformat

from run_notebook import _arguments, _code_cells


ROOT = Path(__file__).resolve().parents[1]


class NotebookExecutorTests(unittest.TestCase):
    def test_skip_interactive_omits_tagged_cells_only(self) -> None:
        notebook = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_code_cell("print('pipeline')"),
            nbformat.v4.new_code_cell(
                "display_widgets()",
                metadata={"tags": ["interactive"]},
            ),
            nbformat.v4.new_markdown_cell("Documentation"),
        ])

        self.assertEqual(
            [cell.source for _, cell in _code_cells(notebook, skip_interactive=True)],
            ["print('pipeline')"],
        )
        self.assertEqual(
            len(_code_cells(notebook, skip_interactive=False)),
            2,
        )

    def test_skip_interactive_argument_is_optional(self) -> None:
        normal = _arguments(["pipeline.ipynb"])
        headless = _arguments(["--skip-interactive", "pipeline.ipynb"])

        self.assertFalse(normal.skip_interactive)
        self.assertTrue(headless.skip_interactive)
        self.assertEqual(str(headless.notebook), "pipeline.ipynb")
        output = _arguments(["--output", "executed.ipynb", "pipeline.ipynb"])
        self.assertEqual(output.output, Path("executed.ipynb"))

    def test_execution_can_preserve_original_and_save_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.ipynb"
            output = Path(directory) / "outputs" / "executed.ipynb"
            notebook = nbformat.v4.new_notebook(cells=[
                nbformat.v4.new_code_cell("print('executed')"),
            ])
            nbformat.write(notebook, path)
            original = path.read_bytes()
            result = subprocess.run(
                [sys.executable, str(ROOT / "run_notebook.py"), "--output",
                 str(output), str(path)],
                capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(path.read_bytes(), original)
            executed = nbformat.read(output, as_version=4)
            self.assertIn("executed", executed.cells[0].outputs[0].text)

    def test_phase2_widget_cells_are_tagged_for_batch_runs(self) -> None:
        paths = (
            "notebooks/breast-cancer-wisconsin-data/08_export_web.ipynb",
            "notebooks/polycystic-ovary-syndrome-pcos/05_export_web.ipynb",
            "notebooks/medical-routes/01_otimizacao_rotas.ipynb",
            "notebooks/phase2/01_otimizacao_modelos.ipynb",
        )
        for relative_path in paths:
            with self.subTest(notebook=relative_path):
                notebook = nbformat.read(ROOT / relative_path, as_version=4)
                selected = _code_cells(notebook, skip_interactive=True)
                original = _code_cells(notebook, skip_interactive=False)
                self.assertEqual(len(original) - len(selected), 1)
                self.assertIn(
                    "interactive",
                    notebook.cells[-1].metadata.get("tags", []),
                )

    @unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell is required")
    def test_all_launcher_runs_every_notebook_in_dependency_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "run_notebooks.ps1", root / "run_notebooks.ps1")
            scripts = root / ".venv" / "Scripts"
            scripts.mkdir(parents=True)
            shutil.copyfile(sys.executable, scripts / "python.exe")
            (root / ".venv" / "pyvenv.cfg").write_text(
                f"home = {sys.base_prefix}\ninclude-system-site-packages = true\n",
                encoding="utf-8",
            )
            (root / "pre_execucao.ps1").write_text("# mock\n", encoding="utf-8")
            (root / "run_notebook.py").write_text(
                "import sys\nfrom pathlib import Path\n"
                "with Path('executed.txt').open('a', encoding='utf-8') as log:\n"
                "    log.write(sys.argv[-1] + '\\n')\n",
                encoding="utf-8",
            )
            expected = sorted((ROOT / "notebooks").rglob("*.ipynb"))
            for path in expected:
                target = root / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            command = [
                "powershell", "-NoProfile", "-NonInteractive", "-File",
                str(root / "run_notebooks.ps1"), "-All",
            ]
            result = subprocess.run(command, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            executed = [
                Path(line).relative_to(root).as_posix()
                for line in (root / "executed.txt").read_text(encoding="utf-8").splitlines()
            ]
            self.assertCountEqual(executed, [p.relative_to(ROOT).as_posix() for p in expected])
            self.assertEqual(len(executed), len(set(executed)))
            ordered = [
                p.relative_to(ROOT).as_posix()
                for folder in (
                    "breast-cancer-wisconsin-data",
                    "polycystic-ovary-syndrome-pcos",
                    "phase2",
                    "medical-routes",
                )
                for p in sorted((ROOT / "notebooks" / folder).glob("*.ipynb"))
            ]
            self.assertEqual(executed, ordered)
            (root / "notebooks" / "unregistered.ipynb").write_text("{}", encoding="utf-8")
            rejected = subprocess.run(command, capture_output=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(b"unregistered.ipynb", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
