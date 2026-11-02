import unittest
from pathlib import Path

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

    def test_phase2_widget_cells_are_tagged_for_batch_runs(self) -> None:
        paths = (
            "notebooks/breast-cancer-wisconsin-data/08_export_web.ipynb",
            "notebooks/polycystic-ovary-syndrome-pcos/05_export_web.ipynb",
            "notebooks/medical-routes/01_otimizacao_rotas.ipynb",
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


if __name__ == "__main__":
    unittest.main()
