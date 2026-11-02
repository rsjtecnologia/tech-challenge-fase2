"""Executa um notebook .ipynb com progresso por célula de código."""

from __future__ import annotations

import asyncio
import argparse
import sys
import warnings
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from typing import Any

import nbformat
from nbclient import NotebookClient

# No Windows (< 3.14), o ProactorEventLoop do asyncio não cobre bem o zmq do kernel.
if sys.platform.startswith("win") and sys.version_info < (3, 14):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

warnings.filterwarnings(
    "ignore",
    message=".*Proactor event loop does not implement add_reader.*",
)


class ProgressNotebookClient(NotebookClient):
    def process_message(
        self, msg: dict[str, Any], cell: nbformat.NotebookNode, cell_index: int
    ) -> nbformat.NotebookNode | None:
        if msg["msg_type"] == "stream":
            stream = sys.stderr if msg["content"]["name"] == "stderr" else sys.stdout
            print(msg["content"]["text"], end="", file=stream, flush=True)
        return super().process_message(msg, cell, cell_index)


def _report_wait(stop: Event, started: float, label: str) -> None:
    while not stop.wait(30):
        elapsed = monotonic() - started
        print(
            f"{label} em execução há {elapsed:.0f}s; aguardando o kernel "
            "(não confirma progresso do treinamento).",
            flush=True,
        )


def _preview(source: str, max_len: int = 64) -> str:
    line = (source or "").strip().splitlines()
    text = line[0] if line else "(vazia)"
    text = " ".join(text.split())
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa um notebook .ipynb com progresso por célula de código."
    )
    parser.add_argument(
        "--skip-interactive",
        action="store_true",
        help="Pula células marcadas com a tag 'interactive' (widgets/API local).",
    )
    parser.add_argument("notebook", type=Path, help="Caminho do notebook .ipynb.")
    parser.add_argument(
        "--output", type=Path,
        help="Salvar a execução neste arquivo, preservando o notebook original.",
    )
    return parser.parse_args(argv)


def _code_cells(
    nb: nbformat.NotebookNode,
    skip_interactive: bool,
) -> list[tuple[int, nbformat.NotebookNode]]:
    return [
        (index, cell)
        for index, cell in enumerate(nb.cells)
        if cell.cell_type == "code"
        and not (
            skip_interactive
            and "interactive" in cell.get("metadata", {}).get("tags", [])
        )
    ]


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    path = args.notebook.resolve()
    if not path.is_file():
        print(f"Notebook não encontrado: {path}", file=sys.stderr)
        return 1

    nb = nbformat.read(path, as_version=4)
    selected_cells = _code_cells(nb, args.skip_interactive)
    code_total = len(selected_cells)
    skipped_total = sum(
        1 for cell in nb.cells
        if cell.cell_type == "code"
        and "interactive" in cell.get("metadata", {}).get("tags", [])
    ) if args.skip_interactive else 0
    print(f"Notebook: {path.name} ({code_total} células de código)")
    if skipped_total:
        print(f"Células interativas puladas: {skipped_total}")
    print("")

    notebook_dir = path.parent
    print(f"Diretório: {notebook_dir}")
    print("")

    client = ProgressNotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_dir)}},
    )
    done = 0

    with client.setup_kernel():
        for index, cell in selected_cells:
            done += 1
            label = f"[{done}/{code_total}]"
            print(f"{label} {_preview(cell.source)}", flush=True)
            started = monotonic()
            stop = Event()
            reporter = Thread(target=_report_wait, args=(stop, started, label), daemon=True)
            reporter.start()
            try:
                client.execute_cell(cell, index)
            finally:
                stop.set()
                reporter.join()
            print(f"{label} concluída em {monotonic() - started:.1f}s", flush=True)

    output_path = args.output.resolve() if args.output else path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output_path)
    print("")
    print(f"Salvo: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
