from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell is required")
class UnifiedOllamaLauncherTests(unittest.TestCase):
    def prepare_project(self, root: Path) -> Path:
        shutil.copyfile(ROOT / "run.ps1", root / "run.ps1")
        scripts = root / ".venv" / "Scripts"
        scripts.mkdir(parents=True)
        shutil.copyfile(sys.executable, scripts / "python.exe")
        (root / ".venv" / "pyvenv.cfg").write_text(
            f"home = {sys.base_prefix}\n"
            "include-system-site-packages = true\n"
            f"version = {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n",
            encoding="utf-8",
        )
        (root / "pre_execucao.ps1").write_text("# mocked pre-execution\n", encoding="utf-8")
        (root / "src" / "web").mkdir(parents=True)
        (root / "src" / "web" / "app.py").write_text("# mocked app\n", encoding="utf-8")
        helper_directory = root / "scripts"
        helper_directory.mkdir()
        (helper_directory / "start_ollama.ps1").write_text(
            'param([string]$Model)\n'
            '$env:LLM_PROVIDER = "ollama"\n'
            '$env:LLM_MODEL = $Model\n'
            '$env:LLM_BASE_URL = "http://localhost:11434"\n',
            encoding="utf-8",
        )
        modules = root / "modules"
        modules.mkdir()
        (modules / "uvicorn.py").write_text(
            "import os\n"
            "assert os.environ['LLM_PROVIDER'] == 'ollama'\n"
            "assert os.environ['LLM_MODEL'] == os.environ['EXPECTED_MODEL']\n"
            "assert os.environ['LLM_BASE_URL'] == 'http://localhost:11434'\n"
            "assert os.environ['APP_PROFILE'] == 'full'\n"
            "print('OLLAMA_WEB_STARTED')\n",
            encoding="utf-8",
        )
        return modules

    def test_unified_launcher_prepares_ollama_and_starts_web(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = self.prepare_project(root)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = (
                f'$env:PYTHONPATH="{modules}"; $env:EXPECTED_MODEL="custom-model"; '
                f'& "{root / "run.ps1"}" -Provider Ollama -Model custom-model -Port {port} -NoBrowser'
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 '$ErrorActionPreference="Stop"; ' + command],
                capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OLLAMA_WEB_STARTED", result.stdout)

    def test_ollama_preparation_failure_stops_web_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = self.prepare_project(root)
            (root / "scripts" / "start_ollama.ps1").write_text(
                'throw "OLLAMA_PREPARATION_FAILED"\n',
                encoding="utf-8",
            )
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 '$ErrorActionPreference="Stop"; '
                 f'$env:PYTHONPATH="{modules}"; '
                 f'& "{root / "run.ps1"}" -Provider Ollama -Port {port}'],
                capture_output=True, text=True, timeout=30,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OLLAMA_PREPARATION_FAILED", result.stderr)
        self.assertNotIn("OLLAMA_WEB_STARTED", result.stdout)


if __name__ == "__main__":
    unittest.main()
