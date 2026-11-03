from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell is required")
class UnifiedLauncherTests(unittest.TestCase):
    def prepare_project(self, root: Path, expected_environment: dict[str, str]) -> Path:
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
        (root / "scripts").mkdir()
        shutil.copyfile(ROOT / "scripts" / "open_browser.py", root / "scripts" / "open_browser.py")
        modules = root / "modules"
        modules.mkdir()
        web = modules / "src" / "web"
        web.mkdir(parents=True)
        (modules / "src" / "__init__.py").touch()
        (web / "__init__.py").touch()
        (web / "preflight.py").write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "print('PREFLIGHT_CHECKED', flush=True)\n"
            "if '--llm-only' in sys.argv:\n"
            "    print('ISOLATED_LLM_CHECK', flush=True)\n"
            "if Path('preflight_failure').exists():\n"
            "    print('Arquivo ausente: missing.json', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "if Path('preflight_provider_failure').exists():\n"
            "    print('Teste LLM falhou: HTTP 503', file=sys.stderr)\n"
            "    sys.exit(2)\n",
            encoding="utf-8",
        )
        checks = "\n".join(
            f"assert os.environ.get({name!r}) == {value!r}, {('environment: ' + name)!r}"
            for name, value in expected_environment.items()
        )
        (modules / "uvicorn.py").write_text(
            "import os, sys\n"
            + checks
            + "\nprint('API_WEB_STARTED')\n",
            encoding="utf-8",
        )
        return modules

    def test_single_launcher_configures_provider_and_starts_web(self) -> None:
        for no_browser in (False, True):
            for provider_failure in (False, True):
                with self.subTest(no_browser=no_browser, provider_failure=provider_failure):
                    self.assert_launcher_starts_web(no_browser, provider_failure)

    def assert_launcher_starts_web(self, no_browser: bool, provider_failure: bool) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = self.prepare_project(root, {
                "LLM_PROVIDER": "openai_compatible",
                "LLM_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai",
                "LLM_MODEL": "gemini-3.1-flash-lite",
                "LLM_API_KEY": "fake-test-key",
                "LLM_TIMEOUT_SECONDS": "45",
                "LLM_REASONING_EFFORT": "minimal",
                "APP_PROFILE": "full",
            })
            if provider_failure:
                (root / "preflight_provider_failure").touch()
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = (
                f'$env:PYTHONPATH="{modules}"; '
                '$env:GEMINI_API_KEY="fake-test-key"; '
                '$env:LLM_API_KEY="previous-key"; '
                'function Invoke-RestMethod { throw "Unexpected separate provider request" }; '
                'function Start-Process { param($FilePath,$ArgumentList,[switch]$NoNewWindow,[switch]$PassThru) '
                'if (-not (Test-Path -LiteralPath $FilePath)) { throw "python path" }; '
                f'if ($ArgumentList[0] -ne \'"{root / "scripts" / "open_browser.py"}"\') {{ throw "browser script" }}; '
                f'if ($ArgumentList[1] -ne "--port" -or $ArgumentList[2] -ne "{port}") {{ throw "browser port" }}; '
                f'if (($ArgumentList -contains "--no-browser") -ne ${str(no_browser).lower()}) {{ throw "browser flag" }}; '
                'Write-Host "BROWSER_HELPER_STARTED"; return [pscustomobject]@{HasExited=$true} }; '
                f'& "{root / "run.ps1"}" -Port {port} {"-NoBrowser" if no_browser else ""}; '
                'if ($env:LLM_API_KEY -ne "previous-key") { throw "env not restored" }'
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 '$ErrorActionPreference="Stop"; ' + command],
                capture_output=True, text=True, timeout=60,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("API_WEB_STARTED", result.stdout)
        self.assertIn("PREFLIGHT_CHECKED", result.stdout)
        self.assertLess(result.stdout.index("PREFLIGHT_CHECKED"), result.stdout.index("API_WEB_STARTED"))
        self.assertIn("BROWSER_HELPER_STARTED", result.stdout)
        if provider_failure:
            self.assertIn("Teste LLM falhou: HTTP 503", result.stdout)
            self.assertIn("LLM nao validada", result.stdout)
        self.assertNotIn("fake-test-key", result.stdout + result.stderr)

    def test_failed_preflight_prevents_server_and_browser_start_and_restores_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = self.prepare_project(root, {})
            (root / "preflight_failure").touch()
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = (
                f'$env:PYTHONPATH="{modules}"; $env:GEMINI_API_KEY="fake-test-key"; '
                '$env:LLM_API_KEY="previous-key"; '
                'function Invoke-RestMethod { return @{data=@(@{id="models/gemini-3.1-flash-lite"})} }; '
                'function Start-Process { throw "BROWSER_MUST_NOT_START" }; '
                f'try {{ & "{root / "run.ps1"}" -Port {port} }} '
                'catch { Write-Host "EXPECTED_STARTUP_FAILURE"; Write-Host $_.Exception.Message; '
                'if ($env:LLM_API_KEY -ne "previous-key") { throw "env not restored" }; exit 7 }'
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 '$ErrorActionPreference="Stop"; ' + command],
                capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("PREFLIGHT_CHECKED", result.stdout)
        self.assertIn("EXPECTED_STARTUP_FAILURE", result.stdout)
        self.assertNotIn("API_WEB_STARTED", result.stdout)
        self.assertNotIn("BROWSER_MUST_NOT_START", result.stdout + result.stderr)
        self.assertIn("Arquivo ausente: missing.json", result.stdout)

    def test_busy_port_fails_before_provider_or_web_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            root = Path(directory)
            modules = self.prepare_project(root, {})
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f'$env:PYTHONPATH="{modules}"; $env:GEMINI_API_KEY="not-requested"; '
                 f'& "{root / "run.ps1"}" -Port {port}'],
                capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ocupada", result.stderr)
            self.assertNotIn("API_WEB_STARTED", result.stdout)
            self.assertNotIn("Chave API", result.stdout)

    def test_isolated_check_works_with_busy_port_and_does_not_start_services(self) -> None:
        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            root = Path(directory)
            modules = self.prepare_project(root, {})
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            command = (
                f'$env:PYTHONPATH="{modules}"; $env:GEMINI_API_KEY="fake-test-key"; '
                '$env:LLM_API_KEY="previous-key"; '
                'function Start-Process { throw "BROWSER_MUST_NOT_START" }; '
                f'& "{root / "run.ps1"}" -Port {port} -CheckLLMOnly; '
                'if ($env:LLM_API_KEY -ne "previous-key") { throw "env not restored" }'
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 '$ErrorActionPreference="Stop"; ' + command],
                capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ISOLATED_LLM_CHECK", result.stdout)
        self.assertNotIn("API_WEB_STARTED", result.stdout)
        self.assertNotIn("BROWSER_MUST_NOT_START", result.stdout + result.stderr)
        self.assertNotIn("fake-test-key", result.stdout + result.stderr)

    def test_isolated_check_fails_explicitly_without_launching_web(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = self.prepare_project(root, {})
            (root / "preflight_provider_failure").touch()
            command = (
                f'$env:PYTHONPATH="{modules}"; $env:GEMINI_API_KEY="fake-test-key"; '
                '$env:LLM_API_KEY="previous-key"; '
                'function Start-Process { throw "BROWSER_MUST_NOT_START" }; '
                f'try {{ & "{root / "run.ps1"}" -CheckLLMOnly }} '
                'catch { Write-Host $_.Exception.Message; '
                'if ($env:LLM_API_KEY -ne "previous-key") { throw "env not restored" }; exit 7 }'
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 '$ErrorActionPreference="Stop"; ' + command],
                capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("HTTP 503", result.stdout)
        self.assertNotIn("API_WEB_STARTED", result.stdout)
        self.assertNotIn("Teste isolado concluido", result.stdout)
        self.assertNotIn("BROWSER_MUST_NOT_START", result.stdout + result.stderr)

    def test_invalid_provider_options_fail_before_environment_setup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "run.ps1", root / "run.ps1")
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-File",
                 str(root / "run.ps1"), "-Provider", "OpenAI", "-QwenRegion", "us-east-1"],
                capture_output=True, text=True, timeout=30,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("QwenRegion", result.stderr)


if __name__ == "__main__":
    unittest.main()
