from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts.load_autoscaling import run_load
import src.web.app as web_app


ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "infra" / "autoscaling"


class AutoscalingConfigurationTests(unittest.TestCase):
    def test_hpa_targets_deployment_with_cpu_requests_and_bounded_scaling(self) -> None:
        manifest = json.loads((INFRA / "kubernetes.json").read_text())
        resources = {item["kind"]: item for item in manifest["items"]}
        deployment = resources["Deployment"]
        hpa = resources["HorizontalPodAutoscaler"]
        service = resources["Service"]
        self.assertEqual(hpa["apiVersion"], "autoscaling/v2")
        self.assertEqual(
            hpa["spec"]["scaleTargetRef"]["name"], deployment["metadata"]["name"]
        )
        self.assertNotIn("replicas", deployment["spec"])
        self.assertEqual(hpa["spec"]["minReplicas"], 1)
        self.assertEqual(hpa["spec"]["maxReplicas"], 4)
        metric = hpa["spec"]["metrics"][0]["resource"]
        self.assertEqual(metric["name"], "cpu")
        self.assertEqual(metric["target"]["averageUtilization"], 60)
        pod = deployment["spec"]["template"]
        self.assertEqual(service["spec"]["selector"], pod["metadata"]["labels"])
        self.assertEqual(service["spec"]["type"], "ClusterIP")
        api = pod["spec"]["containers"][0]
        self.assertEqual(api["resources"]["requests"]["cpu"], "100m")
        self.assertEqual(api["resources"]["limits"]["cpu"], "500m")
        self.assertEqual(api["readinessProbe"]["httpGet"]["path"], "/readyz")
        self.assertEqual(api["livenessProbe"]["httpGet"]["path"], "/healthz")
        self.assertTrue(api["securityContext"]["readOnlyRootFilesystem"])
        behavior = hpa["spec"]["behavior"]
        self.assertGreater(behavior["scaleDown"]["stabilizationWindowSeconds"], 0)

    def test_load_job_uses_same_image_and_balanced_service_not_port_forward(self) -> None:
        manifest = json.loads((INFRA / "kubernetes.json").read_text())
        deployment = next(item for item in manifest["items"] if item["kind"] == "Deployment")
        api = deployment["spec"]["template"]["spec"]["containers"][0]
        job = json.loads((INFRA / "load-test-job.json").read_text())
        load = job["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(api["image"], load["image"])
        self.assertIn("http://diagnostics-api:8000/api/otimizacao/breast_cancer", load["args"])
        self.assertEqual(job["spec"]["backoffLimit"], 0)
        self.assertLessEqual(job["spec"]["activeDeadlineSeconds"], 300)

    def test_logging_configuration_enables_application_latency_logs(self) -> None:
        config = json.loads((INFRA / "logging.json").read_text())
        self.assertFalse(config["disable_existing_loggers"])
        self.assertEqual(config["root"]["level"], "INFO")
        self.assertEqual(config["handlers"]["console"]["stream"], "ext://sys.stdout")


class DiagnosticsProfileTests(unittest.TestCase):
    def test_readiness_excludes_mutable_logistics_and_requires_web_artifacts(self) -> None:
        with patch.object(web_app, "APP_PROFILE", "diagnostics"), patch.object(
            Path, "is_file", return_value=False
        ), TestClient(web_app.app) as client:
            response = client.get("/readyz")
        self.assertEqual(response.status_code, 503)
        missing = response.json()["missing_files"]
        self.assertEqual(len(missing), 9)
        self.assertFalse(any("phase2-logistics" in path for path in missing))
        self.assertTrue(any("predicoes_longas.parquet" in path for path in missing))
        with patch.object(web_app, "APP_PROFILE", "diagnostics"), patch.object(
            Path, "is_file", return_value=True
        ), TestClient(web_app.app) as client:
            self.assertEqual(client.get("/readyz").status_code, 200)

    def test_diagnostics_profile_does_not_expose_logistics_writes_or_navigation(self) -> None:
        code = """
from fastapi.testclient import TestClient
from src.web.app import app
with TestClient(app) as client:
    assert client.get('/healthz').status_code == 200
    assert client.get('/rotas').status_code == 404
    assert client.post('/api/rotas/otimizar', json={}).status_code == 404
    for path in ('/', '/pcos'):
        page = client.get(path)
        assert page.status_code == 200
        assert 'href="/rotas"' not in page.text
    assert '/api/interpretacao' in client.get('/openapi.json').json()['paths']
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={**os.environ, "APP_PROFILE": "diagnostics"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_full_profile_preserves_logistics_navigation(self) -> None:
        with TestClient(web_app.app) as client:
            for path in ("/", "/pcos"):
                self.assertIn('href="/rotas"', client.get(path).text)
            self.assertEqual(client.get("/rotas").status_code, 200)

    def test_full_profile_requires_all_dashboard_files_and_logistics_plan(self) -> None:
        with patch.object(web_app, "APP_PROFILE", "full"), patch.object(
            Path, "is_file", return_value=False
        ), TestClient(web_app.app) as client:
            response = client.get("/readyz")
        self.assertEqual(response.status_code, 503)
        missing = response.json()["missing_files"]
        self.assertEqual(len(missing), 10)
        self.assertTrue(any("phase2-logistics" in path for path in missing))
        for filename in (
            "graficos.json", "graficos_web.json",
            "amostras_comparativo.parquet", "predicoes_longas.parquet",
        ):
            with self.subTest(filename=filename), patch.object(
                web_app, "APP_PROFILE", "full"
            ), patch.object(
                Path, "is_file", lambda path: path.name != filename
            ), TestClient(web_app.app) as client:
                response = client.get("/readyz")
                self.assertEqual(response.status_code, 503)
                self.assertTrue(all(
                    Path(path).name == filename
                    for path in response.json()["missing_files"]
                ))

    def test_full_profile_is_ready_when_all_required_files_exist(self) -> None:
        with patch.object(web_app, "APP_PROFILE", "full"), patch.object(
            Path, "is_file", return_value=True
        ), TestClient(web_app.app) as client:
            self.assertEqual(client.get("/readyz").status_code, 200)


@unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell is required")
class AutoscalingOrchestrationTests(unittest.TestCase):
    def test_script_requires_scale_out_scale_in_and_successful_load(self) -> None:
        code = r"""
$ErrorActionPreference = "Stop"
function global:Start-Sleep {}
function global:kubectl {
    $global:LASTEXITCODE = 0
    $command = $args -join " "
    if (-not $command.StartsWith("--context mock ")) { throw "Missing explicit context" }
    if ($command -match "get deployment") {
        $global:observations++
        $count = 1
        if ($global:scenario -eq "success" -and $global:observations -eq 2) { $count = 2 }
        return ('{"spec":{"replicas":' + $count + '},"status":{"readyReplicas":' + $count + '}}')
    }
    if ($command -match "get hpa") { return '{"spec":{"minReplicas":1}}' }
    if ($command -match "get job") {
        if ($global:scenario -eq "load-failed") { return '{"status":{"failed":1}}' }
        return '{"status":{"succeeded":1}}'
    }
    if ($command -match "delete job") { $global:deletions++ }
}
foreach ($scenario in @("success", "no-scale-out", "load-failed")) {
    $global:scenario = $scenario
    $global:observations = 0
    $global:deletions = 0
    $failed = $false
    try { & .\scripts\test_autoscaling.ps1 -Context mock }
    catch { $failed = $true }
    if ($failed -ne ($scenario -ne "success")) { throw "Wrong outcome for $scenario" }
    if ($global:deletions -ne 2) { throw "Load Job was not cleaned up in $scenario" }
}
"""
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS:", result.stdout)


class LoadGeneratorTests(unittest.TestCase):
    def _run_server(self, status: int) -> dict:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b'{"status":"test"}')

            def log_message(self, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            return run_load(f"http://127.0.0.1:{server.server_port}/api/test", 0.15, 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_successful_load_reports_measured_requests_and_latency(self) -> None:
        result = self._run_server(200)
        self.assertGreater(result["requests"], 0)
        self.assertEqual(result["requests"], result["successful_requests"])
        self.assertEqual(result["failed_requests"], 0)
        self.assertGreater(result["p95_latency_ms"], 0)
        self.assertGreater(result["requests_per_second"], 0)

    def test_http_failures_are_reported_not_counted_as_success(self) -> None:
        result = self._run_server(503)
        self.assertGreater(result["failed_requests"], 0)
        self.assertEqual(result["successful_requests"], 0)
        self.assertEqual(result["errors"]["HTTP 503"], result["requests"])

    def test_unreachable_server_is_reported(self) -> None:
        with patch("scripts.load_autoscaling.http.client.HTTPConnection") as connection, patch(
            "scripts.load_autoscaling.time.monotonic",
            side_effect=[0, 0, 0, 0.001, 1, 1],
        ):
            connection.return_value.request.side_effect = ConnectionRefusedError
            result = run_load("http://127.0.0.1:1/test", 0.01, 1)
        self.assertGreater(result["failed_requests"], 0)
        self.assertEqual(result["successful_requests"], 0)
        self.assertIn("ConnectionRefusedError", result["errors"])

    def test_load_limits_and_urls_are_validated(self) -> None:
        for url, seconds, concurrency in (
            ("https://127.0.0.1", 1, 1),
            ("http://user:secret@127.0.0.1", 1, 1),
            ("http://127.0.0.1", 0, 1),
            ("http://127.0.0.1", 301, 1),
            ("http://127.0.0.1", 1, 65),
        ):
            with self.subTest(url=url, seconds=seconds, concurrency=concurrency):
                with self.assertRaises(ValueError):
                    run_load(url, seconds, concurrency)


if __name__ == "__main__":
    unittest.main()
