from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import unittest
from unittest.mock import patch

from fastapi.responses import JSONResponse

from src.optimization.llm import LLMConfigurationError, LLMProviderError
from src.web.preflight import main


class StartupPreflightTests(unittest.TestCase):
    def test_isolated_llm_test_skips_artifacts_and_never_prints_key(self) -> None:
        with patch("src.web.preflight.readiness") as readiness, patch(
            "src.web.preflight._settings",
            return_value=(
                "https://generativelanguage.googleapis.com/v1beta/openai",
                "gemini-3.1-flash-lite", "secret-test-key", 45,
            ),
        ), patch(
            "src.web.preflight.request_chat_completion", return_value='{"ok":true}'
        ), redirect_stdout(StringIO()) as output:
            self.assertEqual(main(llm_only=True), 0)
            readiness.assert_not_called()
            self.assertIn("Modelo: gemini-3.1-flash-lite", output.getvalue())
            self.assertIn("timeout: 45s", output.getvalue())
            self.assertNotIn("secret-test-key", output.getvalue())

    def test_preflight_uses_http_transport_and_handles_provider_outages(self) -> None:
        for statuses, expected in (([503, 200], 0), ([503, 503, 503], 2), ([401], 2)):
            with self.subTest(statuses=statuses):
                requests = []

                class Provider(BaseHTTPRequestHandler):
                    def do_POST(self) -> None:
                        requests.append((
                            self.path,
                            json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                        ))
                        status = statuses[len(requests) - 1]
                        body = json.dumps({
                            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
                        }).encode() if status == 200 else b'{"error":"unavailable"}'
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)

                    def log_message(self, format: str, *args: object) -> None:
                        pass

                server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    with patch.dict(os.environ, {
                        **{name: value for name, value in os.environ.items()
                           if not name.startswith("LLM_")},
                        "LLM_PROVIDER": "openai_compatible",
                        "LLM_MODEL": "synthetic-test-model",
                        "LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                        "LLM_TIMEOUT_SECONDS": "10",
                    }, clear=True), patch(
                        "src.web.preflight.readiness", return_value=JSONResponse({"status": "ready"})
                    ), redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                        self.assertEqual(main(), expected)
                        self.assertEqual(len(requests), len(statuses))
                        self.assertTrue(all(path == "/v1/chat/completions" for path, _ in requests))
                        self.assertTrue(all(
                            body["messages"][1]["content"] == 'Return exactly {"ok":true}.'
                            for _, body in requests
                        ))
                        if expected:
                            self.assertIn(f"HTTP {statuses[-1]}", errors.getvalue())
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

    def test_missing_artifacts_fail_before_provider_request(self) -> None:
        with patch("src.web.preflight.readiness", return_value=JSONResponse(
            status_code=503, content={"missing_files": ["missing.json"]}
        )), patch("src.web.preflight.request_chat_completion") as provider, redirect_stderr(
            StringIO()
        ) as errors:
            self.assertEqual(main(), 1)
            provider.assert_not_called()
            self.assertIn("missing.json", errors.getvalue())
            self.assertIn("run_notebooks.ps1", errors.getvalue())

    def test_real_generation_is_required_and_only_synthetic_context_is_sent(self) -> None:
        with patch("src.web.preflight.readiness", return_value=JSONResponse(
            {"status": "ready"}
        )), patch(
            "src.web.preflight.request_chat_completion", return_value='{"ok":true}'
        ) as provider, redirect_stdout(StringIO()) as output:
            self.assertEqual(main(), 0)
            provider.assert_called_once_with(
                "Return only the JSON object requested by the user.",
                'Return exactly {"ok":true}.', json_mode=True, max_tokens=128,
            )
            self.assertIn("Geracao LLM: OK neste momento", output.getvalue())

    def test_generation_failure_is_explicit_and_never_reported_as_success(self) -> None:
        for error in (
            LLMProviderError("HTTP 503"), LLMProviderError("HTTP 401"),
            LLMProviderError("timeout"), LLMConfigurationError("invalid configuration"),
        ):
            with self.subTest(error=error), patch(
                "src.web.preflight.readiness", return_value=JSONResponse({"status": "ready"})
            ), patch(
                "src.web.preflight.request_chat_completion", side_effect=error
            ), redirect_stdout(StringIO()) as output, redirect_stderr(StringIO()) as errors:
                self.assertEqual(main(), 1 if isinstance(error, LLMConfigurationError) else 2)
                self.assertIn(str(error), errors.getvalue())
                if isinstance(error, LLMProviderError):
                    self.assertIn("A API local pode iniciar", errors.getvalue())
                self.assertNotIn("Geracao LLM: OK", output.getvalue())

    def test_invalid_probe_content_fails_without_echoing_content(self) -> None:
        for content in ('secret-response', '{"ok":false}', '{"ok":1}', '[]', 'null'):
            with self.subTest(content=content), patch(
                "src.web.preflight.readiness", return_value=JSONResponse({"status": "ready"})
            ), patch(
                "src.web.preflight.request_chat_completion", return_value=content
            ), redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                self.assertEqual(main(), 2)
                self.assertIn("JSON esperado", errors.getvalue())
                self.assertNotIn(content, errors.getvalue())
