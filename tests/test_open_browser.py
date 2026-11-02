from contextlib import redirect_stderr
from io import BytesIO, StringIO
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
import webbrowser

from scripts.open_browser import main, open_when_ready


class OpenBrowserTests(unittest.TestCase):
    def test_waits_for_api_and_opens_requested_port_once(self) -> None:
        with patch("scripts.open_browser.urlopen") as request, patch(
            "scripts.open_browser.webbrowser.open", return_value=True
        ) as browser, patch("scripts.open_browser.time.sleep") as sleep:
            response = request.return_value
            response.__enter__.return_value.status = 200
            unavailable = HTTPError(
                "http://127.0.0.1:8021/healthz", 503, "Starting", {}, BytesIO()
            )
            request.side_effect = [
                URLError("Connection refused"), unavailable, TimeoutError(), response,
            ]
            self.assertTrue(open_when_ready(8021))
            self.assertEqual(request.call_count, 4)
            self.assertEqual(sleep.call_count, 3)
            self.assertTrue(unavailable.fp.closed)
            self.assertTrue(all(
                call.args[0] == "http://127.0.0.1:8021/healthz"
                for call in request.call_args_list
            ))
            browser.assert_called_once_with("http://127.0.0.1:8021/", new=2)

    def test_timeout_does_not_open_browser_and_reports_manual_url(self) -> None:
        with patch("scripts.open_browser.urlopen", side_effect=URLError("offline")) as request, patch(
            "scripts.open_browser.webbrowser.open"
        ) as browser, patch("scripts.open_browser.time.sleep"), patch(
            "scripts.open_browser.time.monotonic", side_effect=[0, 0, 60, 60]
        ), redirect_stderr(StringIO()) as errors:
            self.assertFalse(open_when_ready(8020))
            request.assert_called_once()
            browser.assert_not_called()
            self.assertIn("A API nao respondeu", errors.getvalue())
            self.assertIn("http://127.0.0.1:8020/", errors.getvalue())

    def test_browser_failure_is_explicit(self) -> None:
        for result in (False, webbrowser.Error("unavailable"), OSError("unavailable")):
            with self.subTest(result=result), patch(
                "scripts.open_browser.urlopen"
            ) as request, patch(
                "scripts.open_browser.webbrowser.open",
                **({"side_effect": result} if isinstance(result, Exception)
                   else {"return_value": result}),
            ), redirect_stderr(StringIO()) as errors:
                request.return_value.__enter__.return_value.status = 200
                self.assertFalse(open_when_ready(8020))
                self.assertIn("http://127.0.0.1:8020/", errors.getvalue())

    def test_cli_passes_port_and_reports_exit_status(self) -> None:
        for opened, expected in ((True, 0), (False, 1)):
            with self.subTest(opened=opened), patch(
                "sys.argv", ["open_browser.py", "--port", "8021"]
            ), patch("scripts.open_browser.open_when_ready", return_value=opened) as open_browser:
                self.assertEqual(main(), expected)
                open_browser.assert_called_once_with(8021)

    def test_invalid_port_is_rejected_before_network(self) -> None:
        for port in ("0", "65536", "invalid"):
            with self.subTest(port=port), patch(
                "sys.argv", ["open_browser.py", "--port", port]
            ), patch("scripts.open_browser.open_when_ready") as open_browser, redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                open_browser.assert_not_called()
