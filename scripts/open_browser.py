"""Open the local dashboard only after the API starts responding."""

import argparse
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import webbrowser


def open_when_ready(port: int, timeout: float = 60) -> bool:
    url = f"http://127.0.0.1:{port}/"
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            with urlopen(f"{url}healthz", timeout=min(1, remaining)) as response:
                ready = response.status == 200
        except HTTPError as error:
            error.close()
            ready = False
        except (URLError, TimeoutError, ConnectionError):
            ready = False
        if ready:
            try:
                opened = webbrowser.open(url, new=2)
            except (webbrowser.Error, OSError) as error:
                print(
                    f"Nao foi possivel abrir o navegador ({type(error).__name__}). "
                    f"Acesse manualmente: {url}",
                    file=sys.stderr,
                )
                return False
            if not opened:
                print(
                    f"Nenhum navegador disponivel. Acesse manualmente: {url}",
                    file=sys.stderr,
                )
            return opened
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    print(
        f"A API nao respondeu em {timeout:g}s. Confira o terminal do servidor "
        f"e acesse manualmente quando estiver disponivel: {url}",
        file=sys.stderr,
    )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port deve estar entre 1 e 65535.")
    return 0 if open_when_ready(args.port) else 1


if __name__ == "__main__":
    sys.exit(main())
