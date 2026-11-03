"""Validate local artifacts and actual LLM generation before server startup."""

import argparse
import json
import sys
import time
from urllib.parse import urlsplit

from src.optimization.llm import (
    LLMConfigurationError,
    LLMProviderError,
    _settings,
    request_chat_completion,
)
from src.web.app import readiness


def main(*, llm_only: bool = False) -> int:
    if not llm_only:
        response = readiness()
        if response.status_code != 200:
            missing = json.loads(response.body)["missing_files"]
            print(
                "Dados locais incompletos. Execute .\\run_notebooks.ps1 antes de iniciar.",
                file=sys.stderr,
            )
            for path in missing:
                print(f"Arquivo ausente: {path}", file=sys.stderr)
            return 1
        print("Dados locais: OK.", flush=True)
    print(
        "Testando geracao LLM com mensagem sintetica, sem dados do projeto. "
        "Provedores online podem cobrar esta chamada.",
        flush=True,
    )
    started = time.monotonic()
    try:
        if llm_only:
            base_url, model, _, timeout = _settings()
            print(
                f"Modelo: {model}; host: {urlsplit(base_url).hostname}; timeout: {timeout}s.",
                flush=True,
            )
        content = request_chat_completion(
            "Return only the JSON object requested by the user.",
            'Return exactly {"ok":true}.',
            json_mode=True,
            max_tokens=128,
        )
    except LLMConfigurationError as error:
        print(
            f"Configuracao LLM invalida: {error}",
            file=sys.stderr,
        )
        return 1
    except LLMProviderError as error:
        print(
            f"Teste LLM falhou apos {time.monotonic() - started:.1f}s: {error}\n"
            "A API local pode iniciar, mas a geracao LLM nao foi validada. "
            "Confira o provedor/modelo; reiniciar a API nao corrige "
            "indisponibilidade externa.",
            file=sys.stderr,
        )
        return 2
    try:
        result = json.loads(content)
        valid = isinstance(result, dict) and result == {"ok": True} and result["ok"] is True
    except json.JSONDecodeError:
        valid = False
    if not valid:
        print("Teste LLM falhou: resposta diferente do JSON esperado.", file=sys.stderr)
        return 2
    print(
        "Geracao LLM: OK neste momento. O provedor pode ficar indisponivel depois.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-only", action="store_true")
    args = parser.parse_args()
    sys.exit(main(llm_only=args.llm_only))
