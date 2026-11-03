"""Validate local artifacts and actual LLM generation before server startup."""

import json
import sys

from src.optimization.llm import (
    LLMConfigurationError,
    LLMProviderError,
    request_chat_completion,
)
from src.web.app import readiness


def main() -> int:
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
    try:
        content = request_chat_completion(
            "Return only the JSON object requested by the user.",
            'Return exactly {"ok":true}.',
            json_mode=True,
            max_tokens=128,
        )
    except (LLMConfigurationError, LLMProviderError) as error:
        print(
            f"Teste LLM falhou: {error}\n"
            "O servidor ainda nao foi iniciado. Confira o provedor/modelo; "
            "reiniciar a API nao corrige indisponibilidade externa.",
            file=sys.stderr,
        )
        return 1
    try:
        result = json.loads(content)
        valid = isinstance(result, dict) and result == {"ok": True} and result["ok"] is True
    except json.JSONDecodeError:
        valid = False
    if not valid:
        print("Teste LLM falhou: resposta diferente do JSON esperado.", file=sys.stderr)
        return 1
    print(
        "Geracao LLM: OK neste momento. O provedor pode ficar indisponivel depois.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
