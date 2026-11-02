"""Bounded, process-local cache for successful LLM interpretations."""

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable


LOGGER = logging.getLogger(__name__)
PROMPT_REVISION = hashlib.sha256(
    Path(__file__).with_name("llm.py").read_bytes()
).hexdigest()


def interpretation_key(context: dict[str, Any], question: str | None) -> str:
    payload = {
        "context": context,
        "question": question.strip() if question is not None else None,
        "settings": {
            name: value for name, value in os.environ.items()
            if name.startswith("LLM_")
        },
        "prompt_revision": PROMPT_REVISION,
    }
    serialized = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class InterpretationCache:
    def __init__(
        self,
        max_entries: int = 64,
        ttl_seconds: float = 3600,
        clock: Callable[[], float] = time.monotonic,
        is_cacheable: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        if max_entries < 1 or ttl_seconds <= 0:
            raise ValueError("Cache capacity and TTL must be positive.")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.is_cacheable = is_cacheable or (
            lambda response: response.get("source") == "llm"
            and response.get("quality_evaluation", {}).get("passed") is True
        )
        self._entries: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()
        self._generation_locks = [threading.Lock() for _ in range(16)]

    def get_or_generate(
        self, key: str, generate: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        # Bounded lock striping coalesces identical in-flight requests.
        with self._generation_locks[int(key, 16) % len(self._generation_locks)]:
            now = self.clock()
            with self._lock:
                expired = [
                    stored_key for stored_key, (created, _) in self._entries.items()
                    if now - created >= self.ttl_seconds
                ]
                for stored_key in expired:
                    del self._entries[stored_key]
                entry = self._entries.get(key)
                if entry is not None:
                    created, response = entry
                    self._entries.move_to_end(key)
                    result = deepcopy(response)
                    result["cache"] = {
                        "hit": True,
                        "age_seconds": round(now - created, 3),
                        "ttl_seconds": self.ttl_seconds,
                    }
                    LOGGER.info("interpretation_cache status=hit")
                    return result

            response = generate()
            cacheable = self.is_cacheable(response)
            if cacheable:
                with self._lock:
                    self._entries[key] = (self.clock(), deepcopy(response))
                    while len(self._entries) > self.max_entries:
                        self._entries.popitem(last=False)
            result = deepcopy(response)
            result["cache"] = {
                "hit": False,
                "stored": cacheable,
                "ttl_seconds": self.ttl_seconds,
            }
            LOGGER.info(
                "interpretation_cache status=%s", "miss" if cacheable else "skipped"
            )
            return result
