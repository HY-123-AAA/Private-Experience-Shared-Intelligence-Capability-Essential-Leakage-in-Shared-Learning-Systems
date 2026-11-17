from __future__ import annotations

import ast
import re
from collections.abc import Callable


SERVICE_FAILURE = "<SERVICE_FAILURE>"
Verifier = Callable[[str], bool]


def _extract_code(completion: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", completion, flags=re.S | re.I)
    return (match.group(1) if match else completion).strip()


def canonicalize_python(source: str) -> str:
    return ast.unparse(ast.parse(source))


class CandidateReturnService:
    def respond(self, completion: str, verifier: Verifier) -> str:
        if not isinstance(completion, str):
            return SERVICE_FAILURE
        try:
            canonical = canonicalize_python(_extract_code(completion))
            approved = bool(verifier(canonical))
        except Exception:
            return SERVICE_FAILURE
        return canonical if approved else SERVICE_FAILURE

