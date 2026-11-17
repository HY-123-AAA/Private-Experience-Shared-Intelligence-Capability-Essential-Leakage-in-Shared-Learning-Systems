from __future__ import annotations

import ast

from cel import CandidateReturnService, SERVICE_FAILURE, forced_event_cel
from demo import (
    code_model_demo,
    global_consistency_demo,
    navigation_demo,
    uncertainty_demo,
)


def test_navigation() -> None:
    result = navigation_demo()
    assert round(result["strong_reliability"], 7) == 0.8294369
    assert round(result["weak_reliability"], 7) == 0.2087505
    assert all(task["raw_leakage"] == 1.0 for task in result["tasks"].values())
    expected = {"q0": 0.08974, "q1": 0.08155, "q2": 0.07385, "q3": 0.08155, "q4": 0.08974}
    for task, target in expected.items():
        assert abs(result["tasks"][task]["alpha_NI"] - target) <= 5e-5


def test_global_consistency() -> None:
    result = global_consistency_demo()
    assert all(abs(value) <= 1e-10 for value in result["pairwise_CEL"].values())
    assert abs(result["global_CEL"] - 0.5) <= 1e-10


def test_uncertainty() -> None:
    result = uncertainty_demo()
    assert result["certified_lower"] <= result["nominal"] <= result["certified_upper"]
    assert round(result["certified_lower"], 3) == 0.220
    assert round(result["nominal"], 3) == 0.331
    assert round(result["certified_upper"], 3) == 0.443


def test_multiround() -> None:
    for horizon in range(1, 8):
        expected = 1.0 - 0.8**horizon
        assert abs(forced_event_cel(horizon, 0.2) - expected) <= 1e-12


def test_ast_boundary() -> None:
    service = CandidateReturnService()
    sentinel = "DO_NOT_RETURN_RAW"
    completion = f"```python\ndef solve(x):\n return x+1\n```\n{sentinel}"
    response = service.respond(completion, lambda _: True)
    assert sentinel not in response
    assert ast.dump(ast.parse(response)) == ast.dump(ast.parse("def solve(x):\n return x+1"))
    assert service.respond(completion, lambda _: False) == SERVICE_FAILURE
    assert service.respond("def broken(:", lambda _: True) == SERVICE_FAILURE


def test_code_model_demo() -> None:
    result = code_model_demo()
    assert result["failure_is_uniform"]
    assert 0.0 <= result["CEL_alpha_0.10"] <= 1.0
    assert 0.0 <= result["alpha_NI"] <= 1.0


if __name__ == "__main__":
    tests = [
        test_navigation,
        test_global_consistency,
        test_uncertainty,
        test_multiround,
        test_ast_boundary,
        test_code_model_demo,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")

