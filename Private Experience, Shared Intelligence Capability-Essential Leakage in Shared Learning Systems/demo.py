from __future__ import annotations

import json

import numpy as np
from scipy.stats import beta

from cel import (
    CandidateReturnService,
    SERVICE_FAILURE,
    alpha_ni,
    forced_event_cel,
    global_cel,
    pairwise_cel,
    total_variation,
    uncertainty_interval,
)


def navigation_demo() -> dict:
    strong = float(beta.ppf(0.05, 40, 4))
    weak = float(beta.ppf(0.05, 14, 30))
    tasks = {
        "q0": (12, 16, 20),
        "q1": (16, 14, 16),
        "q2": (16, 12, 16),
        "q3": (16, 14, 16),
        "q4": (20, 16, 12),
    }

    def fast(reliability: float, length: int) -> float:
        return 0.995**length * reliability

    def protected(reliability: float, length: int) -> float:
        return 0.995**length * (0.62 + 0.38 * reliability)

    def public(length: int) -> float:
        return 0.995**length * 0.82

    rows = {}
    for task, (upper_length, public_length, lower_length) in tasks.items():
        h0 = np.array(
            [
                fast(strong, upper_length),
                protected(strong, upper_length),
                public(public_length),
                protected(weak, lower_length),
                fast(weak, lower_length),
            ]
        )
        h1 = np.array(
            [
                fast(weak, upper_length),
                protected(weak, upper_length),
                public(public_length),
                protected(strong, lower_length),
                fast(strong, lower_length),
            ]
        )
        capability = np.array([h0.max(), h1.max()])
        frontier = {
            f"{alpha:.2f}": pairwise_cel([h0, h1], capability, alpha).value
            for alpha in np.linspace(0.0, 0.12, 7)
        }
        raw_h0 = np.eye(5)[int(h0.argmax())]
        raw_h1 = np.eye(5)[int(h1.argmax())]
        rows[task] = {
            "raw_leakage": total_variation(raw_h0, raw_h1),
            "alpha_NI": alpha_ni([h0, h1], capability),
            "frontier": frontier,
        }
    return {"strong_reliability": strong, "weak_reliability": weak, "tasks": rows}


def global_consistency_demo() -> dict:
    utilities = np.ones((3, 3))
    capabilities = np.ones(3)
    supports = np.array(
        [
            [True, True, False],
            [False, True, True],
            [True, False, True],
        ]
    )
    pair_values = {}
    for left, right in [(0, 1), (1, 2), (2, 0)]:
        pair_values[f"H{left}-H{right}"] = pairwise_cel(
            utilities[[left, right]],
            capabilities[[left, right]],
            0.0,
            supports[[left, right]],
        ).value
    global_result = global_cel(
        utilities,
        capabilities,
        0.0,
        supports,
        edges=[(0, 1), (1, 2), (2, 0)],
    )
    return {
        "pairwise_CEL": pair_values,
        "global_CEL": global_result.value,
        "distributions": global_result.distributions.tolist(),
    }


def uncertainty_demo() -> dict:
    navigation = navigation_demo()
    strong = navigation["strong_reliability"]
    weak = navigation["weak_reliability"]
    upper_length, public_length, lower_length = 12, 16, 20
    h0 = np.array(
        [
            0.995**upper_length * strong,
            0.995**upper_length * (0.62 + 0.38 * strong),
            0.995**public_length * 0.82,
            0.995**lower_length * (0.62 + 0.38 * weak),
            0.995**lower_length * weak,
        ]
    )
    h1 = np.array(
        [
            0.995**upper_length * weak,
            0.995**upper_length * (0.62 + 0.38 * weak),
            0.995**public_length * 0.82,
            0.995**lower_length * (0.62 + 0.38 * strong),
            0.995**lower_length * strong,
        ]
    )
    lower, nominal, upper = uncertainty_interval(
        [h0, h1], [h0.max(), h1.max()], alpha=0.06, epsilon=0.005
    )
    return {
        "certified_lower": lower.value,
        "nominal": nominal.value,
        "certified_upper": upper.value,
    }


def code_model_demo() -> dict:
    service = CandidateReturnService()
    completion = """```python
# formatting is intentionally non-canonical
def solve(payload):
    return payload['value']+1
```
RAW_COMPLETION_METADATA"""
    response = service.respond(completion, lambda source: "def solve" in source)
    failure = service.respond("```python\ndef broken(:\n```", lambda _: True)

    utilities = [[1.0, 0.6, 0.0], [0.0, 0.6, 1.0]]
    capability = [1.0, 1.0]
    cel = pairwise_cel(utilities, capability, alpha=0.1)
    return {
        "service_response": response,
        "failure_response": failure,
        "failure_is_uniform": failure == SERVICE_FAILURE,
        "CEL_alpha_0.10": cel.value,
        "alpha_NI": alpha_ni(utilities, capability),
    }


def multiround_demo() -> dict:
    return {
        str(horizon): forced_event_cel(horizon, event_probability=0.2)
        for horizon in range(1, 5)
    }


def run_all() -> dict:
    return {
        "navigation": navigation_demo(),
        "global_consistency": global_consistency_demo(),
        "uncertainty": uncertainty_demo(),
        "multiround": multiround_demo(),
        "code_model": code_model_demo(),
    }


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))

