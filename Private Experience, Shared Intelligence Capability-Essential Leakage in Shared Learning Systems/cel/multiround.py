from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Distribution:
    name: str
    probabilities: tuple[float, ...]


class BellmanGameSolver:
    def __init__(self, instance: dict[str, Any]):
        self.instance = instance
        self.outputs = tuple(instance["outputs"])
        self.memo: dict[tuple[int, str, str], float] = {}
        self.policy: dict[tuple[int, str, str], dict[str, Any]] = {}

    @staticmethod
    def _belief_key(belief: float) -> str:
        return format(float(belief), ".17g")

    @staticmethod
    def _terminal(belief: float) -> float:
        return max(belief, 1.0 - belief)

    def _catalog(self, history: str, state: str, query: str) -> list[Distribution]:
        return [
            Distribution(row["id"], tuple(float(x) for x in row["probs"]))
            for row in self.instance["feasible"][history][state][query]
        ]

    def value(self, horizon: int, belief: float, state: str) -> float:
        key = (int(horizon), self._belief_key(belief), state)
        if key in self.memo:
            return self.memo[key]
        if horizon == 0:
            value = self._terminal(belief)
            self.memo[key] = value
            return value

        best_value = -1.0
        best_policy: dict[str, Any] | None = None
        for query in self.instance["queries_by_state"][state]:
            query_value = float("inf")
            query_policy = None
            for p0 in self._catalog("H0", state, query):
                for p1 in self._catalog("H1", state, query):
                    pair_value = 0.0
                    branches = []
                    for index, output in enumerate(self.outputs):
                        mixture = (1.0 - belief) * p0.probabilities[index] + belief * p1.probabilities[index]
                        if mixture == 0.0:
                            posterior = None
                            continuation = 0.0
                        else:
                            posterior = belief * p1.probabilities[index] / mixture
                            continuation = sum(
                                float(probability) * self.value(horizon - 1, posterior, next_state)
                                for next_state, probability in self.instance["transition"][state][query][output].items()
                            )
                        pair_value += mixture * continuation
                        branches.append(
                            {
                                "output": output,
                                "mixture_probability": mixture,
                                "posterior": posterior,
                                "continuation": continuation,
                            }
                        )
                    if pair_value < query_value - 1e-15:
                        query_value = pair_value
                        query_policy = {
                            "query": query,
                            "H0_distribution": p0.name,
                            "H1_distribution": p1.name,
                            "branches": branches,
                        }
            if query_value > best_value + 1e-15:
                best_value = query_value
                best_policy = query_policy

        self.memo[key] = best_value
        self.policy[key] = best_policy or {}
        return best_value


def forced_event_cel(horizon: int, event_probability: float) -> float:
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if not 0.0 <= event_probability <= 1.0:
        raise ValueError("event_probability must lie in [0, 1]")
    value = 0.0
    for _ in range(horizon):
        value = event_probability + (1.0 - event_probability) * value
    return value

