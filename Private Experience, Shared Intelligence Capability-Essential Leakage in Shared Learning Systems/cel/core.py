from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog


ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class CELResult:
    value: float
    distributions: np.ndarray
    success: bool
    solver_status: int
    solver_message: str
    max_residual: float


def total_variation(p: ArrayLike, q: ArrayLike) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("distributions must have the same shape")
    return float(0.5 * np.abs(p - q).sum())


def _problem_arrays(
    utilities: Sequence[ArrayLike],
    capabilities: ArrayLike,
    supports: Sequence[Sequence[bool]] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(utilities, dtype=float)
    c = np.asarray(capabilities, dtype=float)
    if u.ndim != 2 or u.shape[0] < 2:
        raise ValueError("utilities must be an H x O matrix with H >= 2")
    if c.shape != (u.shape[0],):
        raise ValueError("one capability value is required per history")
    if supports is None:
        s = np.ones_like(u, dtype=bool)
    else:
        s = np.asarray(supports, dtype=bool)
        if s.shape != u.shape:
            raise ValueError("supports must have the same shape as utilities")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(c)):
        raise ValueError("utilities and capabilities must be finite")
    return u, c, s


def pairwise_cel(
    utilities: Sequence[ArrayLike],
    capabilities: ArrayLike,
    alpha: float,
    supports: Sequence[Sequence[bool]] | None = None,
) -> CELResult:
    u, capability, support = _problem_arrays(utilities, capabilities, supports)
    if u.shape[0] != 2:
        raise ValueError("pairwise_cel requires exactly two histories")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    n = u.shape[1]
    variables = 3 * n
    objective = np.zeros(variables)
    objective[2 * n :] = 0.5

    a_eq = np.zeros((2, variables))
    a_eq[0, :n] = 1.0
    a_eq[1, n : 2 * n] = 1.0
    b_eq = np.ones(2)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for history in range(2):
        row = np.zeros(variables)
        row[history * n : (history + 1) * n] = -u[history]
        a_ub.append(row)
        b_ub.append(-(capability[history] - alpha))

    for output in range(n):
        positive = np.zeros(variables)
        positive[output] = 1.0
        positive[n + output] = -1.0
        positive[2 * n + output] = -1.0
        a_ub.append(positive)
        b_ub.append(0.0)

        negative = np.zeros(variables)
        negative[output] = -1.0
        negative[n + output] = 1.0
        negative[2 * n + output] = -1.0
        a_ub.append(negative)
        b_ub.append(0.0)

    bounds = [
        (0.0, None) if support[h, o] else (0.0, 0.0)
        for h in range(2)
        for o in range(n)
    ] + [(0.0, None)] * n

    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return CELResult(
            value=float("nan"),
            distributions=np.empty((0, n)),
            success=False,
            solver_status=int(result.status),
            solver_message=str(result.message),
            max_residual=float("inf"),
        )

    distributions = result.x[: 2 * n].reshape(2, n)
    residuals = [
        abs(distributions[0].sum() - 1.0),
        abs(distributions[1].sum() - 1.0),
        max(0.0, capability[0] - alpha - distributions[0] @ u[0]),
        max(0.0, capability[1] - alpha - distributions[1] @ u[1]),
        abs(float(result.fun) - total_variation(distributions[0], distributions[1])),
    ]
    return CELResult(
        value=float(result.fun),
        distributions=distributions,
        success=True,
        solver_status=int(result.status),
        solver_message=str(result.message),
        max_residual=float(max(residuals)),
    )


def strict_noninterference(
    utilities: Sequence[ArrayLike],
    capabilities: ArrayLike,
    alpha: float,
    supports: Sequence[Sequence[bool]] | None = None,
) -> tuple[bool, np.ndarray | None, float]:
    u, capability, support = _problem_arrays(utilities, capabilities, supports)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    common_support = np.all(support, axis=0)
    result = linprog(
        np.zeros(u.shape[1]),
        A_ub=-u,
        b_ub=-(capability - alpha),
        A_eq=np.ones((1, u.shape[1])),
        b_eq=np.ones(1),
        bounds=[(0.0, None) if allowed else (0.0, 0.0) for allowed in common_support],
        method="highs",
    )
    if not result.success:
        return False, None, float("inf")

    distribution = np.asarray(result.x, dtype=float)
    residual = max(
        abs(distribution.sum() - 1.0),
        float(np.max(np.maximum(0.0, capability - alpha - u @ distribution))),
    )
    return True, distribution, residual


def alpha_ni(
    utilities: Sequence[ArrayLike],
    capabilities: ArrayLike,
    supports: Sequence[Sequence[bool]] | None = None,
    *,
    upper: float = 1.0,
    tolerance: float = 1e-10,
) -> float:
    feasible, _, _ = strict_noninterference(utilities, capabilities, upper, supports)
    if not feasible:
        raise ValueError("strict non-interference is infeasible at the upper bound")

    lower = 0.0
    while upper - lower > tolerance:
        middle = 0.5 * (lower + upper)
        feasible, _, _ = strict_noninterference(
            utilities, capabilities, middle, supports
        )
        if feasible:
            upper = middle
        else:
            lower = middle
    return upper


def global_cel(
    utilities: Sequence[ArrayLike],
    capabilities: ArrayLike,
    alpha: float,
    supports: Sequence[Sequence[bool]] | None = None,
    edges: Iterable[tuple[int, int]] | None = None,
) -> CELResult:
    u, capability, support = _problem_arrays(utilities, capabilities, supports)
    histories, outputs = u.shape
    edge_list = list(combinations(range(histories), 2) if edges is None else edges)
    if not edge_list:
        raise ValueError("at least one history edge is required")

    n_p = histories * outputs
    n_z = len(edge_list) * outputs
    t_index = n_p + n_z
    variables = t_index + 1
    objective = np.zeros(variables)
    objective[t_index] = 1.0

    a_eq = np.zeros((histories, variables))
    for history in range(histories):
        a_eq[history, history * outputs : (history + 1) * outputs] = 1.0

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for history in range(histories):
        row = np.zeros(variables)
        row[history * outputs : (history + 1) * outputs] = -u[history]
        a_ub.append(row)
        b_ub.append(-(capability[history] - alpha))

    for edge_index, (left, right) in enumerate(edge_list):
        if left == right or not (0 <= left < histories and 0 <= right < histories):
            raise ValueError("invalid history edge")
        for output in range(outputs):
            z_index = n_p + edge_index * outputs + output
            positive = np.zeros(variables)
            positive[left * outputs + output] = 1.0
            positive[right * outputs + output] = -1.0
            positive[z_index] = -1.0
            a_ub.append(positive)
            b_ub.append(0.0)

            negative = np.zeros(variables)
            negative[left * outputs + output] = -1.0
            negative[right * outputs + output] = 1.0
            negative[z_index] = -1.0
            a_ub.append(negative)
            b_ub.append(0.0)

        tv_bound = np.zeros(variables)
        start = n_p + edge_index * outputs
        tv_bound[start : start + outputs] = 0.5
        tv_bound[t_index] = -1.0
        a_ub.append(tv_bound)
        b_ub.append(0.0)

    bounds = [
        (0.0, None) if support[h, o] else (0.0, 0.0)
        for h in range(histories)
        for o in range(outputs)
    ] + [(0.0, None)] * n_z + [(0.0, None)]

    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=np.ones(histories),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return CELResult(
            value=float("nan"),
            distributions=np.empty((0, outputs)),
            success=False,
            solver_status=int(result.status),
            solver_message=str(result.message),
            max_residual=float("inf"),
        )

    distributions = result.x[:n_p].reshape(histories, outputs)
    edge_tvs = [total_variation(distributions[i], distributions[j]) for i, j in edge_list]
    residuals = [abs(row.sum() - 1.0) for row in distributions]
    residuals.extend(
        max(0.0, capability[h] - alpha - distributions[h] @ u[h])
        for h in range(histories)
    )
    residuals.append(abs(float(result.fun) - max(edge_tvs)))
    return CELResult(
        value=float(result.fun),
        distributions=distributions,
        success=True,
        solver_status=int(result.status),
        solver_message=str(result.message),
        max_residual=float(max(residuals)),
    )


def uncertainty_interval(
    utilities: Sequence[ArrayLike],
    capabilities: ArrayLike,
    alpha: float,
    epsilon: float,
    supports: Sequence[Sequence[bool]] | None = None,
) -> tuple[CELResult, CELResult, CELResult]:
    u, capability, support = _problem_arrays(utilities, capabilities, supports)
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    nominal = pairwise_cel(u, capability, alpha, support)
    lower = pairwise_cel(u + epsilon, capability - epsilon, alpha, support)
    upper = pairwise_cel(u - epsilon, capability + epsilon, alpha, support)
    return lower, nominal, upper

