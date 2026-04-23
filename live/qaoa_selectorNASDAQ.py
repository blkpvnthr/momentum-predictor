#!/usr/bin/env python3
"""
qaoa_selector_output_ready.py

Universe builder that is designed to ACTUALLY FINISH and produce output on a laptop.

Behavior:
1. Load ranked candidates and prices from the same folder.
2. Apply liquidity / price filters to reduce micro-cap concentration.
3. Build a small core-selection problem.
4. Try QAOA first if enabled.
5. If QAOA is unavailable, too slow, or fails, fall back to:
      exact -> classical greedy/local-search
6. Expand the selected core into a final 30-name universe.
7. Save output files every run.

Expected input files in the same directory:
    - tqqq_nasdaq_comovers.csv
    - tqqq_nasdaq_prices.csv

Outputs:
    - qaoa_core_symbols.txt
    - qaoa_core_symbols.json
    - qaoa_top30_symbols.txt
    - qaoa_top30_symbols.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# -------------------------
# OPTIONAL QISKIT IMPORTS
# -------------------------
QISKIT_IMPORT_ERROR = None
HAVE_QISKIT = False
QAOA = None
NumPyMinimumEigensolver = None
COBYLA = None
QuadraticProgram = None
MinimumEigenOptimizer = None
StatevectorSampler = None

try:
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit.primitives import StatevectorSampler

    HAVE_QISKIT = True
except Exception as e:
    QISKIT_IMPORT_ERROR = repr(e)
    HAVE_QISKIT = False


# -------------------------
# PATHS / DEFAULTS
# -------------------------
THIS_FILE = Path(__file__).resolve()
LIVE_DIR = THIS_FILE.parent

RANKED_CSV = LIVE_DIR / "tqqq_nasdaq_comovers.csv"
PRICES_CSV = LIVE_DIR / "tqqq_nasdaq_prices.csv"

CORE_TXT = LIVE_DIR / "qaoa_core_symbols.txt"
CORE_JSON = LIVE_DIR / "qaoa_core_symbols.json"
TOP30_TXT = LIVE_DIR / "qaoa_top30_symbols.txt"
TOP30_JSON = LIVE_DIR / "qaoa_top30_symbols.json"

# Solver policy
PREFERRED_SOLVER = "qaoa"   # "qaoa", "exact", or "classical"
ALLOW_FALLBACK = True

# Keep the quantum problem intentionally small
CANDIDATE_COUNT = 10
CORE_SIZE = 4

# Final expanded universe size
FINAL_UNIVERSE_SIZE = 30

# Optimization penalties
LAMBDA_DIVERSITY = 0.65
GAMMA_CARDINALITY = 10.0
QAOA_REPS = 1
QAOA_MAXITER = 20
QAOA_WARN_SECONDS = 20.0

# Liquidity / anti-microcap controls
MIN_PRICE = 5.00
MIN_AVG_VOLUME = 300_000
MIN_AVG_DOLLAR_VOLUME = 5_000_000

# Count lower-liquidity names within the filtered pool
MICROCAP_DOLLAR_VOLUME_PERCENTILE = 0.35
MAX_MICROCAP_COUNT_CORE = 1
MAX_MICROCAP_COUNT_FINAL = 6

# Candidate attractiveness blend
FACTOR_BLEND = {
    "score": 0.50,
    "corr": 0.10,
    "same_direction_pct": 0.08,
    "up_up_pct": 0.05,
    "liquidity_score": 0.12,
    "avg_dollar_volume": 0.15,
}

# Final expansion scoring
EXPANSION_ALPHA_WEIGHT = 0.45
EXPANSION_CORE_AFFINITY_WEIGHT = 0.30
EXPANSION_LIQUIDITY_WEIGHT = 0.20
EXPANSION_REDUNDANCY_PENALTY = 0.20


@dataclass
class CoreSelectionResult:
    core_symbols: list[str]
    objective_value: float
    solve_mode_used: str
    metadata: dict[str, object]


@dataclass
class FullSelectionResult:
    core_symbols: list[str]
    final_symbols: list[str]
    objective_value: float
    solve_mode_used: str
    metadata: dict[str, object]


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    finite = s.replace([np.inf, -np.inf], np.nan)
    s_min = finite.min()
    s_max = finite.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series(0.5, index=s.index, dtype=float)
    return (finite - s_min) / (s_max - s_min)


def require_inputs() -> None:
    if not RANKED_CSV.exists():
        raise RuntimeError(f"Missing ranked CSV: {RANKED_CSV}")
    if not PRICES_CSV.exists():
        raise RuntimeError(f"Missing prices CSV: {PRICES_CSV}")


def load_ranked_candidates() -> pd.DataFrame:
    ranked = pd.read_csv(RANKED_CSV)
    if ranked.empty:
        raise RuntimeError(f"Ranked CSV is empty: {RANKED_CSV}")

    ranked["symbol"] = ranked["symbol"].astype(str).str.upper()
    ranked = ranked.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    needed = [
        "score",
        "corr",
        "same_direction_pct",
        "up_up_pct",
        "liquidity_score",
        "avg_dollar_volume",
        "avg_volume",
        "price",
    ]
    for col in needed:
        if col not in ranked.columns:
            ranked[col] = np.nan

    ranked["price"] = pd.to_numeric(ranked["price"], errors="coerce")
    ranked["avg_volume"] = pd.to_numeric(ranked["avg_volume"], errors="coerce")
    ranked["avg_dollar_volume"] = pd.to_numeric(ranked["avg_dollar_volume"], errors="coerce")

    ranked = ranked[
        (ranked["price"].fillna(0.0) >= MIN_PRICE)
        & (ranked["avg_volume"].fillna(0.0) >= MIN_AVG_VOLUME)
        & (ranked["avg_dollar_volume"].fillna(0.0) >= MIN_AVG_DOLLAR_VOLUME)
    ].copy()

    if ranked.empty:
        raise RuntimeError(
            "No candidates remain after liquidity/price filters. "
            "Relax MIN_PRICE / MIN_AVG_VOLUME / MIN_AVG_DOLLAR_VOLUME."
        )

    ranked = ranked.sort_values(
        ["score", "avg_dollar_volume", "corr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return ranked


def load_price_matrix(symbols: Iterable[str]) -> pd.DataFrame:
    prices = pd.read_csv(PRICES_CSV, index_col=0)
    if prices.empty:
        raise RuntimeError(f"Prices CSV is empty: {PRICES_CSV}")

    prices.index = pd.to_datetime(prices.index, utc=True, errors="coerce")
    prices = prices[~prices.index.isna()].sort_index()

    desired = [str(s).upper() for s in symbols]
    available = [s for s in desired if s in prices.columns]
    if not available:
        raise RuntimeError("None of the candidate symbols were found in the price matrix.")

    prices = prices[available].apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        raise RuntimeError("Price matrix is empty after alignment.")
    return prices


def classify_microcaps(df: pd.DataFrame) -> set[str]:
    adv = pd.to_numeric(df["avg_dollar_volume"], errors="coerce")
    cutoff = adv.quantile(MICROCAP_DOLLAR_VOLUME_PERCENTILE)
    return set(df.loc[adv <= cutoff, "symbol"].astype(str).str.upper().tolist())


def build_alpha(df: pd.DataFrame) -> pd.Series:
    tmp = df.copy()
    for col in FACTOR_BLEND:
        if col not in tmp.columns:
            tmp[col] = np.nan

    tmp["log_avg_dollar_volume"] = np.log1p(pd.to_numeric(tmp["avg_dollar_volume"], errors="coerce"))
    tmp["price"] = pd.to_numeric(tmp["price"], errors="coerce")
    tmp["avg_volume"] = pd.to_numeric(tmp["avg_volume"], errors="coerce")

    alpha = pd.Series(0.0, index=tmp.index, dtype=float)
    for col, weight in FACTOR_BLEND.items():
        src = tmp[col]
        if col == "avg_dollar_volume":
            src = tmp["log_avg_dollar_volume"]
        alpha += weight * normalize_series(src).fillna(0.0)

    alpha += 0.08 * normalize_series(tmp["log_avg_dollar_volume"]).fillna(0.0)
    alpha += 0.04 * normalize_series(tmp["price"]).fillna(0.0)
    alpha += 0.03 * normalize_series(tmp["avg_volume"]).fillna(0.0)

    alpha.index = tmp["symbol"].astype(str).str.upper()
    return alpha


def build_similarity_penalty_matrix(prices: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    ret = prices[symbols].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if len(ret) < 20:
        return pd.DataFrame(np.zeros((len(symbols), len(symbols))), index=symbols, columns=symbols)

    corr = ret.corr().abs().fillna(0.0)
    arr = corr.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)

    # Amplify very high similarity to discourage redundancy
    arr = np.where(arr >= 0.80, arr * 1.50, arr)
    arr = np.where(arr >= 0.90, arr * 1.75, arr)

    return pd.DataFrame(arr, index=corr.index, columns=corr.columns)


def qubo_energy(
    x: np.ndarray,
    alpha: np.ndarray,
    similarity: np.ndarray,
    target_size: int,
    lambda_diversity: float,
    gamma_cardinality: float,
) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    reward_term = -float(alpha @ x)
    diversity_term = float(lambda_diversity * (x @ similarity @ x))
    cardinality_term = float(gamma_cardinality * (x.sum() - target_size) ** 2)
    return reward_term + diversity_term + cardinality_term


def shortlist_for_core(ranked: pd.DataFrame) -> pd.DataFrame:
    shortlist = ranked.head(CANDIDATE_COUNT).copy().reset_index(drop=True)
    if len(shortlist) < CORE_SIZE:
        raise RuntimeError(f"Only {len(shortlist)} candidates remain; need at least {CORE_SIZE}.")
    return shortlist


def greedy_core_selection(
    symbols: list[str],
    alpha_s: pd.Series,
    similarity_df: pd.DataFrame,
    core_size: int,
    lambda_diversity: float,
    microcap_symbols: set[str],
    max_microcap_count: int,
) -> CoreSelectionResult:
    selected: list[str] = []
    remaining = list(symbols)

    while len(selected) < core_size and remaining:
        best_sym = None
        best_gain = None
        current_microcaps = sum(1 for s in selected if s in microcap_symbols)

        for sym in remaining:
            if sym in microcap_symbols and current_microcaps >= max_microcap_count:
                continue

            reward = float(alpha_s[sym])
            redundancy = float(similarity_df.loc[sym, selected].sum()) if selected else 0.0
            gain = reward - lambda_diversity * redundancy

            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_sym = sym

        if best_sym is None:
            break

        selected.append(best_sym)
        remaining.remove(best_sym)

    if len(selected) != core_size:
        raise RuntimeError(f"Could not construct a {core_size}-name core under current constraints.")

    x = np.array([1.0 if s in selected else 0.0 for s in symbols], dtype=float)
    obj = qubo_energy(
        x=x,
        alpha=alpha_s[symbols].to_numpy(),
        similarity=similarity_df.loc[symbols, symbols].to_numpy(),
        target_size=core_size,
        lambda_diversity=lambda_diversity,
        gamma_cardinality=GAMMA_CARDINALITY,
    )

    return CoreSelectionResult(
        core_symbols=sorted(selected),
        objective_value=float(obj),
        solve_mode_used="classical_greedy_core",
        metadata={"microcap_count": sum(1 for s in selected if s in microcap_symbols)},
    )


def local_improvement_core(
    base_symbols: list[str],
    alpha_s: pd.Series,
    similarity_df: pd.DataFrame,
    core_size: int,
    lambda_diversity: float,
    microcap_symbols: set[str],
    max_microcap_count: int,
    max_passes: int = 2,
) -> list[str]:
    selected = sorted(base_symbols)
    all_syms = list(alpha_s.index)

    def feasible(current: list[str]) -> bool:
        return sum(1 for s in current if s in microcap_symbols) <= max_microcap_count

    def score(current: list[str]) -> float:
        x = np.array([1.0 if s in current else 0.0 for s in all_syms], dtype=float)
        return -qubo_energy(
            x=x,
            alpha=alpha_s[all_syms].to_numpy(),
            similarity=similarity_df.loc[all_syms, all_syms].to_numpy(),
            target_size=core_size,
            lambda_diversity=lambda_diversity,
            gamma_cardinality=GAMMA_CARDINALITY,
        )

    best_score = score(selected)

    for _ in range(max_passes):
        improved = False
        not_selected = [s for s in all_syms if s not in selected]

        for out_sym in list(selected):
            for in_sym in list(not_selected):
                trial = sorted([s for s in selected if s != out_sym] + [in_sym])
                if not feasible(trial):
                    continue
                trial_score = score(trial)
                if trial_score > best_score:
                    selected = trial
                    best_score = trial_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return selected


def build_quadratic_program(
    symbols: list[str],
    alpha_s: pd.Series,
    similarity_df: pd.DataFrame,
    target_size: int,
    lambda_diversity: float,
    gamma_cardinality: float,
    microcap_symbols: set[str],
    max_microcap_count: int,
) -> "QuadraticProgram":
    qp = QuadraticProgram("qaoa_core_selection")
    for sym in symbols:
        qp.binary_var(name=sym)

    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}

    for sym in symbols:
        linear[sym] = float(-alpha_s[sym] + gamma_cardinality * (1.0 - 2.0 * target_size))

    for i, si in enumerate(symbols):
        for j in range(i + 1, len(symbols)):
            sj = symbols[j]
            sij = float(similarity_df.loc[si, sj])
            quadratic[(si, sj)] = float(2.0 * gamma_cardinality + 2.0 * lambda_diversity * sij)

    qp.minimize(
        linear=linear,
        quadratic=quadratic,
        constant=float(gamma_cardinality * target_size**2),
    )

    if microcap_symbols:
        coeffs = {s: 1 for s in symbols if s in microcap_symbols}
        if coeffs:
            qp.linear_constraint(
                linear=coeffs,
                sense="<=",
                rhs=max_microcap_count,
                name="max_microcap_count",
            )
    return qp


def solve_with_qiskit(
    qp: "QuadraticProgram",
    solver: str,
    reps: int,
) -> tuple[list[str], float, str]:
    if not HAVE_QISKIT:
        raise RuntimeError(f"Qiskit solver requested but imports failed. Import error: {QISKIT_IMPORT_ERROR}")

    start = time.time()

    if solver == "exact":
        eigen_solver = NumPyMinimumEigensolver()
        optimizer = MinimumEigenOptimizer(eigen_solver)
        result = optimizer.solve(qp)
        used = "exact"
    elif solver == "qaoa":
        sampler = StatevectorSampler()
        qaoa = QAOA(
            sampler=sampler,
            optimizer=COBYLA(maxiter=QAOA_MAXITER),
            reps=reps,
        )
        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(qp)
        used = "qaoa"
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    elapsed = time.time() - start
    print(f"[solver] mode={used} elapsed={elapsed:.2f}s")
    if used == "qaoa" and elapsed > QAOA_WARN_SECONDS:
        print(f"[warning] QAOA solve took longer than {QAOA_WARN_SECONDS:.0f}s")

    vars_sorted = [v.name for v in qp.variables]
    selected = [name for name, val in zip(vars_sorted, result.x) if int(round(val)) == 1]
    return sorted(selected), float(result.fval), used


def select_core(shortlist: pd.DataFrame, prices: pd.DataFrame) -> CoreSelectionResult:
    symbols = shortlist["symbol"].astype(str).str.upper().tolist()
    alpha_s = build_alpha(shortlist)
    similarity_df = build_similarity_penalty_matrix(prices, symbols)
    microcap_symbols = classify_microcaps(shortlist)

    classical_core = greedy_core_selection(
        symbols=symbols,
        alpha_s=alpha_s,
        similarity_df=similarity_df,
        core_size=CORE_SIZE,
        lambda_diversity=LAMBDA_DIVERSITY,
        microcap_symbols=microcap_symbols,
        max_microcap_count=MAX_MICROCAP_COUNT_CORE,
    )
    improved_core = local_improvement_core(
        base_symbols=classical_core.core_symbols,
        alpha_s=alpha_s,
        similarity_df=similarity_df,
        core_size=CORE_SIZE,
        lambda_diversity=LAMBDA_DIVERSITY,
        microcap_symbols=microcap_symbols,
        max_microcap_count=MAX_MICROCAP_COUNT_CORE,
    )
    x_class = np.array([1.0 if s in improved_core else 0.0 for s in symbols], dtype=float)
    classical_obj = qubo_energy(
        x=x_class,
        alpha=alpha_s[symbols].to_numpy(),
        similarity=similarity_df.loc[symbols, symbols].to_numpy(),
        target_size=CORE_SIZE,
        lambda_diversity=LAMBDA_DIVERSITY,
        gamma_cardinality=GAMMA_CARDINALITY,
    )
    classical_result = CoreSelectionResult(
        core_symbols=sorted(improved_core),
        objective_value=float(classical_obj),
        solve_mode_used="classical_greedy_local_search",
        metadata={"microcap_count": sum(1 for s in improved_core if s in microcap_symbols)},
    )

    if PREFERRED_SOLVER == "classical":
        return classical_result

    if PREFERRED_SOLVER in {"qaoa", "exact"}:
        try:
            qp = build_quadratic_program(
                symbols=symbols,
                alpha_s=alpha_s,
                similarity_df=similarity_df,
                target_size=CORE_SIZE,
                lambda_diversity=LAMBDA_DIVERSITY,
                gamma_cardinality=GAMMA_CARDINALITY,
                microcap_symbols=microcap_symbols,
                max_microcap_count=MAX_MICROCAP_COUNT_CORE,
            )
            selected, obj, used = solve_with_qiskit(qp=qp, solver=PREFERRED_SOLVER, reps=QAOA_REPS)

            if len(selected) == CORE_SIZE:
                return CoreSelectionResult(
                    core_symbols=selected,
                    objective_value=float(obj),
                    solve_mode_used=used,
                    metadata={
                        "microcap_count": sum(1 for s in selected if s in microcap_symbols),
                        "qiskit_import_error": QISKIT_IMPORT_ERROR,
                    },
                )

            message = f"{used} returned {len(selected)} symbols instead of {CORE_SIZE}"
            if not ALLOW_FALLBACK:
                raise RuntimeError(message)
            print(f"[warning] {message}; falling back to classical result")

        except Exception as e:
            if not ALLOW_FALLBACK:
                raise
            print(f"[warning] {PREFERRED_SOLVER} failed: {e}")
            print("[fallback] using classical_greedy_local_search core")

    return classical_result


def expand_core_to_final(
    ranked: pd.DataFrame,
    prices: pd.DataFrame,
    core_symbols: list[str],
    final_size: int,
) -> list[str]:
    ranked = ranked.copy()
    ranked["symbol"] = ranked["symbol"].astype(str).str.upper()
    ranked = ranked[ranked["symbol"].isin(prices.columns)].copy().reset_index(drop=True)

    alpha_s = build_alpha(ranked)
    symbols = ranked["symbol"].tolist()
    similarity_df = build_similarity_penalty_matrix(prices, symbols)
    microcap_symbols = classify_microcaps(ranked)

    adv_norm = normalize_series(np.log1p(pd.to_numeric(ranked["avg_dollar_volume"], errors="coerce")))
    price_norm = normalize_series(pd.to_numeric(ranked["price"], errors="coerce"))

    selected = list(core_symbols)
    selected_set = set(selected)
    current_microcaps = sum(1 for s in selected if s in microcap_symbols)

    while len(selected) < final_size:
        candidates = [s for s in symbols if s not in selected_set]
        if not candidates:
            break

        best_sym = None
        best_score = None

        for sym in candidates:
            if sym in microcap_symbols and current_microcaps >= MAX_MICROCAP_COUNT_FINAL:
                continue

            row_idx = ranked.index[ranked["symbol"] == sym][0]

            alpha_part = float(alpha_s[sym])
            core_affinity = float(similarity_df.loc[sym, core_symbols].mean()) if core_symbols else 0.0
            liquidity_part = 0.70 * float(adv_norm.loc[row_idx]) + 0.30 * float(price_norm.loc[row_idx])
            redundancy = float(similarity_df.loc[sym, selected].mean()) if selected else 0.0

            score = (
                EXPANSION_ALPHA_WEIGHT * alpha_part
                + EXPANSION_CORE_AFFINITY_WEIGHT * core_affinity
                + EXPANSION_LIQUIDITY_WEIGHT * liquidity_part
                - EXPANSION_REDUNDANCY_PENALTY * redundancy
            )

            if best_score is None or score > best_score:
                best_score = score
                best_sym = sym

        if best_sym is None:
            break

        selected.append(best_sym)
        selected_set.add(best_sym)
        if best_sym in microcap_symbols:
            current_microcaps += 1

    if len(selected) != final_size:
        # deterministic backfill by rank, still respecting microcap cap when possible
        for sym in ranked["symbol"].tolist():
            if len(selected) >= final_size:
                break
            if sym in selected_set:
                continue
            if sym in microcap_symbols and current_microcaps >= MAX_MICROCAP_COUNT_FINAL:
                continue
            selected.append(sym)
            selected_set.add(sym)
            if sym in microcap_symbols:
                current_microcaps += 1

    if len(selected) != final_size:
        raise RuntimeError(f"Could not expand core to {final_size} names. Reached {len(selected)}.")

    core_part = [s for s in core_symbols if s in selected_set]
    rest = [s for s in ranked["symbol"].tolist() if s in selected_set and s not in set(core_part)]
    final = core_part + rest
    return final[:final_size]


def select_universe() -> FullSelectionResult:
    require_inputs()

    ranked = load_ranked_candidates()
    prices = load_price_matrix(ranked["symbol"].tolist())
    ranked = ranked[ranked["symbol"].isin(prices.columns)].copy().reset_index(drop=True)

    shortlist = shortlist_for_core(ranked)
    shortlist_prices = prices[shortlist["symbol"].tolist()].copy()

    core_result = select_core(shortlist=shortlist, prices=shortlist_prices)
    final_symbols = expand_core_to_final(
        ranked=ranked,
        prices=prices,
        core_symbols=core_result.core_symbols,
        final_size=FINAL_UNIVERSE_SIZE,
    )

    return FullSelectionResult(
        core_symbols=core_result.core_symbols,
        final_symbols=final_symbols,
        objective_value=core_result.objective_value,
        solve_mode_used=core_result.solve_mode_used,
        metadata={
            "preferred_solver": PREFERRED_SOLVER,
            "allow_fallback": ALLOW_FALLBACK,
            "candidate_count": CANDIDATE_COUNT,
            "core_size": CORE_SIZE,
            "final_universe_size": FINAL_UNIVERSE_SIZE,
            "lambda_diversity": LAMBDA_DIVERSITY,
            "gamma_cardinality": GAMMA_CARDINALITY,
            "qaoa_reps": QAOA_REPS,
            "qaoa_maxiter": QAOA_MAXITER,
            "qiskit_import_error": QISKIT_IMPORT_ERROR,
        },
    )


def save_outputs(result: FullSelectionResult) -> None:
    CORE_TXT.write_text(repr(result.core_symbols), encoding="utf-8")
    TOP30_TXT.write_text(repr(result.final_symbols), encoding="utf-8")

    CORE_JSON.write_text(
        json.dumps(
            {
                "core_symbols": result.core_symbols,
                "objective_value": result.objective_value,
                "solve_mode_used": result.solve_mode_used,
                "metadata": result.metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    TOP30_JSON.write_text(
        json.dumps(
            {
                "core_symbols": result.core_symbols,
                "final_symbols": result.final_symbols,
                "objective_value": result.objective_value,
                "solve_mode_used": result.solve_mode_used,
                "metadata": result.metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    print("===== QAOA CORE + EXPANSION UNIVERSE SELECTION =====")
    print(f"HAVE_QISKIT: {HAVE_QISKIT}")
    if QISKIT_IMPORT_ERROR:
        print(f"Qiskit import error: {QISKIT_IMPORT_ERROR}")
    print(f"preferred_solver: {PREFERRED_SOLVER}")
    print(f"allow_fallback: {ALLOW_FALLBACK}")
    print(f"candidate_count: {CANDIDATE_COUNT}")
    print(f"core_size: {CORE_SIZE}")
    print(f"final_universe_size: {FINAL_UNIVERSE_SIZE}")

    result = select_universe()
    save_outputs(result)

    print(f"\nsolve_mode_used: {result.solve_mode_used}")
    print(f"core_objective_value: {result.objective_value:.6f}")

    print("\n[CORE SYMBOLS]")
    print(result.core_symbols)

    print("\n[FINAL 30 SYMBOLS]")
    print(result.final_symbols)

    print("\n[CORE PYTHON LIST]")
    print(repr(result.core_symbols))

    print("\n[FINAL PYTHON LIST]")
    print(repr(result.final_symbols))

    print(f"\nSaved core txt: {CORE_TXT}")
    print(f"Saved core json: {CORE_JSON}")
    print(f"Saved final txt: {TOP30_TXT}")
    print(f"Saved final json: {TOP30_JSON}")
    print("\n===== DONE =====")


if __name__ == "__main__":
    main()
