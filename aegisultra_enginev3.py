# aegisultra_enginev3.py

from __future__ import annotations

import copy
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln
from scipy.stats import poisson

import aegisultra_enginev2 as v2


# ============================================================
# AEGIS ULTRA V3
#
# Extends the tested V2 reconstruction engine with:
#
# 1. Shin de-vigging.
# 2. Pre-projection prior comparison:
#       - Dixon-Coles
#       - Independent Poisson
#       - Independent COM-Poisson
# 3. Dixon-Coles remains the sole projection baseline.
# 4. Structured odds-movement analysis:
#       - External bookmaker consensus
#       - Current Pinnacle confirmation
#       - HKJC lag / follow / conflict
#       - Supported / Neutral / Conflicted verdict
#
# This file intentionally imports and extends V2 so that all
# existing settlement, target-line-out, family-out, stress,
# HT/FT coherence and recommendation logic remains unchanged.
# ============================================================


ENGINE_NAME = "Aegis Ultra Hit-First Reconstruction Engine"
ENGINE_VERSION = "3.0.0"

TOL = getattr(v2, "TOL", 1e-10)

PRIOR_NAMES = (
    "DIXON_COLES",
    "INDEPENDENT_POISSON",
    "COM_POISSON",
)

SUPPORTED_DEVIG_METHODS = {
    "MULTIPLICATIVE",
    "POWER",
    "SHIN",
}

DEFAULT_MOVEMENT_SETTINGS = {
    # A movement smaller than this is treated as neutral.
    "neutral_threshold_pp": 0.50,

    # Used when classifying a strong probability movement.
    "strong_threshold_pp": 2.00,

    # Prefer at least this many independent external books.
    "minimum_external_books": 2,

    # Agreement required for a normal usable consensus.
    "minimum_agreement": 0.60,

    # Agreement required for a strong consensus.
    "strong_agreement": 0.75,

    # Allows a nearby main line to provide directional evidence.
    "adjacent_line_tolerance": 0.50,
}

PRIOR_FIT_MAX_EVALUATIONS = 350


# ============================================================
# 1. Preserve original V2 functions across Streamlit reloads
# ============================================================

if not hasattr(v2, "__aegis_v3_originals__"):
    v2.__aegis_v3_originals__ = {
        "validate_input_data": v2.validate_input_data,
        "run_engine": v2.run_engine,
        "devig_probabilities": v2.devig_probabilities,
        "build_market_model": v2.build_market_model,
        "evaluate_model_scenarios": (
            v2.evaluate_model_scenarios
        ),
    }


_ORIGINALS = v2.__aegis_v3_originals__

_original_validate_input_data = _ORIGINALS[
    "validate_input_data"
]

_original_run_engine = _ORIGINALS[
    "run_engine"
]


# Make V2-generated metadata identify the upgraded engine.
v2.ENGINE_NAME = ENGINE_NAME
v2.ENGINE_VERSION = ENGINE_VERSION


# ============================================================
# 2. Generic helpers
# ============================================================

def _safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(value):
        return default

    return value


def _clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def _summary(values) -> Dict[str, Any]:
    return v2.summary_statistics(
        value
        for value in values
        if value is not None
    )


def _sign(
    value: float,
    threshold: float = 0.0,
) -> int:
    if value > threshold:
        return 1

    if value < -threshold:
        return -1

    return 0


def _median(values) -> Optional[float]:
    clean = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(float(value))
    ]

    if not clean:
        return None

    return float(np.median(clean))


def _normalize_book_key(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "_")
    )


def _parse_timestamp(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()

    if not text:
        return None

    normalized = text.replace(
        "Z",
        "+00:00",
    )

    try:
        return datetime.fromisoformat(
            normalized
        )
    except ValueError:
        return None


# ============================================================
# 3. Shin de-vigging
# ============================================================

def shin_devig(
    odds,
) -> np.ndarray:
    """
    Shin insider-trading de-vigging.

    For normal overround markets, solve the Shin insider
    proportion z so that the adjusted probabilities sum to one.

    If the supplied market has no positive overround or a stable
    Shin root cannot be found, multiplicative de-vigging is used
    as a safe fallback.
    """

    odds = np.asarray(
        [
            v2.validate_odds(
                value,
                f"Odds #{index + 1}",
            )
            for index, value
            in enumerate(odds)
        ],
        dtype=float,
    )

    inverse = 1.0 / odds
    inverse_sum = float(
        inverse.sum()
    )

    if inverse_sum <= 1.0 + 1e-12:
        return v2.multiplicative_devig(
            odds
        )

    def probabilities_for_z(
        z: float,
    ) -> np.ndarray:
        # Rationalized form of:
        #
        # (sqrt(z² + 4(1-z)q²/S) - z) /
        # (2(1-z))
        #
        # This form is more stable when z approaches one.
        term = (
            z * z
            + (
                4.0
                * (1.0 - z)
                * inverse ** 2
                / inverse_sum
            )
        )

        square_root = np.sqrt(
            np.maximum(term, 0.0)
        )

        denominator = (
            square_root + z
        )

        probabilities = (
            2.0
            * inverse ** 2
            / inverse_sum
            / np.maximum(
                denominator,
                1e-300,
            )
        )

        return probabilities

    def equation(
        z: float,
    ) -> float:
        return float(
            probabilities_for_z(z).sum()
            - 1.0
        )

    lower = 0.0
    upper = 1.0 - 1e-12

    lower_value = equation(lower)
    upper_value = equation(upper)

    if (
        not math.isfinite(lower_value)
        or not math.isfinite(upper_value)
        or lower_value < 0.0
        or upper_value > 0.0
    ):
        return v2.multiplicative_devig(
            odds
        )

    for _ in range(240):
        middle = (
            lower + upper
        ) / 2.0

        middle_value = equation(
            middle
        )

        if middle_value > 0.0:
            lower = middle
        else:
            upper = middle

    z = (
        lower + upper
    ) / 2.0

    probabilities = probabilities_for_z(
        z
    )

    probabilities = np.maximum(
        probabilities,
        0.0,
    )

    total = float(
        probabilities.sum()
    )

    if (
        not math.isfinite(total)
        or total <= TOL
    ):
        return v2.multiplicative_devig(
            odds
        )

    return probabilities / total


def devig_probabilities(
    odds,
    method,
):
    method = str(
        method
    ).strip().upper()

    if method == "MULTIPLICATIVE":
        return v2.multiplicative_devig(
            odds
        )

    if method == "POWER":
        return v2.power_devig(
            odds
        )

    if method == "SHIN":
        return shin_devig(
            odds
        )

    raise ValueError(
        f"Unsupported de-vig method: {method}"
    )


# ============================================================
# 4. Additional pre-projection priors
# ============================================================

def independent_poisson_distribution(
    lambda_home: float,
    lambda_away: float,
    max_goals: int,
) -> np.ndarray:
    values = np.arange(
        max_goals + 1
    )

    home_probability = poisson.pmf(
        values,
        lambda_home,
    )

    away_probability = poisson.pmf(
        values,
        lambda_away,
    )

    matrix = np.outer(
        home_probability,
        away_probability,
    )

    total = float(
        matrix.sum()
    )

    if (
        not math.isfinite(total)
        or total <= TOL
    ):
        raise ValueError(
            "Independent Poisson prior has "
            "invalid probability mass."
        )

    matrix /= total

    return matrix.ravel()


def com_poisson_marginal(
    rate: float,
    dispersion: float,
    max_goals: int,
) -> np.ndarray:
    """
    Truncated COM-Poisson marginal:

        P(X=k) proportional to rate^k / (k!)^dispersion

    dispersion = 1 gives a normal Poisson distribution.
    dispersion > 1 gives under-dispersion.
    dispersion < 1 gives over-dispersion.
    """

    if rate <= 0.0:
        raise ValueError(
            "COM-Poisson rate must be positive."
        )

    if dispersion <= 0.0:
        raise ValueError(
            "COM-Poisson dispersion must be positive."
        )

    values = np.arange(
        max_goals + 1,
        dtype=float,
    )

    log_weights = (
        values * math.log(rate)
        - dispersion * gammaln(
            values + 1.0
        )
    )

    log_weights -= float(
        np.max(log_weights)
    )

    weights = np.exp(
        log_weights
    )

    total = float(
        weights.sum()
    )

    if (
        not math.isfinite(total)
        or total <= TOL
    ):
        raise ValueError(
            "COM-Poisson marginal normalization failed."
        )

    return weights / total


def com_poisson_distribution(
    rate_home: float,
    rate_away: float,
    dispersion_home: float,
    dispersion_away: float,
    max_goals: int,
) -> np.ndarray:
    home_probability = com_poisson_marginal(
        rate_home,
        dispersion_home,
        max_goals,
    )

    away_probability = com_poisson_marginal(
        rate_away,
        dispersion_away,
        max_goals,
    )

    matrix = np.outer(
        home_probability,
        away_probability,
    )

    matrix /= matrix.sum()

    return matrix.ravel()


def _prior_residuals(
    probabilities: np.ndarray,
    enriched_constraints,
) -> np.ndarray:
    return np.asarray(
        [
            (
                v2.effective_fair_probability(
                    probabilities,
                    item["a_coeff"],
                    item["b_coeff"],
                )
                - float(item["target"])
            )
            for item in enriched_constraints
        ],
        dtype=float,
    )


def _fit_result_summary(
    probabilities: np.ndarray,
    residuals: np.ndarray,
    parameters: Dict[str, float],
    optimizer_success: bool,
    optimizer_message: str,
    parameter_count: int,
    constraint_count: int,
) -> Dict[str, Any]:
    return {
        "probabilities": probabilities,
        "parameters": parameters,
        "maximum_prior_residual": float(
            np.max(
                np.abs(residuals)
            )
        ),
        "prior_rmse": float(
            math.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        ),
        "optimizer_success": bool(
            optimizer_success
        ),
        "optimizer_message": str(
            optimizer_message
        ),
        "parameter_count": int(
            parameter_count
        ),
        "constraint_count": int(
            constraint_count
        ),
        "formally_identified": bool(
            constraint_count
            >= parameter_count
        ),
    }


def fit_independent_poisson_prior(
    constraints,
    max_goals,
) -> Dict[str, Any]:
    enriched, _, _ = v2.enrich_constraints(
        constraints,
        max_goals,
    )

    def residual_function(
        parameters,
    ):
        try:
            lambda_home = math.exp(
                float(parameters[0])
            )

            lambda_away = math.exp(
                float(parameters[1])
            )

            probabilities = (
                independent_poisson_distribution(
                    lambda_home,
                    lambda_away,
                    max_goals,
                )
            )

            return _prior_residuals(
                probabilities,
                enriched,
            )

        except Exception:
            return np.full(
                len(enriched),
                100.0,
                dtype=float,
            )

    starts = [
        (0.60, 0.50),
        (1.00, 1.00),
        (1.50, 0.80),
        (0.80, 1.50),
        (1.80, 1.30),
    ]

    results = []

    bounds = (
        np.asarray([
            math.log(0.01),
            math.log(0.01),
        ]),
        np.asarray([
            math.log(10.0),
            math.log(10.0),
        ]),
    )

    for home_start, away_start in starts:
        result = least_squares(
            residual_function,
            x0=np.asarray([
                math.log(home_start),
                math.log(away_start),
            ]),
            bounds=bounds,
            method="trf",
            max_nfev=(
                PRIOR_FIT_MAX_EVALUATIONS
            ),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )

        if np.all(
            np.isfinite(result.fun)
        ):
            results.append(result)

    if not results:
        raise RuntimeError(
            "Independent Poisson fitting failed."
        )

    best = min(
        results,
        key=lambda item: (
            float(
                np.max(
                    np.abs(item.fun)
                )
            ),
            float(
                np.mean(
                    item.fun ** 2
                )
            ),
        ),
    )

    lambda_home = math.exp(
        float(best.x[0])
    )

    lambda_away = math.exp(
        float(best.x[1])
    )

    probabilities = (
        independent_poisson_distribution(
            lambda_home,
            lambda_away,
            max_goals,
        )
    )

    residuals = _prior_residuals(
        probabilities,
        enriched,
    )

    return _fit_result_summary(
        probabilities=probabilities,
        residuals=residuals,
        parameters={
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
        },
        optimizer_success=best.success,
        optimizer_message=best.message,
        parameter_count=2,
        constraint_count=len(enriched),
    )


def fit_com_poisson_prior(
    constraints,
    max_goals,
) -> Dict[str, Any]:
    enriched, _, _ = v2.enrich_constraints(
        constraints,
        max_goals,
    )

    def residual_function(
        parameters,
    ):
        try:
            rate_home = math.exp(
                float(parameters[0])
            )

            rate_away = math.exp(
                float(parameters[1])
            )

            dispersion_home = math.exp(
                float(parameters[2])
            )

            dispersion_away = math.exp(
                float(parameters[3])
            )

            probabilities = (
                com_poisson_distribution(
                    rate_home,
                    rate_away,
                    dispersion_home,
                    dispersion_away,
                    max_goals,
                )
            )

            return _prior_residuals(
                probabilities,
                enriched,
            )

        except Exception:
            return np.full(
                len(enriched),
                100.0,
                dtype=float,
            )

    starts = [
        (0.80, 0.70, 1.00, 1.00),
        (1.20, 1.00, 1.00, 1.00),
        (1.60, 0.90, 0.80, 1.20),
        (0.90, 1.60, 1.20, 0.80),
    ]

    lower_bounds = np.asarray([
        math.log(0.01),
        math.log(0.01),
        math.log(0.25),
        math.log(0.25),
    ])

    upper_bounds = np.asarray([
        math.log(12.0),
        math.log(12.0),
        math.log(4.0),
        math.log(4.0),
    ])

    results = []

    for (
        home_start,
        away_start,
        home_dispersion_start,
        away_dispersion_start,
    ) in starts:
        result = least_squares(
            residual_function,
            x0=np.asarray([
                math.log(home_start),
                math.log(away_start),
                math.log(
                    home_dispersion_start
                ),
                math.log(
                    away_dispersion_start
                ),
            ]),
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
            method="trf",
            max_nfev=(
                PRIOR_FIT_MAX_EVALUATIONS
            ),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )

        if np.all(
            np.isfinite(result.fun)
        ):
            results.append(result)

    if not results:
        raise RuntimeError(
            "COM-Poisson fitting failed."
        )

    best = min(
        results,
        key=lambda item: (
            float(
                np.max(
                    np.abs(item.fun)
                )
            ),
            float(
                np.mean(
                    item.fun ** 2
                )
            ),
        ),
    )

    rate_home = math.exp(
        float(best.x[0])
    )

    rate_away = math.exp(
        float(best.x[1])
    )

    dispersion_home = math.exp(
        float(best.x[2])
    )

    dispersion_away = math.exp(
        float(best.x[3])
    )

    probabilities = com_poisson_distribution(
        rate_home,
        rate_away,
        dispersion_home,
        dispersion_away,
        max_goals,
    )

    residuals = _prior_residuals(
        probabilities,
        enriched,
    )

    return _fit_result_summary(
        probabilities=probabilities,
        residuals=residuals,
        parameters={
            "rate_home": rate_home,
            "rate_away": rate_away,
            "dispersion_home": (
                dispersion_home
            ),
            "dispersion_away": (
                dispersion_away
            ),
        },
        optimizer_success=best.success,
        optimizer_message=best.message,
        parameter_count=4,
        constraint_count=len(enriched),
    )


def _public_prior_diagnostics(
    prior_fit: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        key: value
        for key, value in prior_fit.items()
        if key != "probabilities"
    }


# ============================================================
# 5. V3 market model
# ============================================================

def build_market_model(
    constraints,
    max_goals,
):
    if not v2.validate_constraint_subset(
        constraints
    ):
        raise ValueError(
            "Insufficient independent market structure."
        )

    # --------------------------------------------------------
    # Baseline prior: Dixon-Coles
    # --------------------------------------------------------

    dixon_coles_fit = (
        v2.fit_dixon_coles_prior(
            constraints,
            max_goals,
        )
    )

    prior_fits: Dict[
        str,
        Dict[str, Any],
    ] = {
        "DIXON_COLES": {
            "probabilities": (
                dixon_coles_fit[
                    "probabilities"
                ]
            ),
            "parameters": {
                "lambda_home": (
                    dixon_coles_fit[
                        "lambda_home"
                    ]
                ),
                "lambda_away": (
                    dixon_coles_fit[
                        "lambda_away"
                    ]
                ),
                "rho": dixon_coles_fit[
                    "rho"
                ],
            },
            "maximum_prior_residual": (
                dixon_coles_fit[
                    "maximum_prior_residual"
                ]
            ),
            "prior_rmse": (
                dixon_coles_fit[
                    "prior_rmse"
                ]
            ),
            "optimizer_success": (
                dixon_coles_fit[
                    "optimizer_success"
                ]
            ),
            "optimizer_message": (
                dixon_coles_fit[
                    "optimizer_message"
                ]
            ),
            "parameter_count": 3,
            "constraint_count": len(
                constraints
            ),
            "formally_identified": bool(
                len(constraints) >= 3
            ),
        }
    }

    prior_errors = {}

    # --------------------------------------------------------
    # Diagnostic prior: Independent Poisson
    # --------------------------------------------------------

    try:
        prior_fits[
            "INDEPENDENT_POISSON"
        ] = fit_independent_poisson_prior(
            constraints,
            max_goals,
        )

    except Exception as error:
        prior_errors[
            "INDEPENDENT_POISSON"
        ] = str(error)

    # --------------------------------------------------------
    # Diagnostic prior: COM-Poisson
    # --------------------------------------------------------

    try:
        prior_fits[
            "COM_POISSON"
        ] = fit_com_poisson_prior(
            constraints,
            max_goals,
        )

    except Exception as error:
        prior_errors[
            "COM_POISSON"
        ] = str(error)

    # --------------------------------------------------------
    # Projection always starts from Dixon-Coles.
    # --------------------------------------------------------

    enriched, matrix, targets = (
        v2.enrich_constraints(
            constraints,
            max_goals,
        )
    )

    projection = v2.entropy_projection(
        dixon_coles_fit[
            "probabilities"
        ],
        matrix,
        targets,
    )

    probabilities = projection[
        "probabilities"
    ]

    effective_residuals = [
        abs(
            v2.effective_fair_probability(
                probabilities,
                item["a_coeff"],
                item["b_coeff"],
            )
            - item["target"]
        )
        for item in enriched
    ]

    boundary_mass = v2.score_boundary_mass(
        probabilities,
        max_goals,
    )

    output = {
        "probabilities": probabilities,

        # Backward-compatible Dixon-Coles fields.
        "lambda_home": (
            dixon_coles_fit[
                "lambda_home"
            ]
        ),
        "lambda_away": (
            dixon_coles_fit[
                "lambda_away"
            ]
        ),
        "rho": dixon_coles_fit["rho"],
        "maximum_prior_residual": (
            dixon_coles_fit[
                "maximum_prior_residual"
            ]
        ),
        "prior_rmse": (
            dixon_coles_fit[
                "prior_rmse"
            ]
        ),

        "minimum_slack": projection[
            "minimum_slack"
        ],
        "allowed_slack": projection[
            "allowed_slack"
        ],
        "maximum_equation_residual": (
            projection[
                "maximum_equation_residual"
            ]
        ),
        "maximum_effective_residual": (
            max(effective_residuals)
            if effective_residuals
            else 0.0
        ),
        "projection_method": projection[
            "projection_method"
        ],
        "projection_baseline_prior": (
            "DIXON_COLES"
        ),
        "constraint_count": len(
            constraints
        ),
        "market_group_count": (
            v2.constraint_group_count(
                constraints
            )
        ),
        "market_family_count": (
            v2.constraint_family_count(
                constraints
            )
        ),
        "boundary_mass": boundary_mass,
        "max_goals": max_goals,

        # Public diagnostic metadata.
        "prior_diagnostics": {
            name: (
                _public_prior_diagnostics(
                    prior_fit
                )
            )
            for name, prior_fit
            in prior_fits.items()
        },
        "prior_errors": prior_errors,

        # Internal arrays are removed from public output.
        "_prior_probabilities": {
            name: prior_fit[
                "probabilities"
            ]
            for name, prior_fit
            in prior_fits.items()
        },
    }

    output["quality_status"] = (
        v2.scenario_quality_status(
            output
        )
    )

    return output


# ============================================================
# 6. Candidate evaluation with pre-projection priors
# ============================================================

def evaluate_model_scenarios(
    candidate,
    scenarios,
    a_coeff,
    b_coeff,
):
    scenario_results = []

    for scenario in scenarios:
        projected_metrics = (
            v2.candidate_metrics(
                probabilities=scenario[
                    "probabilities"
                ],
                a_coeff=a_coeff,
                b_coeff=b_coeff,
                odds=candidate["odds"],
            )
        )

        discrepancy = None

        if (
            scenario.get(
                "direct_devig_target"
            )
            is not None
        ):
            discrepancy = (
                projected_metrics[
                    "effective_fair_probability"
                ]
                - scenario[
                    "direct_devig_target"
                ]
            )

        prior_metrics = {}

        for (
            prior_name,
            prior_probabilities,
        ) in scenario.get(
            "_prior_probabilities",
            {},
        ).items():
            metrics = v2.candidate_metrics(
                probabilities=(
                    prior_probabilities
                ),
                a_coeff=a_coeff,
                b_coeff=b_coeff,
                odds=candidate["odds"],
            )

            prior_metrics[
                prior_name
            ] = {
                **metrics,
                "fit_diagnostics": (
                    scenario.get(
                        "prior_diagnostics",
                        {},
                    ).get(
                        prior_name,
                        {},
                    )
                ),
            }

        scenario_results.append({
            "scenario_id": scenario["id"],
            "period": scenario["period"],
            "source": scenario["source"],
            "source_title": (
                scenario["source_title"]
            ),
            "devig_method": (
                scenario["devig_method"]
            ),
            "scenario_type": (
                scenario["scenario_type"]
            ),
            "excluded_group": scenario.get(
                "excluded_group"
            ),
            "excluded_family": scenario.get(
                "excluded_family"
            ),
            "direct_devig_target": (
                scenario.get(
                    "direct_devig_target"
                )
            ),
            "fair_probability_discrepancy": (
                discrepancy
            ),
            "model_quality": (
                scenario["quality_status"]
            ),
            "projection_baseline_prior": (
                "DIXON_COLES"
            ),
            "pre_projection_priors": (
                prior_metrics
            ),
            "prior_errors": scenario.get(
                "prior_errors",
                {},
            ),
            "model_diagnostics": {
                "lambda_home": (
                    scenario["lambda_home"]
                ),
                "lambda_away": (
                    scenario["lambda_away"]
                ),
                "rho": scenario["rho"],
                "minimum_slack": (
                    scenario["minimum_slack"]
                ),
                "maximum_equation_residual": (
                    scenario[
                        "maximum_equation_residual"
                    ]
                ),
                "maximum_effective_residual": (
                    scenario[
                        "maximum_effective_residual"
                    ]
                ),
                "projection_method": (
                    scenario[
                        "projection_method"
                    ]
                ),
                "projection_baseline_prior": (
                    "DIXON_COLES"
                ),
                "constraint_count": (
                    scenario[
                        "constraint_count"
                    ]
                ),
                "market_group_count": (
                    scenario[
                        "market_group_count"
                    ]
                ),
                "market_family_count": (
                    scenario[
                        "market_family_count"
                    ]
                ),
                "boundary_mass": (
                    scenario["boundary_mass"]
                ),
                "prior_diagnostics": (
                    scenario.get(
                        "prior_diagnostics",
                        {},
                    )
                ),
            },
            **projected_metrics,
        })

    return scenario_results


def summarize_pre_projection_priors(
    scenario_records,
) -> Dict[str, Any]:
    grouped: Dict[
        str,
        List[Dict[str, Any]],
    ] = {}

    for scenario in scenario_records:
        priors = scenario.get(
            "pre_projection_priors",
            {},
        )

        if not isinstance(priors, dict):
            continue

        for prior_name, metrics in (
            priors.items()
        ):
            if isinstance(metrics, dict):
                grouped.setdefault(
                    prior_name,
                    [],
                ).append(metrics)

    comparison = {}

    for prior_name in PRIOR_NAMES:
        records = grouped.get(
            prior_name,
            [],
        )

        if not records:
            comparison[prior_name] = {
                "available": False,
                "scenario_count": 0,
            }
            continue

        comparison[prior_name] = {
            "available": True,
            "scenario_count": len(records),
            "probability": {
                "hit": _summary(
                    record.get(
                        "hit_probability"
                    )
                    for record in records
                ),
                "nonloss": _summary(
                    record.get(
                        "nonloss_probability"
                    )
                    for record in records
                ),
                "full_loss": _summary(
                    record.get(
                        "full_loss"
                    )
                    for record in records
                ),
            },
            "expected_return": _summary(
                record.get(
                    "expected_return"
                )
                for record in records
            ),
            "effective_fair_probability": (
                _summary(
                    record.get(
                        "effective_fair_probability"
                    )
                    for record in records
                )
            ),
            "fair_odds": _summary(
                record.get("fair_odds")
                for record in records
            ),
        }

    median_hits = {
        prior_name: (
            record.get(
                "probability",
                {},
            ).get(
                "hit",
                {},
            ).get("median")
        )
        for prior_name, record
        in comparison.items()
        if record.get("available")
    }

    clean_median_hits = {
        key: value
        for key, value
        in median_hits.items()
        if value is not None
    }

    if clean_median_hits:
        minimum_prior = min(
            clean_median_hits,
            key=clean_median_hits.get,
        )

        maximum_prior = max(
            clean_median_hits,
            key=clean_median_hits.get,
        )

        disparity = {
            "minimum_prior": (
                minimum_prior
            ),
            "minimum_median_hit": (
                clean_median_hits[
                    minimum_prior
                ]
            ),
            "maximum_prior": (
                maximum_prior
            ),
            "maximum_median_hit": (
                clean_median_hits[
                    maximum_prior
                ]
            ),
            "median_hit_range": (
                clean_median_hits[
                    maximum_prior
                ]
                - clean_median_hits[
                    minimum_prior
                ]
            ),
            "median_hit_range_pp": (
                (
                    clean_median_hits[
                        maximum_prior
                    ]
                    - clean_median_hits[
                        minimum_prior
                    ]
                )
                * 100.0
            ),
        }

    else:
        disparity = {
            "minimum_prior": None,
            "minimum_median_hit": None,
            "maximum_prior": None,
            "maximum_median_hit": None,
            "median_hit_range": None,
            "median_hit_range_pp": None,
        }

    return {
        "projection_baseline": (
            "DIXON_COLES"
        ),
        "comparison_stage": (
            "BEFORE_PROJECTION"
        ),
        "priors": comparison,
        "disparity": disparity,
    }


def attach_prior_comparisons(
    output: Dict[str, Any],
) -> None:
    sections = [
        output.get(
            "candidate_markets",
            [],
        ),
        output.get(
            "recommendations",
            [],
        ),
    ]

    for records in sections:
        if not isinstance(records, list):
            continue

        for record in records:
            if not isinstance(record, dict):
                continue

            record[
                "pre_projection_prior_comparison"
            ] = summarize_pre_projection_priors(
                record.get(
                    "scenarios",
                    [],
                )
            )


# ============================================================
# 7. Odds-movement input normalization
# ============================================================

def _movement_outcomes(
    market: str,
) -> Tuple[str, ...]:
    if market in {
        "1X2",
        "HHAD",
    }:
        return (
            "home",
            "draw",
            "away",
        )

    if market == "AH":
        return (
            "home",
            "away",
        )

    return (
        "over",
        "under",
    )


def _convert_movement_odds(
    value: Any,
    odds_format: str,
    label: str,
) -> float:
    number = _safe_float(value)

    if number is None:
        raise ValueError(
            f"{label} must be numeric."
        )

    if odds_format == "HK":
        if number <= 0.0:
            raise ValueError(
                f"{label} HK odds must exceed 0."
            )

        number += 1.0

    return v2.validate_odds(
        number,
        label,
    )


def _infer_movement_role(
    bookmaker_key: str,
    bookmaker_title: str,
    explicit_role: Any,
    primary_source: str,
) -> str:
    role = str(
        explicit_role or ""
    ).strip().upper()

    aliases = {
        "MARKET": "EXTERNAL",
        "BOOK": "EXTERNAL",
        "SHARP": "EXTERNAL",
        "PINNACLE": "PRIMARY",
        "PRIMARY_SOURCE": "PRIMARY",
        "JOCKEY_CLUB": "HKJC",
    }

    role = aliases.get(
        role,
        role,
    )

    if role in {
        "EXTERNAL",
        "PRIMARY",
        "HKJC",
    }:
        return role

    combined = (
        bookmaker_key
        + " "
        + bookmaker_title.lower()
    )

    if (
        "hkjc" in combined
        or "hong_kong_jockey_club"
        in combined
        or "jockey club" in combined
        or "馬會" in bookmaker_title
    ):
        return "HKJC"

    if bookmaker_key == primary_source:
        return "PRIMARY"

    return "EXTERNAL"


def normalize_movement_settings(
    raw: Any,
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    output = copy.deepcopy(
        DEFAULT_MOVEMENT_SETTINGS
    )

    for key in [
        "neutral_threshold_pp",
        "strong_threshold_pp",
        "minimum_agreement",
        "strong_agreement",
        "adjacent_line_tolerance",
    ]:
        if key in raw:
            value = _safe_float(
                raw.get(key)
            )

            if value is None:
                raise ValueError(
                    f"Movement setting {key} "
                    "must be numeric."
                )

            output[key] = value

    if "minimum_external_books" in raw:
        output[
            "minimum_external_books"
        ] = int(
            raw[
                "minimum_external_books"
            ]
        )

    if (
        output["neutral_threshold_pp"]
        < 0.0
    ):
        raise ValueError(
            "neutral_threshold_pp cannot be negative."
        )

    if (
        output["strong_threshold_pp"]
        < output[
            "neutral_threshold_pp"
        ]
    ):
        raise ValueError(
            "strong_threshold_pp cannot be below "
            "neutral_threshold_pp."
        )

    if not (
        0.0
        <= output[
            "minimum_agreement"
        ]
        <= 1.0
    ):
        raise ValueError(
            "minimum_agreement must be between 0 and 1."
        )

    if not (
        0.0
        <= output[
            "strong_agreement"
        ]
        <= 1.0
    ):
        raise ValueError(
            "strong_agreement must be between 0 and 1."
        )

    if (
        output[
            "minimum_external_books"
        ]
        < 1
    ):
        raise ValueError(
            "minimum_external_books must be at least one."
        )

    return output


def normalize_odds_movements(
    raw: Any,
    *,
    primary_source: str,
) -> List[Dict[str, Any]]:
    if raw is None or raw == "":
        return []

    if isinstance(raw, dict):
        raw = raw.get(
            "series",
            raw.get(
                "records",
                raw.get(
                    "odds_movements",
                    [],
                ),
            ),
        )

    if not isinstance(raw, list):
        raise ValueError(
            "odds_movements must be a list or "
            "an object containing series."
        )

    output = []

    for series_index, series in enumerate(
        raw,
        start=1,
    ):
        if not isinstance(series, dict):
            raise ValueError(
                f"Odds movement series #{series_index} "
                "must be an object."
            )

        bookmaker_title = str(
            series.get(
                "bookmaker",
                series.get(
                    "title",
                    series.get(
                        "source",
                        "",
                    ),
                ),
            )
        ).strip()

        bookmaker_key = _normalize_book_key(
            series.get(
                "key",
                bookmaker_title,
            )
        )

        if not bookmaker_key:
            raise ValueError(
                f"Odds movement series #{series_index} "
                "requires a bookmaker."
            )

        if not bookmaker_title:
            bookmaker_title = bookmaker_key

        role = _infer_movement_role(
            bookmaker_key,
            bookmaker_title,
            series.get("role"),
            primary_source,
        )

        period = v2.normalize_period(
            series.get(
                "period",
                "FT",
            )
        )

        market = v2.normalize_market(
            series.get("market")
        )

        team = None

        if market == "TEAM_OU":
            team = v2.normalize_team(
                series.get("team")
            )

        if market in {
            "AH",
            "OU",
            "TEAM_OU",
        }:
            line = v2.validate_quarter_line(
                series.get("line"),
                (
                    f"Odds movement series "
                    f"#{series_index} line"
                ),
            )

        elif market == "HHAD":
            line = v2.validate_integer_line(
                series.get("line"),
                (
                    f"Odds movement series "
                    f"#{series_index} handicap"
                ),
            )

        else:
            line = None

        odds_format = str(
            series.get(
                "odds_format",
                "DECIMAL",
            )
        ).strip().upper()

        format_aliases = {
            "EU": "DECIMAL",
            "EUROPEAN": "DECIMAL",
            "DEC": "DECIMAL",
            "HONG_KONG": "HK",
            "HONGKONG": "HK",
        }

        odds_format = format_aliases.get(
            odds_format,
            odds_format,
        )

        if odds_format not in {
            "DECIMAL",
            "HK",
        }:
            raise ValueError(
                "Movement odds_format must be "
                "DECIMAL or HK."
            )

        observations_raw = series.get(
            "observations",
            series.get(
                "history",
                [],
            ),
        )

        if (
            not isinstance(
                observations_raw,
                list,
            )
            or len(observations_raw) < 2
        ):
            raise ValueError(
                f"Odds movement series #{series_index} "
                "requires at least two observations."
            )

        required_outcomes = (
            _movement_outcomes(
                market
            )
        )

        observations = []

        for observation_index, observation in enumerate(
            observations_raw,
            start=1,
        ):
            if not isinstance(
                observation,
                dict,
            ):
                raise ValueError(
                    f"Movement series #{series_index}, "
                    f"observation #{observation_index} "
                    "must be an object."
                )

            odds_block = observation.get(
                "odds",
                observation,
            )

            normalized_odds = {}

            for outcome in required_outcomes:
                raw_value = odds_block.get(
                    outcome
                )

                normalized_odds[
                    outcome
                ] = _convert_movement_odds(
                    raw_value,
                    odds_format,
                    (
                        f"Movement series "
                        f"#{series_index}, "
                        f"observation "
                        f"#{observation_index}, "
                        f"{outcome}"
                    ),
                )

            observations.append({
                "timestamp": str(
                    observation.get(
                        "timestamp",
                        observation.get(
                            "time",
                            "",
                        ),
                    )
                ).strip(),
                "sequence": int(
                    observation.get(
                        "sequence",
                        observation_index,
                    )
                ),
                "odds": normalized_odds,
            })

        parsed_times = [
            _parse_timestamp(
                observation[
                    "timestamp"
                ]
            )
            for observation in observations
        ]

        if all(
            value is not None
            for value in parsed_times
        ):
            observations = [
                item
                for _, item in sorted(
                    zip(
                        parsed_times,
                        observations,
                    ),
                    key=lambda pair: pair[0],
                )
            ]

        else:
            observations.sort(
                key=lambda item: (
                    item["sequence"]
                )
            )

        output.append({
            "series_id": str(
                series.get(
                    "id",
                    (
                        f"MOVE_{series_index:03d}"
                    ),
                )
            ),
            "bookmaker": bookmaker_key,
            "bookmaker_title": (
                bookmaker_title
            ),
            "role": role,
            "period": period,
            "market": market,
            "line": line,
            "team": team,
            "original_odds_format": (
                odds_format
            ),
            "stored_odds_format": (
                "DECIMAL"
            ),
            "observations": observations,
        })

    return output


# ============================================================
# 8. V3 input validation
# ============================================================

def validate_input_data(
    input_data,
):
    if not isinstance(input_data, dict):
        raise ValueError(
            "Input must be an object."
        )

    original = copy.deepcopy(
        input_data
    )

    settings_raw = original.get(
        "settings",
        {},
    )

    if not isinstance(settings_raw, dict):
        settings_raw = {}

    methods_raw = settings_raw.get(
        "devig_methods",
        v2.CONFIG[
            "DEFAULT_DEVIG_METHODS"
        ],
    )

    if isinstance(methods_raw, str):
        methods_raw = [
            item.strip()
            for item in methods_raw.split(",")
            if item.strip()
        ]

    requested_methods = list(
        dict.fromkeys(
            str(item).strip().upper()
            for item in methods_raw
        )
    )

    if not requested_methods:
        raise ValueError(
            "At least one de-vig method is required."
        )

    for method in requested_methods:
        if method not in (
            SUPPORTED_DEVIG_METHODS
        ):
            raise ValueError(
                f"Unsupported de-vig method: {method}"
            )

    # Let V2 validate all established fields. Replace Shin
    # temporarily only because the original V2 validator does
    # not know the new name.
    validation_proxy = copy.deepcopy(
        original
    )

    proxy_settings = validation_proxy.setdefault(
        "settings",
        {},
    )

    proxy_methods = list(
        dict.fromkeys(
            (
                "MULTIPLICATIVE"
                if method == "SHIN"
                else method
            )
            for method in requested_methods
        )
    )

    proxy_settings[
        "devig_methods"
    ] = proxy_methods

    data = _original_validate_input_data(
        validation_proxy
    )

    data["settings"][
        "devig_methods"
    ] = requested_methods

    movement_settings = (
        normalize_movement_settings(
            settings_raw.get(
                "movement",
                {},
            )
        )
    )

    data["settings"][
        "movement"
    ] = movement_settings

    features_raw = settings_raw.get(
        "features",
        {},
    )

    if not isinstance(features_raw, dict):
        features_raw = {}

    data["settings"][
        "features"
    ][
        "prior_comparison"
    ] = bool(
        features_raw.get(
            "prior_comparison",
            True,
        )
    )

    data["settings"][
        "features"
    ][
        "odds_shift_analysis"
    ] = bool(
        features_raw.get(
            "odds_shift_analysis",
            True,
        )
    )

    data["odds_movements"] = (
        normalize_odds_movements(
            original.get(
                "odds_movements"
            ),
            primary_source=(
                data["settings"][
                    "primary_source"
                ]
            ),
        )
    )

    return data


# ============================================================
# 9. Odds movement calculations
# ============================================================

def _selection_outcome_key(
    market: str,
    selection: str,
) -> str:
    selection = str(
        selection
    ).strip().upper()

    mapping = {
        "HOME": "home",
        "DRAW": "draw",
        "AWAY": "away",
        "OVER": "over",
        "UNDER": "under",
    }

    if selection not in mapping:
        raise ValueError(
            f"Unsupported movement selection: {selection}"
        )

    return mapping[selection]


def _series_line_distance(
    candidate: Dict[str, Any],
    series: Dict[str, Any],
) -> Optional[float]:
    if (
        candidate["period"]
        != series["period"]
        or candidate["market"]
        != series["market"]
    ):
        return None

    if (
        candidate["market"]
        == "TEAM_OU"
        and candidate.get("team")
        != series.get("team")
    ):
        return None

    if candidate["market"] == "1X2":
        return 0.0

    candidate_line = _safe_float(
        candidate.get("line")
    )

    series_line = _safe_float(
        series.get("line")
    )

    if (
        candidate_line is None
        or series_line is None
    ):
        return None

    return abs(
        candidate_line
        - series_line
    )


def _observation_probability(
    series: Dict[str, Any],
    observation: Dict[str, Any],
    selection: str,
    devig_methods: List[str],
) -> Dict[str, Any]:
    outcome_names = _movement_outcomes(
        series["market"]
    )

    odds = [
        observation["odds"][
            outcome
        ]
        for outcome in outcome_names
    ]

    selection_key = (
        _selection_outcome_key(
            series["market"],
            selection,
        )
    )

    selection_index = outcome_names.index(
        selection_key
    )

    method_probabilities = {}

    for method in devig_methods:
        probabilities = (
            devig_probabilities(
                odds,
                method,
            )
        )

        method_probabilities[
            method
        ] = float(
            probabilities[
                selection_index
            ]
        )

    return {
        "timestamp": observation[
            "timestamp"
        ],
        "probability": _median(
            method_probabilities.values()
        ),
        "by_devig_method": (
            method_probabilities
        ),
    }


def _movement_series_record(
    series: Dict[str, Any],
    candidate: Dict[str, Any],
    devig_methods: List[str],
    neutral_threshold: float,
    line_distance: float,
) -> Dict[str, Any]:
    probability_path = [
        _observation_probability(
            series,
            observation,
            candidate["selection"],
            devig_methods,
        )
        for observation
        in series["observations"]
    ]

    values = [
        record["probability"]
        for record in probability_path
    ]

    opening_probability = values[0]
    latest_probability = values[-1]

    probability_change = (
        latest_probability
        - opening_probability
    )

    steps = np.diff(
        np.asarray(
            values,
            dtype=float,
        )
    )

    net_sign = _sign(
        probability_change,
        neutral_threshold,
    )

    meaningful_step_threshold = max(
        neutral_threshold / 2.0,
        1e-6,
    )

    meaningful_steps = [
        float(step)
        for step in steps
        if abs(step)
        >= meaningful_step_threshold
    ]

    if (
        net_sign == 0
        or not meaningful_steps
    ):
        sustained_ratio = None
        reversal_detected = False

    else:
        sustained_ratio = float(
            np.mean([
                1.0
                if _sign(step) == net_sign
                else 0.0
                for step in meaningful_steps
            ])
        )

        reversal_detected = any(
            _sign(step) == -net_sign
            for step in meaningful_steps
        )

    return {
        "series_id": series[
            "series_id"
        ],
        "bookmaker": series[
            "bookmaker"
        ],
        "bookmaker_title": (
            series[
                "bookmaker_title"
            ]
        ),
        "role": series["role"],
        "market": series["market"],
        "line": series.get("line"),
        "line_distance": line_distance,
        "line_match": (
            "EXACT"
            if line_distance <= 1e-8
            else "ADJACENT"
        ),
        "observation_count": len(
            probability_path
        ),
        "opening_timestamp": (
            probability_path[0][
                "timestamp"
            ]
        ),
        "latest_timestamp": (
            probability_path[-1][
                "timestamp"
            ]
        ),
        "opening_probability": (
            opening_probability
        ),
        "latest_probability": (
            latest_probability
        ),
        "probability_change": (
            probability_change
        ),
        "probability_change_pp": (
            probability_change * 100.0
        ),
        "direction": (
            "TOWARD_SELECTION"
            if net_sign > 0
            else (
                "AGAINST_SELECTION"
                if net_sign < 0
                else "NEUTRAL"
            )
        ),
        "sustained_ratio": (
            sustained_ratio
        ),
        "reversal_detected": (
            reversal_detected
        ),
        "probability_path": (
            probability_path
        ),
    }


def _best_series_per_bookmaker(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[
        str,
        List[Dict[str, Any]],
    ] = {}

    for record in records:
        grouped.setdefault(
            record["bookmaker"],
            [],
        ).append(record)

    selected = []

    for bookmaker_records in (
        grouped.values()
    ):
        bookmaker_records.sort(
            key=lambda item: (
                item["line_distance"],
                -item["observation_count"],
            )
        )

        selected.append(
            bookmaker_records[0]
        )

    return selected


def _current_primary_probability(
    data: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    primary_source = data[
        "settings"
    ]["primary_source"]

    primary_book = next(
        (
            book
            for book in data[
                "sharp_books"
            ]
            if book["key"]
            == primary_source
        ),
        None,
    )

    if primary_book is None:
        return {
            "status": "NOT_AVAILABLE",
            "probability": None,
        }

    candidate_line = _safe_float(
        candidate.get("line")
    )

    values = []
    distances = []

    for method in data[
        "settings"
    ]["devig_methods"]:
        constraints = (
            v2.build_book_constraints(
                primary_book,
                candidate["period"],
                method,
            )
        )

        possible = []

        for constraint in constraints:
            if (
                constraint["market"]
                != candidate["market"]
                or constraint["selection"]
                != candidate["selection"]
            ):
                continue

            if (
                candidate["market"]
                == "TEAM_OU"
                and constraint.get("team")
                != candidate.get("team")
            ):
                continue

            if candidate["market"] == "1X2":
                distance = 0.0

            else:
                constraint_line = (
                    _safe_float(
                        constraint.get(
                            "line"
                        )
                    )
                )

                if (
                    candidate_line is None
                    or constraint_line is None
                ):
                    continue

                distance = abs(
                    candidate_line
                    - constraint_line
                )

            possible.append(
                (
                    distance,
                    float(
                        constraint[
                            "target"
                        ]
                    ),
                )
            )

        if not possible:
            continue

        possible.sort(
            key=lambda item: item[0]
        )

        distance, probability = (
            possible[0]
        )

        values.append(probability)
        distances.append(distance)

    probability = _median(values)

    if probability is None:
        return {
            "status": "NOT_AVAILABLE",
            "probability": None,
        }

    maximum_distance = max(
        distances
    )

    return {
        "status": "AVAILABLE",
        "source": primary_source,
        "probability": probability,
        "line_distance": (
            maximum_distance
        ),
        "line_match": (
            "EXACT"
            if maximum_distance <= 1e-8
            else "ADJACENT"
        ),
        "devig_method_count": len(
            values
        ),
    }


def _primary_alignment(
    *,
    primary_probability: Optional[float],
    opening_probability: float,
    external_change: float,
    neutral_threshold: float,
    line_match: str,
) -> Dict[str, Any]:
    if primary_probability is None:
        return {
            "status": "NOT_AVAILABLE",
            "progress": None,
        }

    movement_sign = _sign(
        external_change,
        neutral_threshold,
    )

    if movement_sign == 0:
        return {
            "status": "NEUTRAL",
            "progress": 0.0,
        }

    directional_progress = (
        movement_sign
        * (
            primary_probability
            - opening_probability
        )
    )

    required_confirmation = max(
        neutral_threshold,
        abs(external_change) * 0.35,
    )

    if line_match != "EXACT":
        if directional_progress > 0.0:
            status = "INDIRECT_SUPPORT"
        elif (
            directional_progress
            < -neutral_threshold
        ):
            status = "INDIRECT_CONFLICT"
        else:
            status = "INDIRECT_NEUTRAL"

    elif (
        directional_progress
        >= required_confirmation
    ):
        status = "CONFIRMED"

    elif (
        directional_progress
        <= -neutral_threshold
    ):
        status = "CONFLICT"

    elif directional_progress > 0.0:
        status = "PARTIAL"

    else:
        status = "NEUTRAL"

    return {
        "status": status,
        "progress": directional_progress,
        "progress_pp": (
            directional_progress
            * 100.0
        ),
    }


def _hkjc_response(
    hkjc_record: Optional[
        Dict[str, Any]
    ],
    external_change: float,
    neutral_threshold: float,
) -> Dict[str, Any]:
    if hkjc_record is None:
        return {
            "status": "NOT_AVAILABLE",
            "probability_change": None,
            "probability_change_pp": None,
            "follow_ratio": None,
        }

    hkjc_change = float(
        hkjc_record[
            "probability_change"
        ]
    )

    external_sign = _sign(
        external_change,
        neutral_threshold,
    )

    hkjc_sign = _sign(
        hkjc_change,
        neutral_threshold,
    )

    if external_sign == 0:
        status = "NEUTRAL"
        follow_ratio = None

    elif hkjc_sign == -external_sign:
        status = "CONFLICT"
        follow_ratio = (
            abs(hkjc_change)
            / max(
                abs(external_change),
                1e-12,
            )
        )

    elif hkjc_sign == 0:
        status = "LAGGING"
        follow_ratio = 0.0

    else:
        follow_ratio = (
            abs(hkjc_change)
            / max(
                abs(external_change),
                1e-12,
            )
        )

        if follow_ratio >= 0.80:
            status = "FOLLOWED"

        elif follow_ratio >= 0.25:
            status = "PARTIALLY_FOLLOWED"

        else:
            status = "LAGGING"

    return {
        "status": status,
        "probability_change": (
            hkjc_change
        ),
        "probability_change_pp": (
            hkjc_change * 100.0
        ),
        "follow_ratio": (
            follow_ratio
        ),
        "line_match": hkjc_record.get(
            "line_match"
        ),
        "bookmaker_record": (
            hkjc_record
        ),
    }


def analyze_candidate_movement(
    data: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    settings = data[
        "settings"
    ]["movement"]

    neutral_threshold = (
        settings[
            "neutral_threshold_pp"
        ]
        / 100.0
    )

    strong_threshold = (
        settings[
            "strong_threshold_pp"
        ]
        / 100.0
    )

    line_tolerance = settings[
        "adjacent_line_tolerance"
    ]

    matching_records = []

    for series in data.get(
        "odds_movements",
        [],
    ):
        distance = _series_line_distance(
            candidate,
            series,
        )

        if distance is None:
            continue

        if distance > line_tolerance + 1e-8:
            continue

        try:
            record = _movement_series_record(
                series=series,
                candidate=candidate,
                devig_methods=(
                    data["settings"][
                        "devig_methods"
                    ]
                ),
                neutral_threshold=(
                    neutral_threshold
                ),
                line_distance=distance,
            )

            matching_records.append(
                record
            )

        except Exception:
            # A malformed individual series should not erase the
            # model result. Validation normally catches these.
            continue

    matching_records = (
        _best_series_per_bookmaker(
            matching_records
        )
    )

    external_records = [
        record
        for record in matching_records
        if record["role"]
        == "EXTERNAL"
    ]

    hkjc_records = [
        record
        for record in matching_records
        if record["role"] == "HKJC"
    ]

    primary_history_records = [
        record
        for record in matching_records
        if record["role"]
        == "PRIMARY"
    ]

    hkjc_record = (
        sorted(
            hkjc_records,
            key=lambda item: (
                item["line_distance"],
                -item[
                    "observation_count"
                ],
            ),
        )[0]
        if hkjc_records
        else None
    )

    if not external_records:
        return {
            "status": "NOT_AVAILABLE",
            "verdict": "NEUTRAL",
            "strength": "NONE",
            "reason": (
                "No matching external bookmaker "
                "movement series."
            ),
            "external_book_count": 0,
            "external_books": [],
            "primary_history_books": (
                primary_history_records
            ),
            "model_probability_adjustment": None,
            "adjustment_note": (
                "No numerical model adjustment is "
                "made without historical calibration."
            ),
        }

    changes = [
        record[
            "probability_change"
        ]
        for record in external_records
    ]

    openings = [
        record[
            "opening_probability"
        ]
        for record in external_records
    ]

    latest = [
        record[
            "latest_probability"
        ]
        for record in external_records
    ]

    consensus_change = float(
        np.median(changes)
    )

    consensus_opening = float(
        np.median(openings)
    )

    consensus_latest = float(
        np.median(latest)
    )

    consensus_sign = _sign(
        consensus_change,
        neutral_threshold,
    )

    meaningful_records = [
        record
        for record in external_records
        if abs(
            record[
                "probability_change"
            ]
        )
        >= neutral_threshold
    ]

    if consensus_sign == 0:
        agreement_count = 0

    else:
        agreement_count = sum(
            1
            for record in external_records
            if _sign(
                record[
                    "probability_change"
                ],
                neutral_threshold,
            )
            == consensus_sign
        )

    agreement_ratio = (
        agreement_count
        / len(external_records)
        if external_records
        else 0.0
    )

    reversal_count = sum(
        1
        for record in external_records
        if record[
            "reversal_detected"
        ]
    )

    reversal_ratio = (
        reversal_count
        / len(external_records)
    )

    exact_count = sum(
        1
        for record in external_records
        if record["line_match"]
        == "EXACT"
    )

    primary = _current_primary_probability(
        data,
        candidate,
    )

    primary_result = _primary_alignment(
        primary_probability=(
            primary.get("probability")
        ),
        opening_probability=(
            consensus_opening
        ),
        external_change=(
            consensus_change
        ),
        neutral_threshold=(
            neutral_threshold
        ),
        line_match=primary.get(
            "line_match",
            "EXACT",
        ),
    )

    primary_result.update(
        primary
    )

    hkjc_result = _hkjc_response(
        hkjc_record,
        consensus_change,
        neutral_threshold,
    )

    minimum_books = settings[
        "minimum_external_books"
    ]

    minimum_agreement = settings[
        "minimum_agreement"
    ]

    strong_agreement = settings[
        "strong_agreement"
    ]

    enough_books = (
        len(external_records)
        >= minimum_books
    )

    usable_agreement = (
        agreement_ratio
        >= minimum_agreement
    )

    strong_external = bool(
        len(external_records) >= max(
            3,
            minimum_books,
        )
        and abs(consensus_change)
        >= strong_threshold
        and agreement_ratio
        >= strong_agreement
        and reversal_ratio <= 0.25
        and exact_count
        >= max(
            2,
            math.ceil(
                len(external_records)
                / 2
            ),
        )
    )

    primary_status = (
        primary_result.get("status")
    )

    primary_conflict = (
        primary_status
        in {
            "CONFLICT",
            "INDIRECT_CONFLICT",
        }
    )

    primary_confirmed = (
        primary_status
        in {
            "CONFIRMED",
            "PARTIAL",
            "INDIRECT_SUPPORT",
        }
    )

    public_bias_risk = bool(
        consensus_sign != 0
        and (
            primary_conflict
            or not enough_books
            or not usable_agreement
        )
    )

    if (
        consensus_sign == 0
        or not meaningful_records
    ):
        verdict = "NEUTRAL"
        strength = "NONE"
        reason = (
            "External market movement is below "
            "the configured materiality threshold."
        )

    elif (
        not enough_books
        or not usable_agreement
    ):
        verdict = "NEUTRAL"
        strength = "WEAK"
        reason = (
            "External movement lacks sufficient "
            "bookmaker breadth or agreement."
        )

    elif primary_conflict:
        verdict = "NEUTRAL"
        strength = "CONFLICTING"
        reason = (
            "External bookmakers moved, but the "
            "current Pinnacle level does not confirm "
            "the same direction."
        )

    elif consensus_sign > 0:
        verdict = "SUPPORTED"

        if (
            strong_external
            and primary_status
            == "CONFIRMED"
        ):
            strength = "STRONG"

        elif primary_confirmed:
            strength = "MODERATE"

        else:
            strength = "WEAK"

        reason = (
            "The external consensus moved toward "
            "this selection."
        )

    else:
        verdict = "CONFLICTED"

        if (
            strong_external
            and primary_status
            == "CONFIRMED"
        ):
            strength = "STRONG"

        elif primary_confirmed:
            strength = "MODERATE"

        else:
            strength = "WEAK"

        reason = (
            "The external consensus moved against "
            "this selection."
        )

    if hkjc_result["status"] in {
        "FOLLOWED",
        "PARTIALLY_FOLLOWED",
    }:
        actionability = (
            "MOVEMENT_PARTLY_OR_FULLY_REFLECTED_BY_HKJC"
        )

    elif hkjc_result["status"] == "LAGGING":
        actionability = (
            "POSSIBLE_HKJC_LAG"
        )

    elif hkjc_result["status"] == "CONFLICT":
        actionability = (
            "HKJC_EXTERNAL_MARKET_CONFLICT"
        )

    else:
        actionability = (
            "HKJC_RESPONSE_NOT_ESTABLISHED"
        )

    return {
        "status": "COMPLETED",
        "verdict": verdict,
        "strength": strength,
        "reason": reason,
        "actionability": (
            actionability
        ),

        "selection_direction": (
            "TOWARD_SELECTION"
            if consensus_sign > 0
            else (
                "AGAINST_SELECTION"
                if consensus_sign < 0
                else "NEUTRAL"
            )
        ),

        "external_book_count": len(
            external_records
        ),
        "meaningful_external_book_count": (
            len(meaningful_records)
        ),
        "agreement_count": (
            agreement_count
        ),
        "agreement_ratio": (
            agreement_ratio
        ),
        "reversal_count": (
            reversal_count
        ),
        "reversal_ratio": (
            reversal_ratio
        ),
        "exact_line_book_count": (
            exact_count
        ),

        "consensus_opening_probability": (
            consensus_opening
        ),
        "consensus_latest_probability": (
            consensus_latest
        ),
        "market_implied_probability_change": (
            consensus_change
        ),
        "market_implied_probability_change_pp": (
            consensus_change * 100.0
        ),

        "external_consensus_quality": (
            "STRONG"
            if strong_external
            else (
                "USABLE"
                if (
                    enough_books
                    and usable_agreement
                )
                else "NOISY"
            )
        ),

        "pinnacle_confirmation": (
            primary_result
        ),
        "hkjc_response": hkjc_result,
        "public_bias_risk": (
            public_bias_risk
        ),

        "external_books": (
            external_records
        ),
        "primary_history_books": (
            primary_history_records
        ),

        # Deliberately not converted into a model adjustment.
        "model_probability_adjustment": None,
        "adjustment_note": (
            "This is the observed de-vigged market-implied "
            "probability shift. It is not automatically added "
            "to the model probability because no historical "
            "movement-to-outcome calibration was supplied."
        ),
    }


def attach_movement_analysis(
    output: Dict[str, Any],
    data: Dict[str, Any],
) -> None:
    movement_enabled = data[
        "settings"
    ]["features"].get(
        "odds_shift_analysis",
        True,
    )

    movements = data.get(
        "odds_movements",
        [],
    )

    if not movement_enabled:
        output["odds_shift_analysis"] = {
            "status": "DISABLED",
            "candidate_audits": [],
        }
        return

    if not movements:
        output["odds_shift_analysis"] = {
            "status": "NOT_PROVIDED",
            "candidate_audits": [],
            "note": (
                "Provide structured odds_movements JSON "
                "to enable movement analysis."
            ),
        }
        return

    candidate_audits = []

    candidate_lookup = {}

    for candidate in output.get(
        "candidate_markets",
        [],
    ):
        if not isinstance(candidate, dict):
            continue

        audit = analyze_candidate_movement(
            data,
            candidate,
        )

        candidate[
            "odds_shift"
        ] = audit

        candidate_id = str(
            candidate.get("id", "")
        )

        if candidate_id:
            candidate_lookup[
                candidate_id
            ] = audit

        candidate_audits.append({
            "id": candidate.get("id"),
            "label": candidate.get(
                "label"
            ),
            "period": candidate.get(
                "period"
            ),
            "market": candidate.get(
                "market"
            ),
            "selection": candidate.get(
                "selection"
            ),
            "line": candidate.get(
                "line"
            ),
            **audit,
        })

    for recommendation in output.get(
        "recommendations",
        [],
    ):
        recommendation_id = str(
            recommendation.get(
                "id",
                "",
            )
        )

        audit = candidate_lookup.get(
            recommendation_id
        )

        if audit is not None:
            recommendation[
                "odds_shift"
            ] = copy.deepcopy(
                audit
            )

    verdict_counts = {
        "SUPPORTED": 0,
        "NEUTRAL": 0,
        "CONFLICTED": 0,
    }

    for audit in candidate_audits:
        verdict = audit.get(
            "verdict",
            "NEUTRAL",
        )

        verdict_counts[verdict] = (
            verdict_counts.get(
                verdict,
                0,
            )
            + 1
        )

    output["odds_shift_analysis"] = {
        "status": "COMPLETED",
        "movement_series_count": len(
            movements
        ),
        "settings": data[
            "settings"
        ]["movement"],
        "verdict_counts": (
            verdict_counts
        ),
        "candidate_audits": (
            candidate_audits
        ),
        "probability_interpretation": (
            "market_implied_probability_change_pp is "
            "the observed de-vigged market shift, not "
            "an automatically applied model adjustment."
        ),
    }


# ============================================================
# 10. Install V3 extensions into V2 execution namespace
# ============================================================

v2.devig_probabilities = (
    devig_probabilities
)

v2.validate_input_data = (
    validate_input_data
)

v2.build_market_model = (
    build_market_model
)

v2.evaluate_model_scenarios = (
    evaluate_model_scenarios
)


# ============================================================
# 11. Public V3 engine entry point
# ============================================================

def run_engine(
    input_data,
):
    """
    Execute the complete V2 reconstruction pipeline using the
    V3 replacements installed above, then attach V3-only
    diagnostics and odds-movement analysis.
    """

    # Validate once here so movement data is available for the
    # post-processing stage. V2 will validate again internally.
    normalized_data = validate_input_data(
        input_data
    )

    output = _original_run_engine(
        copy.deepcopy(input_data)
    )

    output["engine"] = {
        "name": ENGINE_NAME,
        "version": ENGINE_VERSION,
        "generated_at_utc": (
            output.get(
                "engine",
                {},
            ).get(
                "generated_at_utc"
            )
        ),
    }

    output["settings"] = (
        normalized_data["settings"]
    )

    output["input_snapshot"] = (
        normalized_data
    )

    methodology = output.setdefault(
        "methodology",
        {},
    )

    methodology.update({
        "projection_baseline_prior": (
            "DIXON_COLES"
        ),
        "pre_projection_prior_comparison": [
            "DIXON_COLES",
            "INDEPENDENT_POISSON",
            "COM_POISSON",
        ],
        "alternative_priors_projected": False,
        "prior_comparison_role": (
            "Sensitivity diagnostic before market "
            "projection. Dixon-Coles remains the "
            "actual projection baseline."
        ),
        "devig_methods_supported": [
            "MULTIPLICATIVE",
            "POWER",
            "SHIN",
        ],
        "odds_shift_role": (
            "Separate directional audit using "
            "external consensus, current Pinnacle "
            "confirmation and HKJC response."
        ),
        "odds_shift_used_as_hard_gate": False,
        "odds_shift_automatically_changes_probability": (
            False
        ),
    })

    if normalized_data[
        "settings"
    ]["features"].get(
        "prior_comparison",
        True,
    ):
        attach_prior_comparisons(
            output
        )

    attach_movement_analysis(
        output,
        normalized_data,
    )

    return v2.to_builtin(
        output
    )


# ============================================================
# 12. Compatibility exports required by the Streamlit app
# ============================================================

minimum_ht_ft_violation = (
    v2.minimum_ht_ft_violation
)

candidate_metrics = v2.candidate_metrics
settlement_coefficients = (
    v2.settlement_coefficients
)
effective_fair_probability = (
    v2.effective_fair_probability
)
multiplicative_devig = (
    v2.multiplicative_devig
)
power_devig = v2.power_devig
summary_statistics = (
    v2.summary_statistics
)
to_builtin = v2.to_builtin


def __getattr__(
    name: str,
):
    """
    Preserve access to all unchanged V2 utilities without
    copying the complete V2 source into this extension file.
    """

    return getattr(
        v2,
        name,
    )
