"""Research utilities for transparent, offline slope-risk experiments.

These functions are screening/research implementations. Their inputs, numerical
assumptions, and validation limits are intentionally explicit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Slice:
    """One vertical slice for a circular limit-equilibrium analysis."""

    weight_kn: float
    base_angle_deg: float
    width_m: float
    cohesion_kpa: float
    friction_angle_deg: float
    pore_pressure_kpa: float = 0.0

    def validate(self) -> None:
        values = asdict(self).values()
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Slice inputs must be finite.")
        if self.weight_kn <= 0 or self.width_m <= 0:
            raise ValueError("Slice weight and width must be positive.")
        if not 0 <= self.friction_angle_deg < 90:
            raise ValueError("Friction angle must be in [0, 90) degrees.")
        if self.cohesion_kpa < 0 or self.pore_pressure_kpa < 0:
            raise ValueError("Cohesion and pore pressure must be non-negative.")
        if not -89 < self.base_angle_deg < 89:
            raise ValueError("Slice base angle must be between -89 and 89 degrees.")


def bishop_simplified(
    slices: Sequence[Slice], *, tolerance: float = 1e-8, max_iterations: int = 500
) -> float:
    """Solve Bishop's simplified circular method by fixed-point iteration.

    Formula: F = sum[(c'b + (W - u b)tan(phi')/m_alpha)] /
    sum[W sin(alpha)], where m_alpha = cos(alpha) +
    sin(alpha)tan(phi')/F. Assumes circular slip and interslice shear forces
    cancel in overall moment equilibrium. This is not a general 2D/3D solver.
    """
    if not slices:
        raise ValueError("At least one slice is required.")
    for item in slices:
        item.validate()
    driving = sum(
        item.weight_kn * math.sin(math.radians(item.base_angle_deg))
        for item in slices
    )
    if driving <= 0:
        raise ValueError("Slice set must have positive net driving moment.")

    factor = 1.0
    for _ in range(max_iterations):
        resisting = 0.0
        for item in slices:
            alpha = math.radians(item.base_angle_deg)
            phi = math.radians(item.friction_angle_deg)
            normal_effective = max(0.0, item.weight_kn - item.pore_pressure_kpa * item.width_m)
            m_alpha = math.cos(alpha) + math.sin(alpha) * math.tan(phi) / factor
            if m_alpha <= 0:
                raise ValueError("Invalid slice geometry produced a non-positive m_alpha.")
            resisting += (
                item.cohesion_kpa * item.width_m
                + normal_effective * math.tan(phi)
            ) / m_alpha
        next_factor = resisting / driving
        if not math.isfinite(next_factor) or next_factor <= 0:
            raise ValueError("Bishop iteration did not produce a positive finite FoS.")
        if abs(next_factor - factor) <= tolerance:
            return next_factor
        factor = next_factor
    raise RuntimeError("Bishop simplified method did not converge.")


def janbu_simplified(slices: Sequence[Slice]) -> float:
    """Compute uncorrected Janbu simplified force-equilibrium FoS.

    Uses F = sum[(c'b + (W-u b)tan(phi'))/cos(alpha)] /
    sum[W tan(alpha)]. Omits the empirical Janbu correction factor; this
    approximation should not be used for design acceptance.
    """
    if not slices:
        raise ValueError("At least one slice is required.")
    for item in slices:
        item.validate()
    resistance = 0.0
    driving = 0.0
    for item in slices:
        alpha = math.radians(item.base_angle_deg)
        phi = math.radians(item.friction_angle_deg)
        normal_effective = max(0.0, item.weight_kn - item.pore_pressure_kpa * item.width_m)
        resistance += (
            item.cohesion_kpa * item.width_m + normal_effective * math.tan(phi)
        ) / math.cos(alpha)
        driving += item.weight_kn * math.tan(alpha)
    if driving <= 0:
        raise ValueError("Slice set must have positive driving force.")
    return resistance / driving


def hoek_brown_equivalent_mc(
    *,
    intact_ucs_mpa: float,
    mi: float,
    gsi: float,
    disturbance_factor: float,
    sigma3_max_mpa: float,
    sample_count: int = 200,
) -> tuple[float, float]:
    """Fit equivalent Mohr-Coulomb c (kPa), phi (degrees) to Hoek-Brown.

    A least-squares line is fitted in shear-normal stress space across
    [0, sigma3_max]. Equivalent parameters therefore depend on that chosen
    confining-stress range; this is a conversion aid, not a unique material law.
    """
    params = (intact_ucs_mpa, mi, gsi, disturbance_factor, sigma3_max_mpa)
    if not all(math.isfinite(float(value)) for value in params):
        raise ValueError("Hoek-Brown inputs must be finite.")
    if intact_ucs_mpa <= 0 or mi <= 0 or sigma3_max_mpa <= 0:
        raise ValueError("UCS, mi, and maximum confining stress must be positive.")
    if not 0 <= disturbance_factor <= 1 or not 0 <= gsi <= 100:
        raise ValueError("Disturbance must be in [0, 1] and GSI in [0, 100].")
    if sample_count < 3:
        raise ValueError("At least three stress samples are required.")
    mb = mi * math.exp((gsi - 100.0) / (28.0 - 14.0 * disturbance_factor))
    s = math.exp((gsi - 100.0) / (9.0 - 3.0 * disturbance_factor))
    a = 0.5 + (math.exp(-gsi / 15.0) - math.exp(-20.0 / 3.0)) / 6.0
    sigma3 = np.linspace(0.0, sigma3_max_mpa, sample_count)
    sigma1 = sigma3 + intact_ucs_mpa * (mb * sigma3 / intact_ucs_mpa + s) ** a
    normal_stress = (sigma1 + sigma3) / 2.0
    shear_stress = (sigma1 - sigma3) / 2.0
    slope, intercept = np.polyfit(normal_stress, shear_stress, deg=1)
    if slope <= 0 or intercept < 0:
        raise ValueError("Hoek-Brown envelope did not yield physical equivalent MC parameters.")
    return float(intercept * 1000.0), float(math.degrees(math.atan(float(slope))))


def kinematic_planar_feasible(
    *, slope_dip_deg: float, slope_dip_direction_deg: float,
    joint_dip_deg: float, joint_dip_direction_deg: float,
    friction_angle_deg: float,
) -> bool:
    """Screen planar sliding using daylighting, dip, and friction conditions.

    Assumes a single planar joint and no cohesion, water, or lateral restraint.
    Direction differences use the acute angular distance.
    """
    values = (slope_dip_deg, slope_dip_direction_deg, joint_dip_deg,
              joint_dip_direction_deg, friction_angle_deg)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Kinematic inputs must be finite.")
    if not 0 <= slope_dip_deg < 90 or not 0 <= joint_dip_deg < 90:
        raise ValueError("Dip angles must be in [0, 90) degrees.")
    if not 0 <= friction_angle_deg < 90:
        raise ValueError("Friction angle must be in [0, 90) degrees.")
    direction_delta = abs((slope_dip_direction_deg - joint_dip_direction_deg + 180) % 360 - 180)
    return direction_delta <= 20.0 and friction_angle_deg < joint_dip_deg < slope_dip_deg


def green_ampt_infiltration_mm(
    *, rainfall_mm: float, duration_hours: float, saturated_conductivity_mm_h: float,
    suction_head_mm: float, moisture_deficit: float,
) -> float:
    """Estimate cumulative infiltration with a bounded Green-Ampt time step.

    Integrates dF/dt = Ksat*(1 + psi*delta_theta/F) by explicit substeps and
    caps infiltration at rainfall. Assumes homogeneous soil, ponded infiltration
    when rainfall exceeds capacity, and no runoff routing or spatial variation.
    """
    args = (rainfall_mm, duration_hours, saturated_conductivity_mm_h,
            suction_head_mm, moisture_deficit)
    if not all(math.isfinite(float(value)) for value in args):
        raise ValueError("Infiltration inputs must be finite.")
    if min(args) < 0 or moisture_deficit > 1:
        raise ValueError("Rain, duration, conductivity, suction, and deficit are out of range.")
    if duration_hours == 0 or rainfall_mm == 0 or saturated_conductivity_mm_h == 0:
        return 0.0
    steps = max(1, math.ceil(duration_hours * 60))
    dt_hours = duration_hours / steps
    rainfall_rate = rainfall_mm / duration_hours
    infiltrated = 0.0
    wetting_term = suction_head_mm * moisture_deficit
    for _ in range(steps):
        capacity = saturated_conductivity_mm_h * (
            1.0 + wetting_term / max(infiltrated, 1e-6)
        )
        infiltrated += min(rainfall_rate, capacity) * dt_hours
        infiltrated = min(rainfall_mm, infiltrated)
    return infiltrated


def monte_carlo_probability_of_failure(
    *, means: Sequence[float], standard_deviations: Sequence[float],
    factor_of_safety: Callable[[np.ndarray], float], sample_count: int = 10_000, seed: int = 42,
    failure_threshold: float = 1.0,
) -> tuple[float, list[float]]:
    """Estimate P(FoS < threshold) for independent normal input variables."""
    if len(means) == 0 or len(means) != len(standard_deviations):
        raise ValueError("Means and standard deviations must be non-empty and the same length.")
    if sample_count < 2 or any(sd < 0 for sd in standard_deviations):
        raise ValueError("At least two samples and non-negative deviations are required.")
    if not all(math.isfinite(float(v)) for v in [*means, *standard_deviations, failure_threshold]):
        raise ValueError("Monte Carlo inputs must be finite.")
    rng = np.random.default_rng(seed)
    samples = rng.normal(
        loc=np.asarray(means, dtype=float),
        scale=np.asarray(standard_deviations, dtype=float),
        size=(sample_count, len(means)),
    )
    values = [float(factor_of_safety(row)) for row in samples]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Factor-of-safety function returned non-finite values.")
    return float(np.mean(np.asarray(values) < failure_threshold)), values


def form_hlrf(
    *, means: Sequence[float], standard_deviations: Sequence[float],
    limit_state: Callable[[np.ndarray], float], tolerance: float = 1e-6, max_iterations: int = 200,
) -> tuple[float, list[float]]:
    """Estimate FORM reliability index using Hasofer-Lind-Rackwitz-Fiessler.

    Independent normal variables are assumed. Numerical central differences
    estimate the gradient. Returns beta and the design point in original units.
    """
    mean = np.asarray(means, dtype=float)
    std = np.asarray(standard_deviations, dtype=float)
    if mean.size == 0 or mean.shape != std.shape or np.any(std <= 0):
        raise ValueError("FORM requires matched means and strictly positive deviations.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("FORM inputs must be finite.")

    def g(z: np.ndarray) -> float:
        value = float(limit_state(mean + std * z))
        if not math.isfinite(value):
            raise ValueError("Limit-state function returned a non-finite value.")
        return value

    z = np.zeros_like(mean)
    for _ in range(max_iterations):
        value = g(z)
        gradient = np.empty_like(z)
        for index in range(len(z)):
            step = 1e-5 * max(1.0, abs(float(z[index])))
            plus, minus = z.copy(), z.copy()
            plus[index] += step
            minus[index] -= step
            gradient[index] = (g(plus) - g(minus)) / (2 * step)
        norm_sq = float(np.dot(gradient, gradient))
        if norm_sq <= 1e-20:
            raise ValueError("FORM gradient is zero at the current point.")
        next_z = ((np.dot(gradient, z) - value) / norm_sq) * gradient
        if np.linalg.norm(next_z - z) <= tolerance and abs(g(next_z)) <= tolerance * 10:
            return float(np.linalg.norm(next_z)), (mean + std * next_z).tolist()
        z = next_z
    raise RuntimeError("FORM HLRF iteration did not converge.")


def sample_correlated_random_field(
    *, positions_m: Sequence[float], mean: float, standard_deviation: float,
    correlation_length_m: float, sample_count: int = 1000, seed: int = 42,
) -> np.ndarray:
    """Sample a 1D stationary Gaussian field with exponential covariance."""
    positions = np.asarray(positions_m, dtype=float)
    if positions.ndim != 1 or positions.size < 2 or not np.isfinite(positions).all():
        raise ValueError("At least two finite one-dimensional positions are required.")
    if not math.isfinite(mean) or standard_deviation < 0 or correlation_length_m <= 0:
        raise ValueError("Mean, standard deviation, and correlation length are invalid.")
    if not math.isfinite(standard_deviation) or not math.isfinite(correlation_length_m):
        raise ValueError("Random-field parameters must be finite.")
    if sample_count < 2 or sample_count > 100000:
        raise ValueError("Random-field sample count must be 2..100000.")
    distances = np.abs(positions[:, None] - positions[None, :])
    covariance = standard_deviation**2 * np.exp(-distances / correlation_length_m)
    covariance += np.eye(len(positions)) * max(1e-12, standard_deviation**2 * 1e-10)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(
        mean=np.full(len(positions), mean), cov=covariance,
        size=sample_count, check_valid="raise",
    )


def cost_sensitive_threshold(
    probabilities: Sequence[float], labels: Sequence[int], *,
    false_negative_cost: float, false_positive_cost: float,
    thresholds: Iterable[float] = tuple(np.linspace(0.01, 0.99, 99)),
) -> tuple[float, float]:
    """Choose a probability threshold minimizing empirical binary decision cost."""
    probability = np.asarray(probabilities, dtype=float)
    target = np.asarray(labels, dtype=int)
    if probability.size == 0 or probability.shape != target.shape:
        raise ValueError("Probabilities and labels must be non-empty and the same shape.")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if np.any((target != 0) & (target != 1)):
        raise ValueError("Labels must be binary (0 or 1).")
    if min(false_negative_cost, false_positive_cost) < 0:
        raise ValueError("Decision costs must be non-negative.")
    candidates = list(thresholds)
    if not candidates or any(not math.isfinite(float(t)) or not 0 <= t <= 1 for t in candidates):
        raise ValueError("Thresholds must be a non-empty sequence in [0, 1].")
    scored = []
    for threshold in candidates:
        predicted = probability >= threshold
        fn = np.sum((target == 1) & ~predicted)
        fp = np.sum((target == 0) & predicted)
        scored.append((float(fn * false_negative_cost + fp * false_positive_cost), float(threshold)))
    cost, threshold = min(scored, key=lambda item: (item[0], item[1]))
    return threshold, cost / len(target)


class MonotonicLogisticModel:
    """Binary logistic regression with projected coefficient sign constraints."""

    def __init__(self, monotonic_constraints: Sequence[int], *, l2: float = 1e-3):
        if not monotonic_constraints or any(value not in {-1, 0, 1} for value in monotonic_constraints):
            raise ValueError("Monotonic constraints must be a non-empty sequence of -1, 0, or 1.")
        if l2 < 0 or not math.isfinite(l2):
            raise ValueError("L2 strength must be finite and non-negative.")
        self.constraints = np.asarray(monotonic_constraints, dtype=float)
        self.l2 = float(l2)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, features: Sequence[Sequence[float]], labels: Sequence[int], *, max_iterations: int = 5000) -> "MonotonicLogisticModel":
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.constraints) or y.shape != (x.shape[0],) or x.shape[0] < 2:
            raise ValueError("Feature matrix, labels, and constraint dimensions must match.")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y != 0) & (y != 1)):
            raise ValueError("Features must be finite and labels binary.")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        z = (x - self.mean_) / self.scale_
        coef = np.zeros(x.shape[1], dtype=float)
        intercept = 0.0
        spectral = float(np.linalg.norm(z, ord=2) ** 2 / (4 * len(y))) + self.l2
        learning_rate = 1.0 / max(spectral, 1e-6)
        for _ in range(max_iterations):
            logits = np.clip(z @ coef + intercept, -40, 40)
            predicted = 1.0 / (1.0 + np.exp(-logits))
            error = predicted - y
            next_coef = coef - learning_rate * ((z.T @ error) / len(y) + self.l2 * coef)
            next_coef = np.where(self.constraints > 0, np.maximum(next_coef, 0), next_coef)
            next_coef = np.where(self.constraints < 0, np.minimum(next_coef, 0), next_coef)
            next_intercept = intercept - learning_rate * float(error.mean())
            if np.linalg.norm(next_coef - coef) + abs(next_intercept - intercept) < 1e-8:
                coef, intercept = next_coef, next_intercept
                break
            coef, intercept = next_coef, next_intercept
        self.coef_, self.intercept_ = coef, float(intercept)
        return self

    def predict_proba(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.constraints) or not np.isfinite(x).all():
            raise ValueError("Prediction features must be finite with the fitted dimensions.")
        logits = np.clip(((x - self.mean_) / self.scale_) @ self.coef_ + self.intercept_, -40, 40)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack((1.0 - positive, positive))


def conformal_prediction_sets(
    calibration_probabilities: Sequence[Sequence[float]],
    calibration_labels: Sequence[int],
    test_probabilities: Sequence[Sequence[float]],
    *, alpha: float = 0.1,
) -> dict[str, object]:
    """Return split-conformal class sets using 1-p_true nonconformity scores."""
    calibration = np.asarray(calibration_probabilities, dtype=float)
    labels = np.asarray(calibration_labels, dtype=int)
    test = np.asarray(test_probabilities, dtype=float)
    if calibration.ndim != 2 or test.ndim != 2 or calibration.shape[1] != test.shape[1]:
        raise ValueError("Calibration and test probabilities must have matching class dimensions.")
    if labels.shape != (calibration.shape[0],) or len(labels) == 0:
        raise ValueError("Calibration labels must align with a non-empty calibration set.")
    if not 0 < alpha < 1 or not np.isfinite(calibration).all() or not np.isfinite(test).all():
        raise ValueError("Probabilities must be finite and alpha must lie in (0, 1).")
    if np.any((calibration < 0) | (calibration > 1)) or np.any((test < 0) | (test > 1)):
        raise ValueError("Probabilities must lie in [0, 1].")
    if np.any((labels < 0) | (labels >= calibration.shape[1])):
        raise ValueError("Calibration label is outside the probability class range.")
    scores = 1.0 - calibration[np.arange(len(labels)), labels]
    rank = min(len(scores), math.ceil((len(scores) + 1) * (1 - alpha)))
    quantile = float(np.sort(scores)[rank - 1])
    prediction_sets = [
        [class_id for class_id, probability in enumerate(row) if 1.0 - probability <= quantile]
        for row in test
    ]
    return {
        "quantile": quantile,
        "alpha": alpha,
        "prediction_sets": prediction_sets,
        "assumption": "Split conformal marginal coverage relies on exchangeable calibration and future examples.",
    }


def abstention_curve(
    probabilities: Sequence[Sequence[float]], labels: Sequence[int],
    *, confidence_thresholds: Sequence[float] = tuple(np.linspace(0.25, 0.95, 15)),
) -> list[dict[str, float | int]]:
    """Measure coverage and selective accuracy as low-confidence cases abstain."""
    proba = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if proba.ndim != 2 or y.shape != (proba.shape[0],) or len(y) == 0:
        raise ValueError("Probabilities and labels must align and be non-empty.")
    if not np.isfinite(proba).all() or np.any((proba < 0) | (proba > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if np.any((y < 0) | (y >= proba.shape[1])):
        raise ValueError("Label is outside the probability class range.")
    confidence = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    points = []
    for threshold in confidence_thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence thresholds must lie in [0, 1].")
        selected = confidence >= threshold
        points.append({
            "confidence_threshold": float(threshold),
            "coverage": float(selected.mean()),
            "abstention_rate": float(1 - selected.mean()),
            "selective_accuracy": float((predictions[selected] == y[selected]).mean()) if selected.any() else 0.0,
            "accepted_count": int(selected.sum()),
        })
    return points


def fit_physics_residual(
    features: Sequence[Sequence[float]], physics_factor_of_safety: Sequence[float],
    observed_factor_of_safety: Sequence[float], *, l2: float = 1.0,
) -> dict[str, object]:
    """Fit a ridge residual correction to user-supplied observed FoS values.

    The function deliberately needs observed FoS rather than risk labels. No
    observation data is included in this repository, so this is a research
    primitive and must not be presented as a trained production model.
    """
    x = np.asarray(features, dtype=float)
    physics = np.asarray(physics_factor_of_safety, dtype=float)
    observed = np.asarray(observed_factor_of_safety, dtype=float)
    if x.ndim != 2 or physics.shape != (x.shape[0],) or observed.shape != physics.shape or x.shape[0] < 2:
        raise ValueError("Features, physics FoS, and observed FoS must align with at least two rows.")
    if not np.isfinite(x).all() or not np.isfinite(physics).all() or not np.isfinite(observed).all() or l2 < 0:
        raise ValueError("Residual-training values must be finite and L2 non-negative.")
    design = np.column_stack((np.ones(x.shape[0]), x))
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ (observed - physics))
    return {
        "coefficients": coefficients.tolist(),
        "training_rows": int(x.shape[0]),
        "training_data_requirement": "Observed, independently measured FoS labels are required; synthetic training is not real validation.",
    }


def inverse_velocity_forecast(
    times: Sequence[float], velocities: Sequence[float]
) -> tuple[float, float]:
    """Fit inverse velocity vs time and extrapolate the zero-inverse-velocity time.

    Requires accelerating positive displacement velocity. The linear model is
    an indicative Fukuzono-style extrapolation, not a guaranteed failure time.
    """
    time = np.asarray(times, dtype=float)
    velocity = np.asarray(velocities, dtype=float)
    if time.size < 3 or time.shape != velocity.shape:
        raise ValueError("At least three paired time and velocity samples are required.")
    if not np.isfinite(time).all() or not np.isfinite(velocity).all() or np.any(velocity <= 0):
        raise ValueError("Time and strictly positive velocity samples must be finite.")
    slope, intercept = np.polyfit(time, 1.0 / velocity, deg=1)
    if slope >= 0:
        raise ValueError("Inverse velocity is not decreasing; no accelerating-failure forecast.")
    failure_time = -intercept / slope
    if failure_time <= time.max():
        raise ValueError("Extrapolated failure time is not in the future.")
    return float(failure_time), float(slope)


def cusum_change_points(values: Sequence[float], *, drift: float = 0.5, threshold: float = 5.0) -> list[int]:
    """Return one-sided standardized CUSUM alarm indices for upward shifts."""
    data = np.asarray(values, dtype=float)
    if data.size < 2 or not np.isfinite(data).all():
        raise ValueError("CUSUM needs at least two finite observations.")
    if drift < 0 or threshold <= 0:
        raise ValueError("CUSUM drift must be non-negative and threshold positive.")
    scale = float(data.std(ddof=1))
    if scale <= 0:
        return []
    mean = float(data.mean())
    cumulative = 0.0
    alarms = []
    for index, value in enumerate(data):
        cumulative = max(0.0, cumulative + (float(value) - mean) / scale - drift)
        if cumulative >= threshold:
            alarms.append(index)
            cumulative = 0.0
    return alarms


def score_sensor_health(
    values: Sequence[float], *, expected_min: float, expected_max: float,
    missing_count: int = 0, outlier_count: int = 0,
) -> dict[str, float | bool]:
    """Compute a transparent sensor-quality score and distrust flag."""
    array = np.asarray(values, dtype=float)
    if expected_min >= expected_max or missing_count < 0 or outlier_count < 0:
        raise ValueError("Invalid sensor bounds or quality counts.")
    finite = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Sensor sequence cannot be empty.")
    invalid = int(np.sum((finite < expected_min) | (finite > expected_max)))
    invalid += int(array.size - finite.size) + missing_count + outlier_count
    score = max(0.0, 1.0 - invalid / (array.size + missing_count + outlier_count))
    return {"health_score": score, "do_not_trust": score < 0.8}


def filter_sensor_series(
    values: Sequence[float | None], *, max_interpolated_gap: int = 3,
    mad_threshold: float = 3.5,
) -> dict[str, object]:
    """Flag robust MAD outliers and linearly interpolate short missing runs."""
    if len(values) < 2 or max_interpolated_gap < 0 or mad_threshold <= 0:
        raise ValueError("Need at least two samples, non-negative gap limit, and positive MAD threshold.")
    raw = np.array([np.nan if value is None else float(value) for value in values], dtype=float)
    valid = np.isfinite(raw)
    if not valid.any():
        raise ValueError("At least one finite sensor reading is required.")
    median = float(np.median(raw[valid]))
    mad = float(np.median(np.abs(raw[valid] - median)))
    scale = 1.4826 * mad
    outliers = np.zeros(len(raw), dtype=bool)
    if scale > 0:
        outliers = valid & (np.abs(raw - median) > mad_threshold * scale)
    cleaned = raw.copy()
    cleaned[outliers] = np.nan
    missing = ~np.isfinite(cleaned)
    max_gap_exceeded = False
    run = 0
    for is_missing in missing:
        run = run + 1 if is_missing else 0
        if run > max_interpolated_gap:
            max_gap_exceeded = True
    valid_indexes = np.flatnonzero(np.isfinite(cleaned))
    if valid_indexes.size < 2:
        max_gap_exceeded = True
    interpolated = np.interp(
        np.arange(len(cleaned)),
        valid_indexes,
        cleaned[valid_indexes],
    )
    return {
        "cleaned_values": interpolated.tolist(),
        "outlier_indices": np.flatnonzero(outliers).tolist(),
        "missing_or_outlier_indices": np.flatnonzero(missing).tolist(),
        "max_gap_exceeded": max_gap_exceeded,
        "do_not_trust": max_gap_exceeded or float(missing.mean()) >= 0.2,
        "assumption": "Median absolute deviation flags isolated outliers; short gaps use linear interpolation. This is not a sensor calibration or fault diagnosis.",
    }


def simulate_sensor_series(
    *, sample_count: int = 120, seed: int = 42, baseline_mm: float = 5.0,
    drift_mm_per_sample: float = 0.03, noise_std_mm: float = 0.1,
    acceleration_mm_per_sample2: float = 0.0,
    acceleration_start_index: int = 0,
    dropout_rate: float = 0.03, outlier_rate: float = 0.01,
) -> list[dict[str, float | int | bool | None]]:
    """Create explicitly synthetic displacement samples with drift and faults."""
    args = (
        baseline_mm, drift_mm_per_sample, noise_std_mm,
        acceleration_mm_per_sample2, dropout_rate, outlier_rate,
    )
    if not all(math.isfinite(float(value)) for value in args):
        raise ValueError("Sensor simulator inputs must be finite.")
    if sample_count < 3 or sample_count > 10000 or noise_std_mm < 0:
        raise ValueError("Sample count must be 3..10000 and noise cannot be negative.")
    if not 0 <= acceleration_start_index < sample_count:
        raise ValueError("Acceleration start index must be within the generated series.")
    if not 0 <= dropout_rate < 1 or not 0 <= outlier_rate < 1:
        raise ValueError("Dropout and outlier rates must be in [0, 1).")
    rng = np.random.default_rng(seed)
    result: list[dict[str, float | int | bool | None]] = []
    displacement = baseline_mm
    velocity = drift_mm_per_sample
    for index in range(sample_count):
        if index >= acceleration_start_index:
            velocity += acceleration_mm_per_sample2
        displacement += velocity
        value = displacement + float(rng.normal(0, noise_std_mm))
        dropout = bool(rng.random() < dropout_rate)
        outlier = bool(rng.random() < outlier_rate)
        if outlier:
            value += float(rng.choice([-1, 1])) * max(5 * noise_std_mm, 5.0)
        result.append({
            "sample_index": index,
            "displacement_mm": None if dropout else float(value),
            "dropout": dropout,
            "outlier_injected": outlier,
            "dataset_type": "DEMO/SYNTHETIC",
        })
    return result


def alarm_metrics(
    alarm_indices: Sequence[int], event_indices: Sequence[int], *,
    observation_count: int, samples_per_month: float = 30.0,
) -> dict[str, float | int]:
    """Summarize event recall, false alarms/month, and event lead-time samples.

    An alarm is matched to the next event if it occurs before it. This simple
    matching does not replace rolling-origin evaluation on site-labelled data.
    """
    if observation_count <= 0 or samples_per_month <= 0:
        raise ValueError("Observation count and samples per month must be positive.")
    alarms = sorted(set(int(i) for i in alarm_indices))
    events = sorted(set(int(i) for i in event_indices))
    if any(i < 0 or i >= observation_count for i in alarms + events):
        raise ValueError("Alarm and event indices must lie within the observation window.")
    matched: set[int] = set()
    leads = []
    false_alarms = 0
    for alarm in alarms:
        future = [(event - alarm, event) for event in events if event > alarm and event not in matched]
        if future:
            lead, event = min(future)
            matched.add(event)
            leads.append(float(lead))
        else:
            false_alarms += 1
    return {
        "events": len(events),
        "events_with_prior_alarm": len(matched),
        "event_recall": len(matched) / len(events) if events else 0.0,
        "false_alarms_per_month": false_alarms / observation_count * samples_per_month,
        "mean_lead_samples": float(np.mean(leads)) if leads else 0.0,
    }


def greenfield_bench_screen(
    *, bench_heights_m: Sequence[float], berm_widths_m: Sequence[float],
    slope_angles_deg: Sequence[float], minimum_factor_of_safety: float,
    evaluate_factor_of_safety: Callable[[float, float, float], float],
) -> list[dict[str, float]]:
    """Enumerate candidate bench/berm/angle settings that meet an FoS screen.

    The evaluator and cost assumptions are supplied by the caller; this returns
    feasible geometry only and makes no claim about operational economics.
    """
    if minimum_factor_of_safety <= 0:
        raise ValueError("Minimum FoS must be positive.")
    candidates = []
    for height in bench_heights_m:
        for berm in berm_widths_m:
            for angle in slope_angles_deg:
                if height <= 0 or berm < 0 or not 0 < angle < 90:
                    raise ValueError("Bench height, berm width, or angle is outside valid bounds.")
                fos = float(evaluate_factor_of_safety(height, berm, angle))
                if not math.isfinite(fos):
                    raise ValueError("FoS evaluator returned a non-finite value.")
                if fos >= minimum_factor_of_safety:
                    candidates.append({
                        "bench_height_m": float(height),
                        "berm_width_m": float(berm),
                        "inter_ramp_angle_deg": float(angle),
                        "factor_of_safety": fos,
                    })
    return candidates
