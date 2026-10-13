"""Formula-level regression tests for the offline research methods."""
from __future__ import annotations

import pytest
import numpy as np

from backend.research import (
    Slice,
    alarm_metrics,
    bishop_simplified,
    cost_sensitive_threshold,
    cusum_change_points,
    form_hlrf,
    green_ampt_infiltration_mm,
    greenfield_bench_screen,
    hoek_brown_equivalent_mc,
    inverse_velocity_forecast,
    janbu_simplified,
    kinematic_planar_feasible,
    monte_carlo_probability_of_failure,
    MonotonicLogisticModel,
    conformal_prediction_sets,
    abstention_curve,
    fit_physics_residual,
    filter_sensor_series,
    sample_correlated_random_field,
    score_sensor_health,
    simulate_sensor_series,
)
from backend.physics import calculate_factor_of_safety
from scripts.evaluate_synthetic_alarms import evaluate as evaluate_synthetic_alarms


def example_slices(*, cohesion: float = 20, friction: float = 30, pore_pressure: float = 5):
    return [
        Slice(100, 15, 2, cohesion, friction, pore_pressure),
        Slice(120, 25, 2, cohesion, friction, pore_pressure),
        Slice(140, 35, 2, cohesion, friction, pore_pressure),
    ]


def test_bishop_and_janbu_are_positive_and_respond_to_strength_and_pore_pressure():
    bishop_base = bishop_simplified(example_slices())
    janbu_base = janbu_simplified(example_slices())
    assert bishop_base > 0 and janbu_base > 0
    assert bishop_simplified(example_slices(cohesion=35)) > bishop_base
    assert janbu_simplified(example_slices(cohesion=35)) > janbu_base
    assert bishop_simplified(example_slices(pore_pressure=20)) < bishop_base
    assert janbu_simplified(example_slices(pore_pressure=20)) < janbu_base


def test_limit_equilibrium_rejects_invalid_inputs_and_nonconvergence():
    with pytest.raises(ValueError, match="At least one slice"):
        bishop_simplified([])
    with pytest.raises(ValueError, match="positive driving"):
        janbu_simplified([Slice(100, -10, 2, 10, 25)])
    with pytest.raises(RuntimeError, match="did not converge"):
        bishop_simplified(example_slices(), max_iterations=1, tolerance=0)


def test_existing_infinite_slope_baseline_obeys_requested_monotonic_directions():
    args = dict(
        slope_height_m=60, cohesion_kpa=30, friction_angle_deg=30,
        unit_weight_kn_m3=24, pore_pressure_kpa=40,
    )
    base = calculate_factor_of_safety(slope_angle_deg=40, **args)
    assert calculate_factor_of_safety(slope_angle_deg=45, **args) < base
    assert calculate_factor_of_safety(slope_angle_deg=40, **{**args, "pore_pressure_kpa": 60}) < base
    assert calculate_factor_of_safety(slope_angle_deg=40, **{**args, "cohesion_kpa": 50}) > base
    assert calculate_factor_of_safety(slope_angle_deg=40, **{**args, "friction_angle_deg": 35}) > base


def test_hoek_brown_fit_is_stable_and_parameterized_by_confinement():
    cohesion_low, friction_low = hoek_brown_equivalent_mc(
        intact_ucs_mpa=100, mi=10, gsi=65, disturbance_factor=0.2,
        sigma3_max_mpa=2,
    )
    cohesion_high, friction_high = hoek_brown_equivalent_mc(
        intact_ucs_mpa=100, mi=10, gsi=65, disturbance_factor=0.2,
        sigma3_max_mpa=5,
    )
    assert cohesion_low >= 0 and cohesion_high >= 0
    assert 0 < friction_low < 90 and 0 < friction_high < 90
    assert (cohesion_low, friction_low) != pytest.approx((cohesion_high, friction_high))


def test_planar_kinematic_screen_requires_daylighting_friction_and_direction():
    assert kinematic_planar_feasible(
        slope_dip_deg=55, slope_dip_direction_deg=90,
        joint_dip_deg=35, joint_dip_direction_deg=100, friction_angle_deg=25,
    )
    assert not kinematic_planar_feasible(
        slope_dip_deg=55, slope_dip_direction_deg=90,
        joint_dip_deg=35, joint_dip_direction_deg=180, friction_angle_deg=25,
    )
    assert not kinematic_planar_feasible(
        slope_dip_deg=55, slope_dip_direction_deg=90,
        joint_dip_deg=20, joint_dip_direction_deg=90, friction_angle_deg=25,
    )


def test_green_ampt_infiltration_is_bounded_by_rainfall_and_assumption_units():
    assert green_ampt_infiltration_mm(
        rainfall_mm=40, duration_hours=2, saturated_conductivity_mm_h=5,
        suction_head_mm=100, moisture_deficit=0.2,
    ) <= 40
    infiltration = green_ampt_infiltration_mm(
        rainfall_mm=1, duration_hours=0.1, saturated_conductivity_mm_h=0.01,
        suction_head_mm=100, moisture_deficit=0.2,
    )
    assert 0 <= infiltration <= 1


def test_monte_carlo_is_reproducible_and_form_linear_limit_state():
    probability, samples = monte_carlo_probability_of_failure(
        means=[1.0], standard_deviations=[0.2],
        factor_of_safety=lambda row: row[0], sample_count=20_000, seed=7,
    )
    assert probability == pytest.approx(0.5, abs=0.02)
    assert len(samples) == 20_000
    beta, design_point = form_hlrf(
        means=[2.0], standard_deviations=[0.5],
        limit_state=lambda values: values[0] - 1.0,
    )
    assert beta == pytest.approx(2.0, abs=1e-5)
    assert design_point == pytest.approx([1.0], abs=1e-5)


def test_spatial_random_field_preserves_correlation_and_is_seeded():
    field_a = sample_correlated_random_field(
        positions_m=[0, 1, 2, 20], mean=30, standard_deviation=5,
        correlation_length_m=3, sample_count=5000, seed=9,
    )
    field_b = sample_correlated_random_field(
        positions_m=[0, 1, 2, 20], mean=30, standard_deviation=5,
        correlation_length_m=3, sample_count=5000, seed=9,
    )
    assert (field_a == field_b).all()
    empirical = np.corrcoef(field_a, rowvar=False)
    assert empirical[0, 1] > empirical[0, 3]


def test_cost_threshold_and_inverse_velocity_forecast():
    threshold, cost = cost_sensitive_threshold(
        [0.1, 0.4, 0.7, 0.9], [0, 1, 1, 1],
        false_negative_cost=10, false_positive_cost=1, thresholds=[0.2, 0.5, 0.8],
    )
    assert threshold == 0.2
    assert cost == pytest.approx(0.0)
    forecast, slope = inverse_velocity_forecast(
        [0, 1, 2, 3], [1.0, 1.25, 1.67, 2.5]
    )
    assert slope < 0
    assert forecast > 3


def test_cusum_sensor_health_and_alarm_metrics_expose_limits():
    assert cusum_change_points([0, 0, 0, 0, 2, 4, 6], drift=0.1, threshold=1)
    health = score_sensor_health([1, 2, 100, float("nan")], expected_min=0, expected_max=10)
    assert health["do_not_trust"] is True
    metrics = alarm_metrics([2, 8], [5], observation_count=30, samples_per_month=30)
    assert metrics["events_with_prior_alarm"] == 1
    assert metrics["false_alarms_per_month"] == pytest.approx(1)
    assert metrics["mean_lead_samples"] == 3


def test_sensor_simulator_is_seeded_and_labels_injected_faults():
    series_a = simulate_sensor_series(sample_count=30, seed=12, dropout_rate=0.2, outlier_rate=0.2)
    series_b = simulate_sensor_series(sample_count=30, seed=12, dropout_rate=0.2, outlier_rate=0.2)
    assert series_a == series_b
    assert all(item["dataset_type"] == "DEMO/SYNTHETIC" for item in series_a)
    assert any(item["dropout"] for item in series_a)
    assert any(item["outlier_injected"] for item in series_a)


def test_sensor_quality_filter_marks_outliers_and_long_gaps():
    result = filter_sensor_series([1.0, 2.0, 3.0, 4.0, 100.0, None, None, None], max_interpolated_gap=2)
    assert result["outlier_indices"] == [4]
    assert result["max_gap_exceeded"] is True
    assert result["do_not_trust"] is True
    assert len(result["cleaned_values"]) == 8


def test_bench_design_screen_only_returns_feasible_options():
    candidates = greenfield_bench_screen(
        bench_heights_m=[10, 20], berm_widths_m=[5], slope_angles_deg=[30, 40],
        minimum_factor_of_safety=1.3,
        evaluate_factor_of_safety=lambda h, b, angle: 2 - angle / 100 + b / 100,
    )
    assert len(candidates) == 4
    assert all(item["factor_of_safety"] >= 1.3 for item in candidates)


def test_monotonic_logistic_respects_constraints():
    features = [[float(angle), float(pressure)] for angle, pressure in [
        (25, 10), (30, 20), (35, 30), (40, 40), (45, 50),
        (50, 60), (55, 70), (60, 80), (65, 90), (70, 100),
    ]]
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    model = MonotonicLogisticModel([1, 1]).fit(features, labels)
    probabilities = model.predict_proba(features)
    assert model.coef_ is not None and (model.coef_ >= 0).all()
    assert (probabilities[1:, 1] >= probabilities[:-1, 1]).all()


def test_split_conformal_and_abstention_report_explicit_coverage_shape():
    probabilities = [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]
    result = conformal_prediction_sets(
        probabilities, [0, 1, 0, 1], [[0.7, 0.3], [0.2, 0.8]], alpha=0.25
    )
    assert result["quantile"] == pytest.approx(0.3)
    assert result["prediction_sets"] == [[0], [1]]
    curve = abstention_curve(probabilities, [0, 1, 0, 1], confidence_thresholds=[0.5, 0.95])
    assert curve[0]["coverage"] == 1
    assert curve[1]["coverage"] == 0
    assert curve[1]["abstention_rate"] == 1


def test_physics_residual_requires_observed_values_and_returns_coefficients():
    fit = fit_physics_residual(
        [[0], [1], [2], [3]],
        [1.0, 1.2, 1.4, 1.6],
        [1.1, 1.4, 1.7, 2.0],
        l2=0,
    )
    assert fit["training_rows"] == 4
    assert len(fit["coefficients"]) == 2
    assert "independently measured" in fit["training_data_requirement"]


def test_walk_forward_alarm_evaluation_is_explicitly_synthetic():
    summary, table = evaluate_synthetic_alarms()
    assert summary["dataset_type"] == "DEMO/SYNTHETIC"
    assert summary["walk_forward_origins"]
    assert not table.empty
    assert table["data_type_synthetic"].eq(1).all()
    assert "Not evidence" in summary["interpretation"]
