"""Single, intentionally simple paper-production pipeline definition."""

PAPER_MAINLINE_SCRIPTS = (
    "build_search_theme_features.py",
    "build_pressure_baseline.py",
    "build_censored_observation_diagnostics.py",
    "build_hierarchical_censored_pressure.py",
    "build_censored_likelihood_visitor_scale.py",
    "build_q1_timeseries_analysis.py",
    "build_q1_paper_figures.py",
    "build_q2_forecasts.py",
    "build_q3_absolute_resource_plan.py",
    "build_q4_scenario_analysis.py",
    "build_delivery_diagnostics.py",
)
