"""Build the auditable descriptive-analysis outputs required by Question 1."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.q1_timeseries import build_q1_analysis_outputs


annual = pd.read_csv(ROOT / "data/runtime/clean/annual_city_tourism_1999_2024.csv", encoding="utf-8-sig")
annual = annual.loc[annual["指标"].eq("国内旅游人次")].copy()
pressure = pd.read_csv(ROOT / "data/runtime/model_input/daily_region_censored_likelihood_pressure_2023_2025.csv", encoding="utf-8-sig")
regional_visitor_estimates = pd.read_csv(ROOT / "data/runtime/model_input/daily_region_visitor_scale_censored_likelihood_2023_2025.csv", encoding="utf-8-sig")
parameters = pd.read_csv(ROOT / "data/runtime/model_input/censored_likelihood_parameter_diagnostics.csv", encoding="utf-8-sig")
service = pd.read_csv(ROOT / "data/runtime/clean/hourly_service_time_evidence.csv", encoding="utf-8-sig")

report = build_q1_analysis_outputs(
    annual=annual,
    pressure=pressure,
    parameters=parameters,
    service_evidence=service,
    output_directory=ROOT / "outputs/runtime/question1_analysis",
    annual_value_column="数值",
    regional_visitor_estimates=regional_visitor_estimates,
)
print(report)
