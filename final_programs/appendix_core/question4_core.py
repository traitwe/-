# 问题四：客流冲击、资源重优化与稳健性比较
import pandas as pd

def apply_event_pulse(frame, visitor_column, event_date, intensity):
    result = frame.copy()
    offset = (pd.to_datetime(result["date"]) - pd.Timestamp(event_date)).dt.days
    pulse = offset.map({-2: .25, -1: .50, 0: 1., 1: .50, 2: .25}).fillna(0.)
    result["scenario_visitors"] = result[visitor_column] * (1 + intensity * pulse)
    return result

def apply_continuous_rain(frame, visitor_column, start_date, days, attenuation):
    result = frame.copy()
    offset = (pd.to_datetime(result["date"]) - pd.Timestamp(start_date)).dt.days
    factor = (1 - attenuation * (offset + 1)).clip(lower=0)
    result["scenario_visitors"] = result[visitor_column] * factor.where(offset.between(0, days - 1), 1.)
    return result

def compare_plan(baseline, shocked, scenario, resource_plan):
    plans = [resource_plan(v, **scenario) for v in shocked["scenario_visitors"]]
    return baseline.assign(reoptimized_plan=plans)
