from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.models.scale_calibration import calibrate_daily_visitor_scale
p=pd.read_csv(ROOT/'data/model_input/daily_region_censored_likelihood_pressure_2023_2025.csv',encoding='utf-8-sig')
a=pd.read_csv(ROOT/'data/model_input/calibration_anchor_scope_ledger.csv',encoding='utf-8-sig')
e,f=calibrate_daily_visitor_scale(p,a)
e.to_csv(ROOT/'data/model_input/daily_region_visitor_scale_censored_likelihood_2023_2025.csv',index=False,encoding='utf-8-sig')
f.to_csv(ROOT/'data/model_input/visitor_scale_censored_likelihood_anchor_fit.csv',index=False,encoding='utf-8-sig')
print(len(e),len(f))
