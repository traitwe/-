import pandas as pd

from src.models.hierarchical_censored_pressure import (
    prepare_censored_likelihood_data,
    negative_log_posterior,
    fit_censored_pressure_core,
    fit_hierarchical_censored_pressure,
    generate_continuous_hierarchical_pressure,
    project_continuous_pressure,
)


def test_prepare_censored_likelihood_data_keeps_density_and_left_censor_rows_separate():
    panel = pd.DataFrame({
        "date": ["2024-07-01", "2024-07-01", "2024-07-01"],
        "region_code": ["BDH", "BDH", "BDH"],
        "attraction_name": ["甲", "乙", "丙"],
        "is_observed": [True, True, False],
        "visitor_index": [3.0, 7.0, None],
    })
    covariates = pd.DataFrame({"date": ["2024-07-01"], "is_holiday": [0.0]})

    data = prepare_censored_likelihood_data(panel, covariates)

    assert len(data["observed_log_index"]) == 2
    assert data["censored_count"].sum() == 1
    assert data["group_count"] == 1


def test_negative_log_posterior_uses_left_censored_count():
    panel = pd.DataFrame({"date":["2024-07-01","2024-07-01","2024-07-01"],"region_code":["BDH"]*3,"attraction_name":["甲","乙","丙"],"is_observed":[True,True,False],"visitor_index":[3.,7.,None]})
    data = prepare_censored_likelihood_data(panel, pd.DataFrame({"date":["2024-07-01"],"is_holiday":[0.]}))
    params = {"group_eta": [1.0], "attraction_alpha": [0.0, 0.0], "log_sigma": 0.0}
    with_censor = negative_log_posterior(params, data)
    data["groups"] = data["groups"].copy()
    data["groups"]["censored_count"] = 0
    without_censor = negative_log_posterior(params, data)
    assert with_censor > without_censor


def test_fit_censored_pressure_core_returns_finite_sampled_pressure():
    panel = pd.DataFrame({"date":["2024-07-01","2024-07-01","2024-07-02","2024-07-02"],"region_code":["BDH"]*4,"attraction_name":["甲","乙","甲","乙"],"is_observed":[True,False,True,False],"visitor_index":[4.,None,6.,None]})
    data = prepare_censored_likelihood_data(panel, pd.DataFrame({"date":["2024-07-01","2024-07-02"],"is_holiday":[0.,0.]}))
    result = fit_censored_pressure_core(data)
    assert result["converged"]
    assert len(result["group_eta"]) == 2


def test_fit_censored_pressure_core_returns_labelled_sampled_pressure_table():
    panel = pd.DataFrame({"date":["2024-07-01","2024-07-01"],"region_code":["BDH"]*2,"attraction_name":["甲","乙"],"is_observed":[True,False],"visitor_index":[4.,None]})
    data = prepare_censored_likelihood_data(panel, pd.DataFrame({"date":["2024-07-01"],"is_holiday":[0.]}))
    result = fit_censored_pressure_core(data)
    assert result["sampled_pressure"]["estimate_label"].eq("censored_likelihood_pressure_index").all()


def test_hierarchical_fit_returns_joint_covariate_and_dynamic_diagnostics():
    panel = pd.DataFrame({
        "date": ["2024-07-01", "2024-07-01", "2024-07-02", "2024-07-02"],
        "region_code": ["BDH"] * 4,
        "attraction_name": ["A", "B", "A", "B"],
        "is_observed": [True, False, True, False],
        "visitor_index": [4.0, None, 7.0, None],
    })
    covariates = pd.DataFrame({
        "date": ["2024-07-01", "2024-07-02"],
        "is_holiday": [0.0, 1.0],
        "temperature": [20.0, 25.0],
    })
    prepared = prepare_censored_likelihood_data(panel, covariates)

    result = fit_hierarchical_censored_pressure(prepared)

    assert result["converged"]
    assert set(result["parameter_diagnostics"]["parameter_group"]) >= {"covariate", "dynamic", "dispersion"}
    assert result["sampled_pressure"]["estimate_label"].eq("censored_likelihood_pressure_index").all()


def test_continuous_hierarchical_projection_uses_joint_fit_coefficients():
    panel = pd.DataFrame({
        "date": ["2024-07-01", "2024-07-01", "2024-07-02", "2024-07-02"],
        "region_code": ["BDH"] * 4,
        "attraction_name": ["A", "B", "A", "B"],
        "is_observed": [True, False, True, False],
        "visitor_index": [4.0, None, 7.0, None],
    })
    covariates = pd.DataFrame({"date": ["2024-07-01", "2024-07-02", "2024-07-03"], "is_holiday": [0.0, 1.0, 0.0]})
    fit = fit_hierarchical_censored_pressure(prepare_censored_likelihood_data(panel, covariates.iloc[:2]))

    output = generate_continuous_hierarchical_pressure(fit, covariates, ["BDH"])

    assert len(output) == 3
    assert output["estimate_label"].eq("censored_likelihood_pressure_index").all()
    assert output["projection_source"].eq("joint_covariate_region_mean").all()




def test_project_continuous_pressure_returns_all_requested_region_days():
    sampled = pd.DataFrame({"date":["2024-07-01","2024-07-02"],"region_code":["BDH","BDH"],"pressure_index":[2.,4.]})
    covariates = pd.DataFrame({"date":["2024-07-01","2024-07-02","2024-07-03"],"is_holiday":[0.,0.,0.]})
    output = project_continuous_pressure(sampled, covariates, ["BDH"])
    assert len(output) == 3
    assert output["estimate_label"].eq("censored_likelihood_pressure_index").all()
