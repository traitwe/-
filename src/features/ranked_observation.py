"""Prepare ranked-map attraction observations for censored likelihood modelling."""

import pandas as pd


def build_censored_observation_panel(
    observations: pd.DataFrame,
    attractions: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Create all attraction-date pairs without treating absent rankings as zero flow."""
    required_observation = {"date", "attraction_name", "region_code", "visitor_index"}
    required_attraction = {"attraction_name", "region_code"}
    if missing := required_observation.difference(observations.columns):
        raise ValueError(f"observations missing columns: {sorted(missing)}")
    if missing := required_attraction.difference(attractions.columns):
        raise ValueError(f"attractions missing columns: {sorted(missing)}")

    sampled_dates = pd.to_datetime(observations["date"], errors="coerce")
    date_index = pd.DataFrame(
        {
            "date": pd.Series(sampled_dates)
            .loc[lambda values: values.between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
            .dropna()
            .drop_duplicates()
            .sort_values()
        }
    )
    attraction_frame = attractions.loc[:, ["attraction_name", "region_code"]].drop_duplicates()
    date_index["_join_key"] = 1
    attraction_frame = attraction_frame.copy()
    attraction_frame["_join_key"] = 1
    panel = date_index.merge(attraction_frame, on="_join_key", how="inner").drop(columns="_join_key")

    observed = observations.copy()
    observed["date"] = pd.to_datetime(observed["date"])
    keep_columns = ["date", "attraction_name", "region_code", "visitor_index"]
    if "daily_rank" in observed.columns:
        keep_columns.append("daily_rank")
    observed = observed.loc[:, keep_columns].drop_duplicates(
        subset=["date", "attraction_name", "region_code"], keep="last"
    )
    panel = panel.merge(
        observed,
        on=["date", "attraction_name", "region_code"],
        how="left",
    )
    panel["is_observed"] = panel["visitor_index"].notna()
    panel["observation_type"] = panel["is_observed"].map(
        {True: "density", False: "left_censored"}
    )
    panel["threshold_group"] = (
        panel["region_code"].astype(str) + "_" + panel["date"].dt.strftime("%Y-%m")
    )
    return panel.sort_values(["date", "region_code", "attraction_name"]).reset_index(drop=True)
