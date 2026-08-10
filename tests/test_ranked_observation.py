import pandas as pd

from src.features.ranked_observation import build_censored_observation_panel


def test_build_censored_panel_marks_missing_pairs_as_left_censored():
    observations = pd.DataFrame(
        {
            "date": ["2024-07-01", "2024-07-02"],
            "attraction_name": ["景点甲", "景点乙"],
            "region_code": ["BDH", "BDH"],
            "visitor_index": [8.0, 6.0],
            "daily_rank": [2, 5],
        }
    )
    attractions = pd.DataFrame(
        {
            "attraction_name": ["景点甲", "景点乙"],
            "region_code": ["BDH", "BDH"],
        }
    )

    panel = build_censored_observation_panel(
        observations,
        attractions,
        start_date="2024-07-01",
        end_date="2024-07-02",
    )

    assert len(panel) == 4
    observed = panel.loc[(panel["date"] == pd.Timestamp("2024-07-01")) & (panel["attraction_name"] == "景点甲")].iloc[0]
    missing = panel.loc[(panel["date"] == pd.Timestamp("2024-07-01")) & (panel["attraction_name"] == "景点乙")].iloc[0]
    assert observed["is_observed"]
    assert observed["observation_type"] == "density"
    assert observed["visitor_index"] == 8.0
    assert not missing["is_observed"]
    assert missing["observation_type"] == "left_censored"
    assert pd.isna(missing["visitor_index"])
    assert missing["threshold_group"] == "BDH_2024-07"


def test_build_censored_panel_does_not_invent_unsampled_calendar_dates():
    observations = pd.DataFrame(
        {
            "date": ["2024-07-01", "2024-07-03"],
            "attraction_name": ["景点甲", "景点乙"],
            "region_code": ["BDH", "BDH"],
            "visitor_index": [8.0, 6.0],
        }
    )
    attractions = pd.DataFrame(
        {
            "attraction_name": ["景点甲", "景点乙"],
            "region_code": ["BDH", "BDH"],
        }
    )

    panel = build_censored_observation_panel(
        observations,
        attractions,
        start_date="2024-07-01",
        end_date="2024-07-03",
    )

    assert len(panel) == 4
    assert set(panel["date"]) == {pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-03")}
