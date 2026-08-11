import pytest

from src.models.value_of_time import estimate_traveler_time_loss


def test_estimate_traveler_time_loss_converts_unserved_services_to_currency():
    result = estimate_traveler_time_loss(
        daily_visitors=100,
        parking_unserved_spaces=10,
        average_party_size=2,
        parking_delay_hours=0.5,
        transfer_share=0.5,
        shuttle_shortfall_fraction=0.1,
        shuttle_delay_hours=0.25,
        staff_shortfall_fraction=0.2,
        staff_delay_hours=0.1,
        vot_cny_per_hour=20,
    )

    assert result == {
        "parking_time_loss_hours": 10.0,
        "shuttle_time_loss_hours": 1.25,
        "staff_time_loss_hours": 2.0,
        "traveler_time_loss_hours": 13.25,
        "traveler_time_loss_cny": 265.0,
    }


def test_estimate_traveler_time_loss_rejects_invalid_values():
    with pytest.raises(ValueError, match="daily_visitors"):
        estimate_traveler_time_loss(
            daily_visitors=-1,
            parking_unserved_spaces=0,
            average_party_size=2,
            parking_delay_hours=0.5,
            transfer_share=0.5,
            shuttle_shortfall_fraction=0.1,
            shuttle_delay_hours=0.25,
            staff_shortfall_fraction=0.2,
            staff_delay_hours=0.1,
            vot_cny_per_hour=20,
        )
