from src.models.question3_absolute_optimizer import optimize_absolute_resources


def test_higher_experience_penalty_selects_more_service_coverage():
    low, _ = optimize_absolute_resources(
        parking_required=200,
        permanent_spaces=100,
        temporary_capacity=100,
        shuttle_required=10,
        staff_required=20,
        risk_penalty=0.5,
    )
    high, diagnostics = optimize_absolute_resources(
        parking_required=200,
        permanent_spaces=100,
        temporary_capacity=100,
        shuttle_required=10,
        staff_required=20,
        risk_penalty=6.0,
    )

    assert high["selected_temporary_spaces"] >= low["selected_temporary_spaces"]
    assert high["selected_shuttle_vehicles"] >= low["selected_shuttle_vehicles"]
    assert high["selected_staff_shifts"] >= low["selected_staff_shifts"]
    assert high["standardized_experience_loss"] <= low["standardized_experience_loss"]
    assert len(diagnostics) == 27


def test_optimizer_keeps_unavoidable_parking_shortage_in_experience_loss():
    chosen, _ = optimize_absolute_resources(
        parking_required=250,
        permanent_spaces=100,
        temporary_capacity=50,
        shuttle_required=1,
        staff_required=1,
        risk_penalty=10,
    )

    assert chosen["parking_unserved_spaces"] == 100
    assert chosen["standardized_experience_loss"] > 0
