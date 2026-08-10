from src.models.question3_relative_optimizer import optimize_relative_tiers


def test_relative_optimizer_upgrades_high_pressure_service_bundle():
    chosen, diagnostics = optimize_relative_tiers(relative_pressure=1.9, risk_penalty=10.0)

    assert chosen["shuttle_tier"] >= 1
    assert chosen["entry_tier"] >= 1
    assert len(diagnostics) == 81
    assert chosen["objective"] == diagnostics["objective"].min()


def test_budgeted_optimizer_produces_resource_specific_tradeoff():
    chosen, diagnostics = optimize_relative_tiers(relative_pressure=1.6, risk_penalty=4.0, budget=3.0)

    assert chosen["relative_cost"] <= 3.0
    assert len({chosen["staff_tier"], chosen["entry_tier"], chosen["parking_guidance_tier"], chosen["shuttle_tier"]}) > 1
    assert diagnostics["relative_cost"].le(3.0).all()


def test_local_anchor_support_reduces_only_the_matching_resource_risk():
    _, baseline = optimize_relative_tiers(relative_pressure=1.5, risk_penalty=4.0, budget=1.0)
    _, supported = optimize_relative_tiers(relative_pressure=1.5, risk_penalty=4.0, budget=1.0, anchor_support={"parking_guidance": 0.5})

    assert supported.loc[0, "parking_guidance_risk"] < baseline.loc[0, "parking_guidance_risk"]
    assert supported.loc[0, "shuttle_risk"] == baseline.loc[0, "shuttle_risk"]
