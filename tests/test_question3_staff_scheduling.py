from src.models.question3_staff_scheduling import optimize_three_shifts


def test_three_shift_schedule_covers_all_blocks_with_minimum_staff_shifts():
    schedule = optimize_three_shifts([3, 5, 2])

    assert schedule["early_shift"] + schedule["middle_shift"] >= 3
    assert schedule["early_shift"] + schedule["middle_shift"] + schedule["late_shift"] >= 5
    assert schedule["middle_shift"] + schedule["late_shift"] >= 2
    assert schedule["staff_shifts"] == 5


def test_three_shift_schedule_prefers_lower_overstaffing_when_staff_shifts_tie():
    schedule = optimize_three_shifts([2, 2, 2])

    assert schedule["staff_shifts"] == 2
    assert schedule["overstaffing_person_blocks"] == 0


def test_three_shift_schedule_handles_peak_scale_demands_without_cubic_enumeration():
    schedule = optimize_three_shifts([10_000, 20_000, 5_000])

    assert schedule["early_shift"] + schedule["middle_shift"] >= 10_000
    assert schedule["early_shift"] + schedule["middle_shift"] + schedule["late_shift"] >= 20_000
    assert schedule["middle_shift"] + schedule["late_shift"] >= 5_000
    assert schedule["staff_shifts"] == 20_000
