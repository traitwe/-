"""Small integer scheduler for three overlapping operational shifts."""

from __future__ import annotations


def optimize_three_shifts(required_people_by_block: list[int]) -> dict[str, int]:
    """Minimise staff-shifts while covering early/mid/late demand.

    The early shift covers blocks 1–2, the middle shift blocks 1–3 and the
    late shift blocks 2–3.  These are staff-shift assignments, not a record of
    currently employed people.
    """
    if len(required_people_by_block) != 3:
        raise ValueError("required_people_by_block must contain early, middle and late demand")
    if any(int(value) != value or value < 0 for value in required_people_by_block):
        raise ValueError("required_people_by_block must be non-negative integers")
    early_required, middle_required, late_required = required_people_by_block
    upper = max(required_people_by_block)
    candidates: list[dict[str, int]] = []
    for middle_shift in range(upper + 1):
        early_shift = max(early_required - middle_shift, 0)
        late_shift = max(late_required - middle_shift, 0)
        total = early_shift + middle_shift + late_shift
        if total < middle_required:
            late_shift += middle_required - total
            total = middle_required
        coverage = [early_shift + middle_shift, total, middle_shift + late_shift]
        candidates.append({
            "early_shift": early_shift,
            "middle_shift": middle_shift,
            "late_shift": late_shift,
            "staff_shifts": total,
            "overstaffing_person_blocks": sum(provided - required for provided, required in zip(coverage, required_people_by_block)),
        })
    return min(candidates, key=lambda row: (row["staff_shifts"], row["overstaffing_person_blocks"], row["middle_shift"], row["early_shift"]))
