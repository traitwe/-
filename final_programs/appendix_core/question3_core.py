# question3_core.py: 以下函数从完整运行程序的真实实现原样摘取。

# 完整输入、输出与辅助函数见支撑材料中的 question*_model.py。

def optimize_absolute_resources(
    parking_required: int,
    permanent_spaces: int,
    temporary_capacity: int,
    shuttle_required: int,
    staff_required: int,
    risk_penalty: float = 4.0,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Enumerate six coverage levels for temporary parking, shuttles and staff.

    Costs are relative scenario units: temporary-space activation, vehicle
    deployment, and staff-shift deployment each equal one at full coverage.
    Experience loss is the weighted fraction of unmet parking, shuttle and
    staffing need.  It is not a measured satisfaction score.
    """
    values = (parking_required, permanent_spaces, temporary_capacity, shuttle_required, staff_required)
    if any(int(value) != value or value < 0 for value in values) or risk_penalty < 0:
        raise ValueError("resource requirements and capacities must be non-negative integers; risk_penalty non-negative")
    levels = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    records = []
    for parking_level, shuttle_level, staff_level in product(levels, repeat=3):
        temporary = math.ceil(temporary_capacity * parking_level)
        shuttles = math.ceil(shuttle_required * shuttle_level)
        staff = math.ceil(staff_required * staff_level)
        parking_unserved = max(parking_required - permanent_spaces - temporary, 0)
        shuttle_shortfall = max(shuttle_required - shuttles, 0)
        staff_shortfall = max(staff_required - staff, 0)
        parking_loss = parking_unserved / max(parking_required, 1)
        shuttle_loss = shuttle_shortfall / max(shuttle_required, 1)
        staff_loss = staff_shortfall / max(staff_required, 1)
        experience_loss = 0.9 * parking_loss + 1.2 * shuttle_loss + staff_loss
        relative_cost = (
            0.6 * temporary / max(temporary_capacity, 1)
            + 1.2 * shuttles / max(shuttle_required, 1)
            + staff / max(staff_required, 1)
        )
        records.append({
            "selected_temporary_spaces": temporary,
            "selected_shuttle_vehicles": shuttles,
            "selected_staff_shifts": staff,
            "parking_unserved_spaces": parking_unserved,
            "shuttle_shortfall": shuttle_shortfall,
            "staff_shortfall": staff_shortfall,
            "standardized_experience_loss": experience_loss,
            "relative_operating_cost": relative_cost,
            "objective": relative_cost + risk_penalty * experience_loss,
        })
    diagnostics = pd.DataFrame(records).sort_values(["objective", "relative_operating_cost", "standardized_experience_loss"]).reset_index(drop=True)
    return diagnostics.iloc[0].to_dict(), diagnostics
