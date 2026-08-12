# 问题三：停车—接驳—排班联合资源配置
import math
from itertools import product

def resource_plan(visitors, car_share, dwell_hours, turnover, vehicle_capacity,
                  trips_per_vehicle, staff_rate, permanent_spaces, temporary_spaces):
    parking_need = math.ceil(visitors * car_share * dwell_hours / max(turnover, 1e-8))
    shuttle_need = math.ceil(visitors / max(vehicle_capacity * trips_per_vehicle, 1))
    staff_need = math.ceil(visitors / max(staff_rate, 1))
    candidates = []
    for temporary, shuttles, staff in product(range(temporary_spaces + 1),
                                              range(shuttle_need + 4),
                                              range(staff_need + 4)):
        shortage = max(parking_need - permanent_spaces - temporary, 0) + \
                   max(shuttle_need - shuttles, 0) + max(staff_need - staff, 0)
        cost = temporary + 2 * shuttles + 0.6 * staff
        candidates.append((cost + 4 * shortage, temporary, shuttles, staff, shortage))
    _, temporary, shuttles, staff, shortage = min(candidates)
    return {"temporary_spaces": temporary, "shuttle_vehicles": shuttles,
            "staff_shifts": staff, "service_shortfall": shortage}
