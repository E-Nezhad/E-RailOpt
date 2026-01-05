import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import math
from backend.resistance_calculations import (
    get_assumption_vals,
    resistance_force_single,
    totalResistanceForces,
    tractive_power
)

# === HARDCODED INPUTS ===

input_excel_path = "../inputDataNeTrainSimEx.xlsx"
assumptions = {
    "P": 3000000,
    "m_loc": 150000,
    "m_car": 100000,
    "axles_on_car": 4,
    "axles_on_loco": 4,
    "num_locos": 2,
    "num_cars": 10,
    "K_cl": 0.0055,
    "fa_cl": 10.0,
    "train_efficiency": 0.85,
    "regen_efficiency": 0.85,
    "regen_buffer": 0.0,
    "max_decel": 0.5,
    "brake_buffer": 50.0,
    "max_tractive_force": 3000000,
    "max_accel": 0.4
}

def simulate_from_excel():
    # === Load Excel data ===
    velocity_data = pd.read_excel(input_excel_path, sheet_name='Speeds')
    grade_data = pd.read_excel(input_excel_path, sheet_name='Vertical Alignment')
    curvature_data = pd.read_excel(input_excel_path, sheet_name='Horizontal Alignment')

    # === Convert chainage to meters ===
    velocity_data['Chainage'] = velocity_data['Chainage'] * 1000
    grade_data['Chainage'] = grade_data['Chainage'] * 1000
    curvature_data['Chainage'] = curvature_data['Chainage'] * 1000

    # === Interpolation functions ===
    velocity_interp = interp1d(
        velocity_data['Chainage'],
        velocity_data['Speed (km/h)'] * 1000 / 3600,
        kind='previous', fill_value='extrapolate')

    grade_interp = interp1d(
        grade_data['Chainage'],
        grade_data['Gradient'],
        kind='previous', fill_value='extrapolate')

    curvature_interp = interp1d(
        curvature_data['Chainage'],
        curvature_data['Curve Degree'],
        kind='previous', fill_value='extrapolate')
    print("hello")
    # === Set up simulation ===
    dt = 1.0
    total_distance = velocity_data['Chainage'].iloc[-1]
    velocity_seconds = [0.0]
    distance_seconds = [0.0]
    time_seconds = [0]
   
    (P, _, _, _, _, num_locos, _, _, _, _, _, regen_buffer,
     max_decel, brake_buffer, max_tractive_force, max_accel,
     total_mass) = get_assumption_vals(assumptions)
   
    while distance_seconds[-1] < total_distance:
        print("hi")
        v = velocity_seconds[-1]
        s = distance_seconds[-1]
        t = time_seconds[-1]

        G = grade_interp(s)
        C = curvature_interp(s)
        v_limit = velocity_interp(s)
        dist_left = total_distance - s

        F_resist = resistance_force_single(v, G, C, assumptions)

        if v > 0:
            stopping_distance = (v ** 2) / (2 * max_decel)
        else:
            stopping_distance = 0

        if dist_left <= 0.1:
            a = -v / dt
            F_trac = 0
        elif stopping_distance + brake_buffer >= dist_left:
            a = -min(v ** 2 / (2 * max(dist_left, 1)), max_decel)
            F_trac = 0
        else:
            F_trac = min(P * num_locos / v, max_tractive_force) if v > 0 else max_tractive_force
            F_net = F_trac - F_resist
            a = max(min(F_net / total_mass, max_accel), -max_decel)

        v_new = max(0, min(v + a * dt, v_limit))
        s_new = s + v_new * dt

        velocity_seconds.append(v_new)
        distance_seconds.append(s_new)
        time_seconds.append(t + dt)

        if s_new >= total_distance and v_new <= 0.1:
            velocity_seconds[-1] = 0
            distance_seconds[-1] = total_distance
            break
    print("hello1")
    distance_seconds = np.array(distance_seconds)
    velocity_seconds = np.array(velocity_seconds)
    time_seconds = np.array(time_seconds)
    grade_seconds = grade_interp(distance_seconds)
    curvature_seconds = curvature_interp(distance_seconds)
    acceleration_seconds = np.diff(velocity_seconds, prepend=velocity_seconds[0])

    resistance_seconds = totalResistanceForces(velocity_seconds, grade_seconds, curvature_seconds, assumptions)
    _, _, _, _, _, total_energy_consumption, _ = tractive_power(
        acceleration_seconds,
        resistance_seconds,
        velocity_seconds,
        assumptions
    )

    total_time = time_seconds[-1]
    print("hello3")

    print(f"⚡ Total net energy consumed: {total_energy_consumption:.3f} kWh")
    print(f"⏱️ Total travel time: {total_time:.1f} seconds")


# Run the simulation
if __name__ == "__main__":
    simulate_from_excel()
