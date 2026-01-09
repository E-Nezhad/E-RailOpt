import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from resistanceOnly import total_resistance_over_time


def get_assumption_vals(assumptions):
    m_loc = assumptions.get("m_loc", 150000)
    m_car = assumptions.get("m_car", 150000)
    axles_on_car = assumptions.get("axles_on_car", 4)
    axles_on_loco = assumptions.get("axles_on_loco", 4)
    num_locos = assumptions.get("num_locos", 1)
    num_cars = assumptions.get("num_cars", 19)
    K_cl = assumptions.get("K_cl", 0.0055)
    fa_cl = assumptions.get("fa_cl", 10)
    train_efficiency = assumptions.get("train_efficiency", 0.8)
    regen_efficiency = assumptions.get("regen_efficiency", 0.7)
    regen_buffer = assumptions.get("regen_buffer", 0)

    m_total = m_loc * num_locos + m_car * num_cars

    return (
        m_loc, m_car, axles_on_car, axles_on_loco, num_locos, num_cars,
        K_cl, fa_cl, None, train_efficiency, regen_efficiency, regen_buffer,
        None, None, None, None, m_total
    )


def tractive_power(acceleration_seconds, resistance_seconds, velocity_seconds, assumptions):
    if not (len(acceleration_seconds) == len(resistance_seconds) == len(velocity_seconds)):
        raise ValueError("Input lists must have the same length")

    (_, _, _, _, _, _, _, _, _, train_efficiency, regen_efficiency, regen_buffer,
     _, _, _, _, total_mass) = get_assumption_vals(assumptions)

    power_output = []
    regen_power = []
    energy_consumption = []

    for i in range(len(acceleration_seconds)):
        a = acceleration_seconds[i]
        r = resistance_seconds[i]
        v = velocity_seconds[i]

        # Tractive power in watts
        power = (total_mass * a + r) * v
        power = power / train_efficiency
        power_output.append(power)

        # Regen logic
        if a < -1e-5:
            try:
                regen = 1 / math.exp(abs(regen_efficiency / a))
            except OverflowError:
                regen = 0
        else:
            regen = 0
        regen_power.append(regen)

        # Energy in kWh
        power_kWh = power / (1000 * 3600)
        regen_kWh = regen / (1000 * 3600)

        # Net energy consumption logic
        if a > 0:
            if regen_buffer > 0:
                used = min(regen_buffer, power_kWh)
                net_energy = power_kWh - used
                regen_buffer -= used
            else:
                net_energy = power_kWh
        elif a < 0:
            next_is_accel = i < len(acceleration_seconds) - 1 and acceleration_seconds[i + 1] > 0
            if next_is_accel:
                net_energy = power_kWh - regen_kWh
            else:
                regen_buffer += regen_kWh
                net_energy = power_kWh
        else:
            net_energy = power_kWh

        energy_consumption.append(net_energy)

    return power_output, regen_power, energy_consumption


########

def interpret_excel(filepath='./uploads/inputDataTest.xlsx'):
    # Read sheets
    velocity_data = pd.read_excel(filepath, sheet_name='Speeds')
    grade_data = pd.read_excel(filepath, sheet_name='Vertical Alignment')
    curvature_data = pd.read_excel(filepath, sheet_name='Horizontal Alignment')

    # Convert km to m, km/h to m/s
    velocity_interp = interp1d(
        velocity_data['Chainage'] * 1000,
        velocity_data['Speed (km/h)'] * 1000 / 3600,
        kind='previous', fill_value='extrapolate'
    )

    grade_interp = interp1d(
        grade_data['Chainage'] * 1000,
        grade_data['Gradient'],
        kind='previous', fill_value='extrapolate'
    )

    curvature_interp = interp1d(
        curvature_data['Chainage'] * 1000,
        curvature_data['Curve Degree'],
        kind='previous', fill_value='extrapolate'
    )

    total_distance = velocity_data['Chainage'].iloc[-1] * 1000  # m
    return velocity_interp, grade_interp, curvature_interp, total_distance


def simulate_from_velocity_profile(assumptions, filepath='./uploads/inputDataTest.xlsx'):
    dt = 1.0  # time step in seconds

    velocity_interp, grade_interp, curvature_interp, total_distance = interpret_excel(filepath)

    # Sample distance at 1s intervals using forward Euler
    s = 0
    v = velocity_interp(s)
    distance_list = [s]
    velocity_list = [v]
    time_list = [0]

    while s < total_distance:
        s += v * dt
        v = velocity_interp(s)
        velocity_list.append(v)
        distance_list.append(s)
        time_list.append(time_list[-1] + dt)

    # Remove final step if it overshot
    distance_array = np.array(distance_list[:-1])
    velocity_array = np.array(velocity_list[:-1])
    time_array = np.array(time_list[:-1])

    # Compute derived arrays
    acceleration_array = np.diff(velocity_array, prepend=velocity_array[0])
    grade_array = grade_interp(distance_array)
    curvature_array = curvature_interp(distance_array)
    resistance_array = total_resistance_over_time(
        velocity_array, grade_array, curvature_array, assumptions
    )

    return {
        'time': time_array,
        'distance': distance_array,
        'velocity': velocity_array,
        'acceleration': acceleration_array,
        'grade': grade_array,
        'curvature': curvature_array,
        'resistance': resistance_array
    }


# ✅ Run directly
if __name__ == '__main__':
    assumptions = {
        "m_loc": 150000,
        "m_car": 150000,
        "axles_on_car": 4,
        "axles_on_loco": 4,
        "num_locos": 1,
        "num_cars": 19,
        "K_cl": 0.0055,
        "fa_cl": 10
    }

    results = simulate_from_velocity_profile(assumptions)
# Calculate tractive power and energy
    power_out, regen_vals, energy_vals = tractive_power(
        results['acceleration'],
        results['resistance'],
        results['velocity'],
        assumptions
    )

    print("✅ Simulation complete.\n")
    print("📌 First 5 seconds of simulation with power output:")
    print("Time (s) | Velocity (m/s) | Acceleration (m/s²) | Resistance (N) | Power (W) | Energy (kWh)")

    for i in range(min(5, len(results['time']))):
        print(f"{results['time'][i]:>8.1f} | {results['velocity'][i]:>15.3f} | {results['acceleration'][i]:>19.3f} |"
              f" {results['resistance'][i]:>13.2f} | {power_out[i]:>10.2f} | {energy_vals[i]:>12.6f}")

    # Save full data to Excel
    df = pd.DataFrame({
        'Time (s)': results['time'],
        'Distance (m)': results['distance'],
        'Velocity (m/s)': results['velocity'],
        'Acceleration (m/s²)': results['acceleration'],
        'Grade': results['grade'],
        'Curvature (deg)': results['curvature'],
        'Resistance (N)': results['resistance'],
        'Power (W)': power_out,
        'Energy (kWh)': energy_vals
    })
    df.to_excel('./uploads/full_simulation_output.xlsx', index=False)
    print("\n📤 Saved to './uploads/full_simulation_output.xlsx'")

    total_energy = sum(energy_vals)
    print(f"\n🔋 Total Energy Consumed: {total_energy:.6f} kWh")
