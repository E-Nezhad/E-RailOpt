import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad

def get_assumption_vals(assumptions):
    P = assumptions["P"]
    m_loc = assumptions["m_loc"]
    m_car = assumptions["m_car"]
    axles_on_car = assumptions["axles_on_car"]
    axles_on_loco = assumptions["axles_on_loco"]
    num_locos = assumptions["num_locos"]
    num_cars = assumptions["num_cars"]
    K_cl = assumptions["K_cl"]
    fa_cl = assumptions["fa_cl"]
    train_efficiency = assumptions["train_efficiency"]
    regen_efficiency = assumptions["regen_efficiency"]
    regen_buffer = assumptions["regen_buffer"]
    max_decel = assumptions["max_decel"]
    brake_buffer = assumptions["brake_buffer"]
    max_tractive_force = assumptions["max_tractive_force"]
    max_accel = assumptions["max_accel"]

    total_mass = num_locos * m_loc + num_cars * m_car

    return (
        P, m_loc, m_car, axles_on_car, axles_on_loco,
        num_locos, num_cars, K_cl, fa_cl,
        train_efficiency, regen_efficiency, regen_buffer,
        max_decel, brake_buffer, max_tractive_force, max_accel,
        total_mass
    )


def resistance_force_single(v, G, C, assumptions):

    # retreiving values from assumptions
    (_, m_loc, m_car, axles_on_car, axles_on_loco, num_locos, num_cars, K_cl, fa_cl, 
     _, _, _, _, _, _, _, _) = get_assumption_vals(assumptions)
    
    m_axle_loco = m_loc / axles_on_loco
    m_axle_car = m_car / axles_on_car

    R_r = 0
    train_consist = [(m_loc, m_axle_loco)] * num_locos + [(m_car, m_axle_car)] * num_cars
    for m_cl, m_axle in train_consist:
        term1 = 1.5
        term2 = 16329.34 / m_axle
        term3 = 0.0671 * v
        term4 = (48862.37 * K_cl * fa_cl * (v ** 2)) / m_cl
        term5 = 20 * (G + 0.04 * abs(C))
        resistance = term1 + term2 + term3 + term4 + term5
        R_r +=  m_cl * resistance
    
    return R_r * (4.44822 * 1.10231) / 1000  # Newtons

def totalResistanceForces(velocity_seconds, grade_seconds, curvature_seconds, assumptions):

    resistance_seconds = np.zeros_like(velocity_seconds)
    for i in range(len(velocity_seconds)):
        v = velocity_seconds[i]
        G = grade_seconds[i]
        C = curvature_seconds[i]
        resistance_seconds[i] = resistance_force_single(v, G, C, assumptions)
    return resistance_seconds

def interpret_excel():
    # load data from Excel
    file_path = './uploads/inputData.xlsx'
    velocity_data = pd.read_excel(file_path, sheet_name='Speeds')
    grade_data = pd.read_excel(file_path, sheet_name='Vertical Alignment')
    curvature_data = pd.read_excel(file_path, sheet_name='Horizontal Alignment')

    # interpolation functions - extrapolates the veolicty, grade and curvature from given info
    velocity_interp = interp1d(
        velocity_data['Chainage']*1000, 
        velocity_data['Speed (km/h)']*1000/3600, 
        kind='previous', 
        fill_value='extrapolate')
    
    grade_interp = interp1d(
        grade_data['Chainage']*1000, 
        grade_data['Gradient'], 
        kind='previous', 
        fill_value='extrapolate')
    
    curvature_interp = interp1d(
        curvature_data['Chainage']*1000, 
        curvature_data['Curve Degree'], 
        kind='previous', 
        fill_value='extrapolate')
    

    return velocity_interp, grade_interp, curvature_interp, velocity_data

def get_input(assumptions):

    # retreiving values from assumptions
    (P, _, _, _, _,
    num_locos, _, _, _,
    _, _, _,
    max_decel, brake_buffer, max_tractive_force, max_accel,
    total_mass) = get_assumption_vals(assumptions)

    # constants
    dt = 1.0  # time step (seconds)

    # interpret excel input
    velocity_interp, grade_interp, curvature_interp, velocity_data = interpret_excel()

    # determine total distance
    total_distance = velocity_data['Chainage'].iloc[-1]*1000
   
    # simulation state
    velocity_seconds = [0.0]
    distance_seconds = [0.0]
    time_seconds = [0]

    ## determining speed given power and stationary limitations
    # while the train hasn't reached its destination
    while distance_seconds[-1] < total_distance:
        v = velocity_seconds[-1]
        s = distance_seconds[-1]
        t = time_seconds[-1]

        G = grade_interp(s)
        C = curvature_interp(s)
        v_limit = velocity_interp(s)
        dist_left = total_distance - s

        # Resistance
        F_resist = resistance_force_single(v, G, C, assumptions)

        # Compute how much distance is needed to stop at current speed
        if v > 0:
            stopping_distance = (v ** 2) / (2 * max_decel)
        else:
            stopping_distance = 0

        ## Decide on acceleration

        # limited distance left, decel.
        if dist_left <= 0.1:
            a = -v / dt
            F_trac = 0
        
        # early braking
        elif stopping_distance + brake_buffer >= dist_left:
            required_decel = min(v ** 2 / (2 * max(dist_left, 1)), max_decel)
            a = -required_decel
            F_trac = 0

        # accelerate
        else:
            if v > 0:
                F_trac = min(P * num_locos / v, max_tractive_force)
            else:
                F_trac = max_tractive_force

            F_net = F_trac - F_resist
            a = F_net / total_mass
            a = max(min(a, max_accel), -max_decel)

        # Update kinematics
        v_new = max(0, min(v + a * dt, v_limit))
        s_new = s + v_new * dt

        # Append results
        velocity_seconds.append(v_new)
        distance_seconds.append(s_new)
        time_seconds.append(t + dt)
        
        # Stop exactly at destination
        if s_new >= total_distance and v_new <= 0.1:
            velocity_seconds[-1] = 0
            distance_seconds[-1] = total_distance
            break
        
    distance_seconds = np.array(distance_seconds)
    velocity_seconds = np.array(velocity_seconds)
    time_seconds = np.array(time_seconds)
    grade_seconds = grade_interp(distance_seconds)
    curvature_seconds = curvature_interp(distance_seconds)
    acceleration_seconds = np.diff(velocity_seconds, prepend=velocity_seconds[0])

    return acceleration_seconds, velocity_seconds, grade_seconds, curvature_seconds, distance_seconds

def get_input_segment(start_m, end_m, assumptions):

    # get assumptions values
    (P, _, _, _, _, num_locos, _, _, _, _, _, _,
    max_decel, brake_buffer, max_tractive_force, max_accel,
    total_mass) = get_assumption_vals(assumptions)

    # Constants
    dt = 1.0  # time step (seconds)

    # Load data from Excel
    file_path = './uploads/inputData.xlsx'
    velocity_data = pd.read_excel(file_path, sheet_name='Speeds')
    grade_data = pd.read_excel(file_path, sheet_name='Vertical Alignment')
    curvature_data = pd.read_excel(file_path, sheet_name='Horizontal Alignment')

    # Convert chainage to meters
    velocity_data['Chainage'] = velocity_data['Chainage'] * 1000
    grade_data['Chainage'] = grade_data['Chainage'] * 1000
    curvature_data['Chainage'] = curvature_data['Chainage'] * 1000


    def filter_and_add_boundaries(df, start_m, end_m):
        df_segment = df[(df['Chainage'] >= start_m) & (df['Chainage'] <= end_m)].copy()

        for bound in [start_m, end_m]:
            if bound not in df_segment['Chainage'].values:
                before_rows = df[df['Chainage'] <= bound]
                after_rows = df[df['Chainage'] > bound]

                if not after_rows.empty:
                    val = after_rows.iloc[0, 1]
                elif not before_rows.empty:
                    val = before_rows.iloc[-1, 1]
                else:
                    val = 0.0  # 👈 default fallback (neutral curvature or grade)

                row = [bound] + [val] * (len(df.columns) - 1)
                df_segment = pd.concat(
                    [df_segment, pd.DataFrame([row], columns=df.columns)],
                    ignore_index=True
                )

        return df_segment.sort_values('Chainage').reset_index(drop=True)



    v_data = filter_and_add_boundaries(velocity_data, start_m, end_m)
    g_data = filter_and_add_boundaries(grade_data, start_m, end_m)
    c_data = filter_and_add_boundaries(curvature_data, start_m, end_m)

    # Interpolation functions for the segment
    velocity_interp = interp1d(
        v_data['Chainage'], 
        v_data['Speed (km/h)'] * 1000 / 3600,
        kind='previous', fill_value='extrapolate'
    )
    grade_interp = interp1d(
        g_data['Chainage'], 
        g_data['Gradient'],
        kind='previous', fill_value='extrapolate'
    )
    curvature_interp = interp1d(
        c_data['Chainage'], 
        c_data['Curve Degree'],
        kind='previous', fill_value='extrapolate'
    )

    total_distance = end_m - start_m

    # Initialize simulation arrays
    velocity_seconds = [0.0]
    distance_seconds = [0.0]
    time_seconds = [0]

    epsilon = 0.1
    max_steps = 100000
    steps = 0

    while distance_seconds[-1] < total_distance and steps < max_steps:
        v = velocity_seconds[-1]
        s = distance_seconds[-1]
        t = time_seconds[-1]
        actual_s = start_m + s
        G = grade_interp(actual_s)
        C = curvature_interp(actual_s)
        v_limit = velocity_interp(actual_s)
        dist_left = total_distance - s

        if (C > 15):
            print("this is the curvature")
            print(C)

        F_resist = resistance_force_single(v, G, C, assumptions)

        if v > 0:
            stopping_distance = (v ** 2) / (2 * max_decel)
        else:
            stopping_distance = 0

        if dist_left <= 0.1:
            a = -v / dt
            F_trac = 0
        elif stopping_distance + brake_buffer >= dist_left:
            required_decel = min(v ** 2 / (2 * max(dist_left, 1)), max_decel)
            a = -required_decel
            F_trac = 0
        else:
            if v > 0:
                F_trac = min(P * num_locos / v, max_tractive_force)
            else:
                F_trac = max_tractive_force

            F_net = F_trac - F_resist
            a = F_net / total_mass
            a = max(min(a, max_accel), -max_decel)

        v_new = max(0, min(v + a * dt, v_limit))
        s_new = s + v_new * dt

        # === Append before breaking ===
        velocity_seconds.append(v_new)
        distance_seconds.append(s_new)
        time_seconds.append(t + dt)

        # === Proper break condition with epsilon ===
        if (s_new >= total_distance - epsilon) and (v_new <= 0.1):
            velocity_seconds[-1] = 0
            distance_seconds[-1] = total_distance
            break

        steps += 1
        # Deadlock detection: if velocity has been stuck at 0 for the last 5 steps
        if steps > 10 and all(v == 0 for v in velocity_seconds[-5:]):
            print("🚨 Train is stuck — insufficient tractive force to overcome resistance.")
            print(f"  Chainage range: {start_m}–{end_m}")
            print(f"  At distance: {s:.2f} m | Actual location: {actual_s:.2f} m")
            print(f"  Grade: {G}, Curvature: {C}, Velocity limit: {v_limit}")
            print(f"  Resistance: {F_resist}, Tractive force: {F_trac}")
            break

        

    if steps == max_steps:
        print(f"WARNING: Reached max steps at {s_new:.2f} m — simulation did not complete.")

    distance_seconds = np.array(distance_seconds)
    velocity_seconds = np.array(velocity_seconds)
    time_seconds = np.array(time_seconds)

    actual_distances = start_m + distance_seconds
    grade_seconds = grade_interp(actual_distances)
    curvature_seconds = curvature_interp(actual_distances)
    acceleration_seconds = np.diff(velocity_seconds, prepend=velocity_seconds[0])

        # Save interpolated grade and curvature to Excel for debugging
    debug_df = pd.DataFrame({
        'Actual Distance (m)': actual_distances,
        'Grade': grade_seconds,
        'Curvature (deg)': curvature_seconds
    })

    debug_df.to_excel('./uploads/interpolated_geometry_debug_segment.xlsx', index=False)


    return acceleration_seconds, velocity_seconds, grade_seconds, curvature_seconds, distance_seconds


def tractive_power(acceleration_seconds, resistance_seconds, velocity_seconds, assumptions):
    if not (len(acceleration_seconds) == len(resistance_seconds) == len(velocity_seconds)):
        raise ValueError("Input lists must have the same length")
        
    # get assumptions values
    (_, _, _, _, _,_, _, _, _, train_efficiency, regen_efficiency, regen_buffer, _, _, _, _, total_mass) = get_assumption_vals(assumptions)

    power_output = []
    regen_power = []
    energy_consumption = []

    total_power_output = 0
    total_regen_power = 0
    total_energy_consumption = 0

    for i in range(len(acceleration_seconds)):
        a = acceleration_seconds[i]
        r = resistance_seconds[i]
        v = velocity_seconds[i]

        # Calculate raw power needed
        power = (total_mass * a + r) * v
        power = power / train_efficiency
        power_output.append(power)

        # Estimate regen energy from braking
        if a < -1e-5:  # meaningful braking
            try:
                regen = 1 / math.exp(abs(regen_efficiency / a))  # W
            except OverflowError:
                regen = 0
        else:
            regen = 0

        regen_power.append(regen)

        # Convert to kWh
        power_kWh = power / (1000 * 3600)
        regen_kWh = regen / (1000 * 3600)


        # Case 1: Accelerating — try to use saved regen energy
        if a > 0:
            if regen_buffer > 0:
                used = min(regen_buffer, power_kWh)
                net_energy = power_kWh - used
                regen_buffer -= used
            else:
                net_energy = power_kWh

        # Case 2: Decelerating — only count regen if followed by accel
        elif a < 0:
            next_is_accel = i < len(acceleration_seconds) - 1 and acceleration_seconds[i + 1] > 0
            if next_is_accel:
                net_energy = power_kWh - regen_kWh
            else:
                regen_buffer += regen_kWh  # Save for future
                net_energy = power_kWh

        # Case 3: Constant speed
        else:
            net_energy = power_kWh
        

        # Record energy values
        energy_consumption.append(net_energy)
        total_power_output += power
        total_regen_power += regen
        total_energy_consumption += net_energy

    print(f"⚡ Total net energy (adjusted): {total_energy_consumption:.3f} kWh")
    print(f"♻️ Regen energy banked for future use: {regen_buffer:.3f} kWh")
    print(f"🔄 Segment {i}–{i+1}: regen buffer carried to next = {regen_buffer:.3f} kWh\n")

    print("idk why break")

    return power_output, regen_power, energy_consumption, total_power_output, total_regen_power, total_energy_consumption, regen_buffer


