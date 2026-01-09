import numpy as np

def get_assumption_vals(assumptions):
    m_loc = assumptions["m_loc"]
    m_car = assumptions["m_car"]
    axles_on_car = assumptions["axles_on_car"]
    axles_on_loco = assumptions["axles_on_loco"]
    num_locos = assumptions["num_locos"]
    num_cars = assumptions["num_cars"]
    K_cl = assumptions["K_cl"]
    fa_cl = assumptions["fa_cl"]

    return (
        m_loc, m_car, axles_on_car, axles_on_loco,
        num_locos, num_cars, K_cl, fa_cl
    )


def resistance_force_single(v, G, C, assumptions):
    # Retrieve values from assumptions
    (m_loc, m_car, axles_on_car, axles_on_loco,
     num_locos, num_cars, K_cl, fa_cl) = get_assumption_vals(assumptions)

    m_axle_loco = m_loc / axles_on_loco
    m_axle_car = m_car / axles_on_car

    R_r = 0
    train_consist = [(m_loc, m_axle_loco)] * num_locos + [(m_car, m_axle_car)] * num_cars

    for m_cl, m_axle in train_consist:
        term1 = 1.5
        term2 = 16329.34 / m_axle
        term3 = 0.0671 * v
        term4 = (48862.37 * K_cl * fa_cl * v**2) / m_cl
        term5 = 20 * (G + 0.04 * abs(C))
        resistance = term1 + term2 + term3 + term4 + term5
        R_r += m_cl * resistance

    return R_r * (4.44822 * 1.10231) / 1000  # Newtons


def total_resistance_over_time(velocity_array, grade_array, curvature_array, assumptions):
    """
    Returns a NumPy array of resistance values [N] for each time step.
    """
    resistance_array = np.zeros_like(velocity_array)
    for i in range(len(velocity_array)):
        v = velocity_array[i]
        G = grade_array[i]
        C = curvature_array[i]
        resistance_array[i] = resistance_force_single(v, G, C, assumptions)
    return resistance_array
