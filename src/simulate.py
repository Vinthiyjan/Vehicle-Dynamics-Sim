import numpy as np
from scipy.integrate import solve_ivp
from models import rhs_longitudinal_pro

def simulate_drive_cycle(p, t_span, y0, throttle_schedule=None, brake_schedule=None, 
                        gear_schedule=None, road_grade=0.0, wind=0.0, max_step=0.02):
    """
    Simulate vehicle dynamics over a drive cycle.
    
    Parameters
    -----------
    p : dict
        Vehicle parameters
    t_span : tuple
        (t_start, t_end) time span for simulation
    y0 : array_like
        Initial state [v0, x0]
    throttle_schedule : callable, optional
        Function that returns throttle [0-1] given time
    brake_schedule : callable, optional  
        Function that returns brake [0-1] given time
    gear_schedule : callable, optional
        Function that returns gear ratio given time (None for auto)
    road_grade : float, optional
        Road incline in radians
    wind : float, optional
        Headwind/tailwind speed (m/s)
    max_step : float, optional
        Maximum integration step size
    
    Returns
    --------
    sol : OdeSolution
        Solution object from solve_ivp
    """
    
    def vehicle_dynamics(t, y):
        # Get control inputs at current time
        throttle = throttle_schedule(t) if throttle_schedule else 0.0
        brake = brake_schedule(t) if brake_schedule else 0.0
        gear = gear_schedule(t) if gear_schedule else None
        
        return rhs_longitudinal_pro(t, y, p, throttle, brake, gear, road_grade, wind)
    
    # Solve ODE
    sol = solve_ivp(vehicle_dynamics, t_span, y0, method='RK45', 
                   max_step=max_step, rtol=1e-6, atol=1e-8)
    
    return sol

def create_step_schedule(step_time, pre_value=0.0, post_value=1.0):
    """Create a step function schedule"""
    return lambda t: pre_value if t < step_time else post_value

def create_pulse_schedule(start_time, end_time, off_value=0.0, on_value=1.0):
    """Create a pulse function schedule"""
    return lambda t: on_value if start_time <= t < end_time else off_value

def create_gear_schedule(shift_times, gears):
    """Create a gear shift schedule"""
    def gear_func(t):
        for i, shift_time in enumerate(shift_times):
            if t < shift_time:
                return gears[i]
        return gears[-1]
    return gear_func

def simulation_to_dataframe(sol, include_controls=True, control_schedules=None):
    """
    Convert simulation solution to pandas DataFrame with additional computed values.
    """
    try:
        import pandas as pd
    except ImportError:
        print("Pandas not available, returning NumPy arrays")
        return sol.t, sol.y
    
    t = sol.t
    v = sol.y[0]  # velocity
    x = sol.y[1]  # position
    
    # Compute acceleration (numerical differentiation)
    acceleration = np.gradient(v, t)
    
    data = {
        'time': t,
        'velocity': v,
        'position': x, 
        'acceleration': acceleration
    }
    
    # Add control inputs if requested
    if include_controls and control_schedules:
        for name, schedule in control_schedules.items():
            data[name] = [schedule(t_i) for t_i in t]
    
    return pd.DataFrame(data)

def generate_synthetic_telemetry(sol, noise_std=0.3, sample_rate=20):
    """
    Generate synthetic telemetry data from clean simulation.
    
    Parameters
    -----------
    sol : OdeSolution
        Clean simulation solution
    noise_std : float
        Standard deviation of Gaussian noise (m/s)
    sample_rate : float
        Sampling frequency (Hz)
    
    Returns
    --------
    telemetry_df : pandas.DataFrame
        Noisy telemetry data
    """
    
    try:
        import pandas as pd
    except ImportError:
        print("Pandas required for synthetic telemetry")
        return None
    
    # Sample the solution at regular intervals
    t_max = sol.t[-1]
    t_meas = np.arange(0, t_max, 1/sample_rate)
    
    # Interpolate solution at measurement times
    from scipy.interpolate import interp1d
    v_clean = interp1d(sol.t, sol.y[0], kind='linear', fill_value='extrapolate')(t_meas)
    x_clean = interp1d(sol.t, sol.y[1], kind='linear', fill_value='extrapolate')(t_meas)
    
    # Add noise to velocity measurements
    v_noisy = v_clean + np.random.normal(0, noise_std, len(v_clean))
    
    # Compute noisy acceleration
    acceleration_noisy = np.gradient(v_noisy, t_meas)
    
    return pd.DataFrame({
        'time': t_meas,
        'velocity_measured': v_noisy,
        'velocity_clean': v_clean,
        'position': x_clean,
        'acceleration_measured': acceleration_noisy
    })