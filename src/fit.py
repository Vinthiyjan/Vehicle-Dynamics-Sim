import numpy as np
from scipy.optimize import least_squares, minimize
from models import rhs_longitudinal_pro
from simulate import simulate_drive_cycle, create_gear_schedule, create_pulse_schedule, create_step_schedule

def fit_vehicle_parameters(telemetry_data, p_initial, parameters_to_fit, 
                          bounds=None, throttle_schedule=None, brake_schedule=None,
                          method='least_squares', **kwargs):
    """
    Fit vehicle parameters to telemetry data.
    
    Parameters
    -----------
    telemetry_data : pandas.DataFrame
        Data with 'time' and 'velocity_measured' columns
    p_initial : dict
        Initial parameter guess
    parameters_to_fit : list
        List of parameter names to fit (e.g., ['CdA', 'Cr'])
    bounds : dict, optional
        Bounds for parameters {param_name: (lower, upper)}
    throttle_schedule : callable, optional
        Throttle schedule function
    brake_schedule : callable, optional
        Brake schedule function
    method : str
        Optimization method ('least_squares' or 'minimize')
        
    Returns
    --------
    result : OptimizeResult
        Optimization result
    p_optimized : dict
        Optimized parameters
    """
    
    t_meas = telemetry_data['time'].values
    v_meas = telemetry_data['velocity_measured'].values
    
    # Initial guess vector
    x0 = np.array([p_initial[param] for param in parameters_to_fit])
    
    # Parameter bounds
    if bounds is None:
        bounds = (-np.inf, np.inf)
    else:
        lb = [bounds[param][0] for param in parameters_to_fit]
        ub = [bounds[param][1] for param in parameters_to_fit]
        bounds = (lb, ub)
    
    def residuals(x):
        """Compute residuals between model and measurements"""
        # Update parameters with current values
        p_current = p_initial.copy()
        for i, param in enumerate(parameters_to_fit):
            p_current[param] = x[i]
        
        try:
            # Simulate with current parameters
            sol = simulate_drive_cycle(
                p_current, 
                (t_meas[0], t_meas[-1]), 
                [v_meas[0], 0.0],  # Initial state: [v0, x0]
                throttle_schedule=throttle_schedule,
                brake_schedule=brake_schedule,
                max_step=0.01
            )
            
            # Interpolate simulation to measurement times
            from scipy.interpolate import interp1d
            v_sim = interp1d(sol.t, sol.y[0], kind='linear', 
                           fill_value='extrapolate')(t_meas)
            
            return v_sim - v_meas
            
        except Exception as e:
            # Return large residuals if simulation fails
            print(f"Simulation failed: {e}")
            return np.ones_like(v_meas) * 1e6
    
    # Perform optimization
    if method == 'least_squares':
        result = least_squares(residuals, x0, bounds=bounds, **kwargs)
    else:
        def cost_function(x):
            res = residuals(x)
            return np.sum(res**2)
        
        result = minimize(cost_function, x0, bounds=[bounds[0], bounds[1]], **kwargs)
    
    # Create optimized parameter dictionary
    p_optimized = p_initial.copy()
    for i, param in enumerate(parameters_to_fit):
        p_optimized[param] = result.x[i]
        
    
    return result, p_optimized

def fit_coastdown_parameters(telemetry_data, p_initial, parameters_to_fit=None,
                            coastdown_start_time=None, **kwargs):
    """
    Specialized fitting for coastdown data (throttle=0, brake=0).
    
    Parameters
    -----------
    telemetry_data : pandas.DataFrame
        Coastdown telemetry data
    p_initial : dict
        Initial parameters
    parameters_to_fit : list, optional
        Parameters to fit (default: ['CdA', 'Cr'])
    coastdown_start_time : float, optional
        Time when coastdown starts (auto-detected if None)
    """
    
    if parameters_to_fit is None:
        parameters_to_fit = ['CdA', 'Cr']
    
    # Auto-detect coastdown start if not provided
    if coastdown_start_time is None:
        # Simple heuristic: find when velocity starts decreasing rapidly
        acceleration = np.gradient(telemetry_data['velocity_measured'].values, 
                                 telemetry_data['time'].values)
        coastdown_start_time = telemetry_data['time'].values[np.argmin(acceleration)]
    
    # Create zero-throttle schedule
    def coastdown_throttle(t):
        return 1.0 if coastdown_start_time > t else 0.0  # No throttle during coastdown
    
    return fit_vehicle_parameters(
        telemetry_data, p_initial, parameters_to_fit,
        throttle_schedule=coastdown_throttle,
        brake_schedule=lambda t: 0.0,  # No braking
        **kwargs
    )

def calculate_fit_metrics(telemetry_data, p_optimized,coastdown_start_time = 0.0, throttle_schedule=None, 
                         brake_schedule=None):
    """
    Calculate quality metrics for parameter fit.
    
    Returns
    --------
    metrics : dict
        Dictionary of fit quality metrics
    """
    t_meas = telemetry_data['time'].values
    v_meas = telemetry_data['velocity_measured'].values
    
    def coastdown_throttle(t):
        return 1.0 if coastdown_start_time > t else 0.0
    
    if coastdown_start_time > 0.0: 
        sol = simulate_drive_cycle(
        p_optimized,
        (t_meas[0], t_meas[-1]),
        [v_meas[0], 0.0],
        throttle_schedule=coastdown_throttle,
        brake_schedule=brake_schedule,
        max_step=0.01
    )
    
    else:
        # Simulate with optimized parameters
        sol = simulate_drive_cycle(
        p_optimized,
        (t_meas[0], t_meas[-1]),
        [v_meas[0], 0.0],
        throttle_schedule=throttle_schedule,
        brake_schedule=brake_schedule,
        max_step=0.01
    ) 
            
    
    
    
    # Interpolate to measurement times
    from scipy.interpolate import interp1d
    v_sim = interp1d(sol.t, sol.y[0], kind='linear', fill_value='extrapolate')(t_meas)
    
    # Calculate metrics
    residuals = v_sim - v_meas
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    r_squared = 1 - np.sum(residuals**2) / np.sum((v_meas - np.mean(v_meas))**2)
    
    return {
        'rmse': rmse,
        'mae': mae, 
        'r_squared': r_squared,
        'max_error': np.max(np.abs(residuals)),
        'v_simulated': v_sim,
        'residuals': residuals
    }