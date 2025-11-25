"""
Vehicle Dynamics Model
=====================

A high-fidelity longitudinal vehicle dynamics simulator for motorsport applications.
Models powertrain, aerodynamics, tire grip, and braking systems with physical accuracy.
"""

import numpy as np


# Physical constants
GRAVITY = 9.81  # m/s^2
AIR_DENSITY = 1.225  # kg/m³ at sea level


class VehicleParameters:
    """
    Container for vehicle physical parameters with sensible defaults.
    
    Attributes
    ----------
    CdA : float
        Drag area [m^2]
    CLA : float
        Downforce coefficient x area [m^2]
    MASS : float
        Vehicle mass [kg]
    WHEEL_RADIUS : float
        Effective wheel radius [m]
    FINAL_DRIVE_RATIO : float
        Final drive ratio
    GEAR_RATIOS : list of float
        Gear ratios from 1st to 8th
    DRIVETRAIN_EFFICIENCY : float
        Drivetrain mechanical efficiency [0-1]
    ROLLING_RESISTANCE : float
        Rolling resistance coefficient
    BASE_FRICTION : float
        Base friction coefficient mu_0 for racing slicks
    LOAD_SENSITIVITY : float
        Load sensitivity exponent alpha in tire friction model
    BRAKE_GAIN : float
        Brake torque per unit brake pedal [Nm]
    BRAKE_EFFICIENCY : float
        Brake system efficiency [0-1]
    PEAK_TORQUE_RPM : int
        RPM at which peak torque occurs
    PEAK_TORQUE : float
        Maximum engine torque [Nm]
    REDLINE_RPM : int
        Maximum safe RPM
    IDLE_RPM : int
        Minimum operating RPM
    """
    
    # Aerodynamics
    CdA = 1.1    # Drag area [m^2]
    CLA = 3.5     # Downforce coefficient × area [m^2]
    
    # Chassis
    MASS = 750.0  # Vehicle mass [kg]
    WHEEL_RADIUS = 0.33  # Effective wheel radius [m]
    
    # Drivetrain
    FINAL_DRIVE_RATIO = 3.8
    GEAR_RATIOS = [4.8, 4.0, 3.5, 2.8, 2.4, 2.1, 1.9, 1.7]
    DRIVETRAIN_EFFICIENCY = 0.95
    
    # Tire parameters
    ROLLING_RESISTANCE = 0.015
    BASE_FRICTION = 2.5        # mu_0 for racing slicks
    LOAD_SENSITIVITY = 0.06    # alpha in tire friction model
    
    # Brake system
    BRAKE_GAIN = 12000.0       # Nm per unit brake pedal
    BRAKE_EFFICIENCY = 0.97
    
    # Engine (F1-style)
    PEAK_TORQUE_RPM = 11000
    PEAK_TORQUE = 650          # Nm
    REDLINE_RPM = 15000
    IDLE_RPM = 3500


def create_engine_torque_map(peak_rpm: int = 11000, peak_torque: float = 520, 
                            redline: int = 15000, idle: int = 3500):
    """
    Create a realistic engine torque curve.
    
    Parameters
    ----------
    peak_rpm : int, optional
        RPM at which peak torque occurs, by default 11000
    peak_torque : float, optional
        Maximum torque output in Nm, by default 520
    redline : int, optional
        Maximum safe RPM, by default 15000
    idle : int, optional
        Minimum operating RPM, by default 3500
        
    Returns
    -------
    callable
        Function torque_map(rpm, throttle) -> torque [Nm]
    """
    def torque_map(rpm: float, throttle: float) -> float:
        """
        Calculate engine torque at given RPM and throttle position.
        
        Parameters
        ----------
        rpm : float
            Engine speed in RPM
        throttle : float
            Throttle position [0-1]
            
        Returns
        -------
        float
            Engine torque in Nm
        """
        # Clip RPM to valid operating range
        rpm_clipped = np.clip(rpm, idle, redline)
        
        # Bell-shaped torque curve centered at peak RPM
        width = 0.35 * peak_rpm
        torque_shape = np.exp(-0.5 * ((rpm_clipped - peak_rpm) / width) ** 2)
        
        # Scale by throttle and peak torque
        return throttle * peak_torque * torque_shape
    
    return torque_map


def calculate_engine_rpm(vehicle_speed: float, gear_ratio: float, 
                        final_drive_ratio: float, wheel_radius: float) -> float:
    """
    Convert vehicle speed to engine RPM.
    
    Parameters
    ----------
    vehicle_speed : float
        Current vehicle speed in m/s
    gear_ratio : float
        Current gear ratio
    final_drive_ratio : float
        Final drive ratio
    wheel_radius : float
        Effective wheel radius in m
        
    Returns
    -------
    float
        Engine speed in RPM
    """
    effective_speed = vehicle_speed * 0.98  # 2% slip under acceleration
    
    wheel_angular_velocity = effective_speed / wheel_radius  # rad/s
    engine_angular_velocity = wheel_angular_velocity * gear_ratio * final_drive_ratio  # rad/s
    engine_rpm = (engine_angular_velocity * 60) / (2 * np.pi)  # RPM
    return engine_rpm


def calculate_wheel_force(engine_torque: float, gear_ratio: float, 
                         final_drive_ratio: float, efficiency: float, 
                         wheel_radius: float) -> float:
    """
    Convert engine torque to wheel force.
    
    Parameters
    ----------
    engine_torque : float
        Torque at engine in Nm
    gear_ratio : float
        Current gear ratio
    final_drive_ratio : float
        Final drive ratio
    efficiency : float
        Drivetrain efficiency [0-1]
    wheel_radius : float
        Effective wheel radius in m
        
    Returns
    -------
    float
        Force at wheel contact patch in N
    """
    wheel_torque = engine_torque * gear_ratio * final_drive_ratio * efficiency
    return wheel_torque / max(wheel_radius, 1e-6)




def select_optimal_gear(vehicle_speed: float, parameters: dict, 
                       engine_torque_func: callable) -> float:
    """
    Select gear that maximizes wheel force at current speed.
    """
    if vehicle_speed < 0.1:
        return parameters['gears'][0]
    
    best_gear = parameters['gears'][0]
    max_force = -np.inf
    
    for gear_ratio in parameters['gears']:
        # Calculate engine RPM in this gear
        rpm = calculate_engine_rpm(
            vehicle_speed, gear_ratio, 
            parameters['i_final'], parameters['R_w']
        )
        
        # Skip this gear if it over-revs
        if rpm > VehicleParameters.REDLINE_RPM:
            continue  
        
        # Get maximum possible torque (full throttle)
        max_torque = engine_torque_func(rpm, 1.0)
        
        # Calculate wheel force
        wheel_force = calculate_wheel_force(
            max_torque, gear_ratio, parameters['i_final'],
            parameters['eta_driveline'], parameters['R_w']
        )
        
        # Select gear with highest force
        if wheel_force > max_force:
            max_force = wheel_force
            best_gear = gear_ratio
    
    # If all gears over-rev, use the highest gear (lowest ratio)
    if max_force == -np.inf:  # No valid gear found
        return parameters['gears'][-1]  # Return top gear
            
    return best_gear

def calculate_tire_friction(normal_force: float, parameters: dict) -> float:
    """
    Calculate tire friction coefficient with load sensitivity.
    
    Parameters
    ----------
    normal_force : float
        Total vertical force on tires in N
    parameters : dict
        Vehicle parameters dictionary
        
    Returns
    -------
    float
        Friction coefficient mu
    """
    base_friction = parameters['mu_0']
    load_sensitivity = parameters.get('alpha_mu', 0.08)
    
    # Friction decreases slightly with increased normal load
    static_load = parameters['m'] * GRAVITY
    return base_friction * (normal_force / static_load) ** (-load_sensitivity)


def calculate_aerodynamic_forces(air_speed: float, parameters: dict) -> tuple:
    """
    Calculate aerodynamic drag and downforce.
    
    Parameters
    ----------
    air_speed : float
        Air-relative speed in m/s
    parameters : dict
        Vehicle parameters dictionary
        
    Returns
    -------
    tuple
        (drag_force, downforce) in Newtons
    """
    dynamic_pressure = 0.5 * parameters['rho'] * air_speed ** 2
    drag_force = dynamic_pressure * parameters['CdA']
    downforce = dynamic_pressure * parameters['CLA']
    
    return drag_force, downforce


def longitudinal_dynamics(t: float, state: np.ndarray, parameters: dict, 
                         controls: dict, environment: dict) -> np.ndarray:
    """
    Main vehicle dynamics function - calculates acceleration from forces.
    
    Parameters
    ----------
    t : float
        Time in seconds (for time-dependent controls)
    state : np.ndarray
        Vehicle state [velocity, position] in [m/s, m]
    parameters : dict
        Vehicle physical parameters
    controls : dict
        Control inputs with keys:
        - 'throttle': Throttle position [0-1]
        - 'brake': Brake position [0-1] 
        - 'gear': Gear ratio (None for automatic)
    environment : dict
        Environmental conditions with keys:
        - 'road_grade': Road incline in radians
        - 'wind': Headwind/tailwind speed in m/s
        
    Returns
    -------
    np.ndarray
        State derivative [acceleration, velocity] in [m/s^2, m/s]
    """
    velocity, position = state
    velocity = float(velocity)  # Ensure scalar
    
    # Unpack controls and environment
    throttle = controls.get('throttle', 0.0)
    brake = controls.get('brake', 0.0)
    gear = controls.get('gear', None)
    road_grade = environment.get('road_grade', 0.0)
    wind = environment.get('wind', 0.0)
    
    # Aerodynamic forces 
    air_speed = abs(velocity - wind)
    drag_force, downforce = calculate_aerodynamic_forces(air_speed, parameters)
    
    # Normal load calculation 
    weight = parameters['m'] * GRAVITY
    normal_force = weight + downforce
    rolling_resistance = parameters['Cr'] * normal_force
    
    # Tire grip limit
    friction_coefficient = calculate_tire_friction(normal_force, parameters)
    max_tire_force = friction_coefficient * normal_force
    
    # Powertrain force
    if gear is None:  # Automatic gear selection
        gear_ratio = select_optimal_gear(
            max(velocity, 0.0), parameters, parameters['engine_torque']
        )
    else:
        gear_ratio = gear
        
    # Engine torque and wheel force
    rpm = calculate_engine_rpm(
        max(velocity, 0.0), gear_ratio, 
        parameters['i_final'], parameters['R_w']
    )
    rpm_for_torque = np.clip(rpm, VehicleParameters.IDLE_RPM, VehicleParameters.REDLINE_RPM)
    
    engine_torque = parameters['engine_torque'](
        rpm_for_torque, throttle, **parameters.get('engine_map_args', {})
    )
    drive_force = calculate_wheel_force(
        engine_torque, gear_ratio, parameters['i_final'],
        parameters['eta_driveline'], parameters['R_w']
    )
    
    # Limit drive force by available tire grip
    drive_force = np.clip(drive_force, 0.0, max_tire_force)
    
    # Braking force
    brake_torque = parameters['Kb'] * np.clip(brake, 0.0, 1.0) * parameters.get('eta_brake', 1.0)
    brake_force = np.clip(brake_torque / max(parameters['R_w'], 1e-6), 0.0, max_tire_force)
    
    # Net longitudinal force
    net_force = np.clip(drive_force - brake_force, -max_tire_force, max_tire_force)
    
    # Grade resistance
    grade_force = parameters['m'] * GRAVITY * np.sin(road_grade)
    
    # Final acceleration calculation 
    total_resistance = drag_force + rolling_resistance + grade_force
    acceleration = (net_force - total_resistance) / parameters['m']
    
    return np.array([acceleration, velocity], dtype=float)


def rhs_longitudinal_pro(t: float, state: np.ndarray, parameters: dict, 
                        throttle: float = 0.0, brake: float = 0.0, 
                        gear: float = None, road_grade: float = 0.0, 
                        wind: float = 0.0) -> np.ndarray:
    """
    Legacy interface for backward compatibility.
    
    Parameters
    ----------
    t : float
        Time in seconds
    state : np.ndarray
        Vehicle state [velocity, position] in [m/s, m]
    parameters : dict
        Vehicle physical parameters
    throttle : float, optional
        Throttle position [0-1], by default 0.0
    brake : float, optional
        Brake position [0-1], by default 0.0
    gear : float, optional
        Gear ratio, by default None (automatic)
    road_grade : float, optional
        Road incline in radians, by default 0.0
    wind : float, optional
        Headwind/tailwind speed in m/s, by default 0.0
        
    Returns
    -------
    np.ndarray
        State derivative [acceleration, velocity] in [m/s^2, m/s]
    """
    controls = {'throttle': throttle, 'brake': brake, 'gear': gear}
    environment = {'road_grade': road_grade, 'wind': wind}
    
    return longitudinal_dynamics(t, state, parameters, controls, environment)


def create_f1_parameters() -> dict:
    """
    Create a parameter set for a typical Formula 1 car.
    
    Returns
    -------
    dict
        Complete vehicle parameters dictionary for F1 simulation
        
    Examples
    --------
    >>> f1_params = create_f1_parameters()
    >>> print(f"F1 mass: {f1_params['m']} kg")
    >>> print(f"F1 gear ratios: {f1_params['gears']}")
    """
    return {
        'rho': AIR_DENSITY,
        'm': VehicleParameters.MASS,
        'CdA': VehicleParameters.CdA,
        'CLA': VehicleParameters.CLA,
        'Cr': VehicleParameters.ROLLING_RESISTANCE,
        'R_w': VehicleParameters.WHEEL_RADIUS,
        'eta_driveline': VehicleParameters.DRIVETRAIN_EFFICIENCY,
        'i_final': VehicleParameters.FINAL_DRIVE_RATIO,
        'gears': VehicleParameters.GEAR_RATIOS,
        'engine_torque': create_engine_torque_map(),
        'mu_0': VehicleParameters.BASE_FRICTION,
        'alpha_mu': VehicleParameters.LOAD_SENSITIVITY,
        'Kb': VehicleParameters.BRAKE_GAIN,
        'eta_brake': VehicleParameters.BRAKE_EFFICIENCY
    }