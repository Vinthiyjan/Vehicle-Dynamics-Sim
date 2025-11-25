import numpy as np

class PIDController:
    """
    Discrete-time PID controller with anti-windup and derivative filtering.
    """
    
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, dt=0.01, 
                 output_limits=(-np.inf, np.inf), anti_windup_limits=None):
        """
        Initialize PID controller.
        
        Parameters:
        -----------
        Kp : float
            Proportional gain
        Ki : float  
            Integral gain
        Kd : float
            Derivative gain
        setpoint : float
            Target value
        dt : float
            Time step for discrete implementation
        output_limits : tuple
            (min, max) output limits
        anti_windup_limits : tuple, optional
            (min, max) limits for integral term to prevent windup
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.dt = dt
        
        self.output_limits = output_limits
        self.anti_windup_limits = anti_windup_limits if anti_windup_limits else output_limits
        
        # Controller state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_measurement = 0.0
        self.previous_output = 0.0
        
    def update(self, measurement, setpoint=None):
        """
        Update controller with new measurement.
        
        Parameters
        -----------
        measurement : float
            Current process value
        setpoint : float, optional
            New setpoint (uses existing if None)
            
        Returns
        --------
        output : float
            Controller output
        """
        if setpoint is not None:
            self.setpoint = setpoint
            
        error = self.setpoint - measurement
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, 
                               self.anti_windup_limits[0] / (self.Ki + 1e-10),
                               self.anti_windup_limits[1] / (self.Ki + 1e-10))
        I = self.Ki * self.integral
        
        # Derivative term (on measurement to avoid derivative kick)
        D = self.Kd * (self.previous_measurement - measurement) / self.dt
        
        # Compute output
        output = P + I + D
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update state
        self.previous_error = error
        self.previous_measurement = measurement
        self.previous_output = output
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_measurement = 0.0
        self.previous_output = 0.0

class CruiseControl:
    """
    Vehicle cruise control system using PID for speed control.
    """
    
    def __init__(self, vehicle_params, Kp=2000.0, Ki=500.0, Kd=100.0, 
                 target_speed=30.0, dt=0.01):
        """
        Initialize cruise control.
        
        Parameters
        -----------
        vehicle_params : dict
            Vehicle parameters for force conversion
        Kp, Ki, Kd : float
            PID gains
        target_speed : float
            Cruise control set speed (m/s)
        dt : float
            Control time step
        """
        self.params = vehicle_params
        self.pid = PIDController(Kp, Ki, Kd, setpoint=target_speed, dt=dt,
                                output_limits=(0.0, np.inf))  # Throttle limits
        
    def compute_throttle(self, current_speed, target_speed=None):
        """
        Compute throttle command for cruise control.
        
        Parameters
        -----------
        current_speed : float
            Current vehicle speed (m/s)
        target_speed : float, optional
            New target speed (uses existing if None)      
    
        Returns
        ------
        throttle : float
            Throttle command [0-1]
        """
        # Convert PID output (force) to throttle
        force_command = self.pid.update(current_speed, target_speed)
        
        # Simple throttle mapping (could be enhanced with inverse model)
        max_force = self.estimate_max_force(current_speed)
        throttle = np.clip(force_command / max(1e-6, max_force), 0.0, 1.0)
        
        return throttle
    
    def estimate_max_force(self, speed):
        """
        Estimate maximum available drive force at current speed.
        Simplified version - in practice would use full engine model.
        """
        # Simple approximation: assume we're in optimal gear
        from models import select_optimal_gear, calculate_engine_rpm, calculate_wheel_force
    
        gear = select_optimal_gear(speed, self.params, self.params['engine_torque'])  
        rpm = calculate_engine_rpm(speed, gear, self.params['i_final'], self.params['R_w'])
        max_torque = self.params['engine_torque'](rpm, 1.0, **self.params.get('engine_map_args', {}))
    
        return calculate_wheel_force(max_torque, gear, self.params['i_final'], 
                                self.params['eta_driveline'], self.params['R_w'])
    
    def reset(self):
        """Reset cruise control"""
        self.pid.reset()
    