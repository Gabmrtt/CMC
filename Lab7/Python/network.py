"""Oscillator network ODE"""

import numpy as np

from scipy.integrate import ode
from robot_parameters import RobotParameters


def network_ode(_time, state, parameters):
    """Network_ODE

    Parameters
    ----------
    _time: <float>
        Time
    state: <np.array>
        ODE states at time _time
    parameters: <RobotParameters>
        Instance of RobotParameters

    Return
    ------
    :<np.array>
        Returns derivative of state (phases and amplitudes)
    """
    phases = state[:parameters.n_oscillators]
    amplitudes = state[parameters.n_oscillators:2*parameters.n_oscillators]
    
    dphasedt = np.zeros(len(phases))
    dampdt = np.zeros(len(phases))
    
    for i in range(parameters.n_oscillators_body):

        inter_sum = [amplitudes[i+a] * parameters.coupling_weights[i,i+a] * np.sin(phases[i+a] - phases[i] - parameters.phase_bias[i,i+a]) if (i+a>=0 and i+a<=20) else 0 for a in [-10,-1,1,10] ]   # be careful with coupling weights [10,9] and [9,10] !!
        k = parameters.n_oscillators_body + i//5
        inter_sum.append(amplitudes[k] *  parameters.coupling_weights[i,k] * np.sin(phases[k] - phases[i] - parameters.phase_bias[i,k]))
        dphasedt[i] = 2 * np.pi * parameters.freqs[i] + np.sum(np.array( inter_sum ))
        dampdt[i] = 20 * ( parameters.nominal_amplitudes[i] - amplitudes[i] ) 
        
    for j in range(parameters.n_oscillators_legs):
        j = j + parameters.n_oscillators_body
        inter_sum = []
        for i in range(parameters.n_oscillators_legs):
            i = i + parameters.n_oscillators_body
            inter_sum.append( amplitudes[i] * parameters.coupling_weights[j,i] * np.sin(phases[i] - phases[j] - parameters.phase_bias[j,i]))
        dphasedt[j] = 2 * np.pi * parameters.freqs[j] + np.sum(np.array( inter_sum ))
        dampdt[j] = 20 * ( parameters.nominal_amplitudes[j] - amplitudes[j] )
        
    return np.concatenate((dphasedt, dampdt))


def motor_output(phases, amplitudes, iteration=None):
    """Motor output.

    Parameters
    ----------
    phases: <np.array>
        Phases of the oscillator
    amplitudes: <np.array>
        Amplitudes of the oscillator

    Returns
    -------
    : <np.array>
        Motor outputs for joint in the system.
    """
    outputs = np.zeros(int((len(phases)-4)/2)+4)    # to change
    
    for i in range(len(outputs)-4):
        outputs[i] = amplitudes[i] * (1 + np.cos(phases[i])) - amplitudes[i+10] * (1 + np.cos(phases[i+10]))
        
    for i in range(4):
        #outputs[10 + i] = np.cos(phases[len(phases)- 4 + i])
        outputs[10 + i] = phases[len(phases)- 4 + i]
        
        
    return outputs


class RobotState(np.ndarray):
    """Robot state"""

    def __init__(self, *_0, **_1):
        super(RobotState, self).__init__()
        self[:] = 0.0

    @classmethod
    def salamandra_robotica_2(cls, n_iterations):
        """State of Salamandra robotica 2"""
        shape = (n_iterations, 2*24)
        return cls(
            shape,
            dtype=np.float64,
            buffer=np.zeros(shape)
        )

    def phases(self, iteration=None):
        """Oscillator phases"""
        return self[iteration, :24] if iteration is not None else self[:, :24]

    def set_phases(self, iteration, value):
        """Set phases"""
        self[iteration, :24] = value

    def set_phases_left(self, iteration, value):
        """Set body phases on left side"""
        self[iteration, :10] = value

    def set_phases_right(self, iteration, value):
        """Set body phases on right side"""
        self[iteration, 10:20] = value

    def set_phases_legs(self, iteration, value):
        """Set leg phases"""
        self[iteration, 20:24] = value

    def amplitudes(self, iteration=None):
        """Oscillator amplitudes"""
        return self[iteration, 24:] if iteration is not None else self[:, 24:]

    def set_amplitudes(self, iteration, value):
        """Set amplitudes"""
        self[iteration, 24:] = value


class SalamandraNetwork:
    """Salamandra oscillator network"""

    def __init__(self, timestep, parameters, n_iterations):
        super(SalamandraNetwork, self).__init__()
        # States
        self.state = RobotState.salamandra_robotica_2(n_iterations)
        # Parameters
        self.parameters = RobotParameters(parameters)
        # Set initial state
        # Replace your oscillator phases here
        self.state.set_phases(
            iteration=0,
            value=1e-4*np.random.ranf(self.parameters.n_oscillators),
        )
        # Set solver
        self.solver = ode(f=network_ode)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.state[0], t=0.0)

    def step(self, iteration, time, timestep):
        """Step"""
        # self.state += self.integrate(self.state, self.parameters)
        self.solver.set_f_params(self.parameters)
        self.state[iteration+1, :] = self.solver.integrate(time+timestep)

    def outputs(self, iteration=None):
        """Oscillator outputs"""
        # Implement equation here

    def get_motor_position_output(self, iteration=None):
        """Get motor position"""
        return motor_output(
            self.state.phases(iteration=iteration),
            self.state.amplitudes(iteration=iteration),
            iteration=iteration,
        )

